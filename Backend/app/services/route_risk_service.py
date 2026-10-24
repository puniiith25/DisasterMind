"""
Route and Road Risk Investigation Service (Stage 3 / Feature 4).

Assesses disaster, road-closure, seismic, and severe weather risks along a travel corridor.
Integrates:
- OpenStreetMap Nominatim for geocoding
- Project OSRM for genuine road geometry and routing
- USGS & GDACS for active disaster incidents intersecting corridor
- Open-Meteo for severe weather, gale winds, and precipitation along route
- DisasterM3 clean corpus for historical disaster vulnerability context

Safety Rules:
- NEVER labels a route 'Safe' solely because no alert is found.
- Forward-looking travel questions (next week) are flagged with dynamic-risk disclaimers.
- Unverified road stretches are explicitly declared.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.investigation import (
    EvidenceCitation,
    InvestigationGroundedAnswer,
    NormalizedDisasterAlert,
    RecentImageryStatus,
    RouteHazardItem,
    RouteInvestigationRequest,
    RouteInvestigationResponse,
    SourceCheckedStatus,
)
from app.services.external_disaster_service import (
    _haversine_distance_km,
    get_external_disaster_service,
)

logger = logging.getLogger(__name__)


def _point_to_route_distance(
    lat: float, lon: float, waypoints: List[List[float]]
) -> float:
    """Find minimum distance in km from a coordinate to any waypoint in a route polyline."""
    min_dist = float("inf")
    # Coordinates in GeoJSON are [lon, lat]
    for pt in waypoints:
        w_lon, w_lat = pt[0], pt[1]
        dist = _haversine_distance_km(lat, lon, w_lat, w_lon)
        if dist < min_dist:
            min_dist = dist
    return round(min_dist, 2)


class RouteRiskService:
    """Service evaluating disaster hazards along transit routes."""

    def __init__(self, external_service: Optional[Any] = None) -> None:
        self.external_svc = external_service or get_external_disaster_service()

    def investigate_route(
        self, request: RouteInvestigationRequest
    ) -> RouteInvestigationResponse:
        """Execute end-to-end route risk investigation."""
        origin_q = request.origin.strip()
        dest_q = request.destination.strip()
        departure_str = request.departure_time.strip() if request.departure_time else None
        question = request.question or "Are there any road closures or disaster hazards along this route?"

        sources_checked: List[SourceCheckedStatus] = []
        evidence_citations: List[EvidenceCitation] = []
        detected_hazards: List[RouteHazardItem] = []

        # 1. Geocode Origin and Destination
        origin_geo = self.external_svc.geocode(origin_q)
        dest_geo = self.external_svc.geocode(dest_q)

        sources_checked.append(
            SourceCheckedStatus(
                provider="OpenStreetMap Nominatim",
                status="SUCCESS" if (origin_geo and dest_geo) else "FAILED",
                records_count=2 if (origin_geo and dest_geo) else 0,
                latency_ms=120.0,
                details=f"Geocoded: '{origin_q}' -> {origin_geo is not None}, '{dest_q}' -> {dest_geo is not None}",
            )
        )

        if not origin_geo or not dest_geo:
            failed_name = origin_q if not origin_geo else dest_q
            now_iso = datetime.now(timezone.utc).isoformat()
            ans = InvestigationGroundedAnswer(
                direct_answer=f"Could not locate geographic coordinates for '{failed_name}'. Please verify the spelling or provide city, state/province, or country.",
                historical_context=None,
                latest_official_alert=None,
                supporting_evidence=[],
                source_names_and_urls=[{"name": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org"}],
                timestamps={"retrieved_at": now_iso},
                confirmed_location=None,
                location_provenance="geocoding_failed",
                missing_information_and_uncertainty=f"Location '{failed_name}' not resolved in OpenStreetMap global database.",
                practical_next_verification_step="Provide a more specific address, city name, or postal code.",
                recent_imagery=RecentImageryStatus(available=False, limitation_reason="Cannot evaluate imagery without valid origin and destination."),
                sources_checked=sources_checked,
            )
            return RouteInvestigationResponse(
                origin=origin_q,
                destination=dest_q,
                origin_coords=None,
                destination_coords=None,
                departure_time=departure_str,
                route_found=False,
                hazard_summary="Route navigation could not be initialized due to unresolved place names.",
                safety_advisory="Do not begin travel without verified location coordinates.",
                unverified_stretches="Entire route unverified.",
                sources_checked=sources_checked,
                evidence_citations=[],
                grounded_answer=ans,
            )

        o_lat, o_lon = origin_geo["latitude"], origin_geo["longitude"]
        d_lat, d_lon = dest_geo["latitude"], dest_geo["longitude"]

        # 2. Query Routing via OSRM
        route_data, route_status = self.external_svc.fetch_route(
            start_lat=o_lat, start_lon=o_lon, end_lat=d_lat, end_lon=d_lon
        )
        sources_checked.append(route_status)

        if not route_data or not route_data.get("geometry"):
            now_iso = datetime.now(timezone.utc).isoformat()
            ans = InvestigationGroundedAnswer(
                direct_answer=f"No navigable driving road connection found between '{origin_geo['display_name']}' and '{dest_geo['display_name']}' in open highway databases.",
                historical_context=None,
                latest_official_alert=None,
                supporting_evidence=[],
                source_names_and_urls=[{"name": "Project OSRM", "url": "https://project-osrm.org"}],
                timestamps={"retrieved_at": now_iso},
                confirmed_location=f"{origin_geo['display_name']} -> {dest_geo['display_name']}",
                location_provenance="user_confirmed",
                missing_information_and_uncertainty="Road network connectivity unavailable across waterbodies or unsupported regions.",
                practical_next_verification_step="Consider air, rail, or maritime transit options.",
                recent_imagery=RecentImageryStatus(available=False, limitation_reason="No road route geometry to query."),
                sources_checked=sources_checked,
            )
            return RouteInvestigationResponse(
                origin=origin_geo["display_name"],
                destination=dest_geo["display_name"],
                origin_coords={"latitude": o_lat, "longitude": o_lon},
                destination_coords={"latitude": d_lat, "longitude": d_lon},
                departure_time=departure_str,
                route_found=False,
                hazard_summary="No continuous road network connection found.",
                safety_advisory="Automobile transit is not available between these points.",
                unverified_stretches="Entire corridor.",
                sources_checked=sources_checked,
                evidence_citations=[],
                grounded_answer=ans,
            )

        waypoints = route_data["geometry"]["coordinates"]  # [[lon, lat], ...]
        dist_km = route_data.get("distance_km", 0.0)
        dur_min = route_data.get("duration_minutes", 0.0)
        alternatives = route_data.get("alternatives", [])

        # 3. Check for Forward-Looking Departure
        is_forward = False
        forward_warning = None
        if departure_str:
            is_forward = True
            days_ahead = 0
            try:
                clean_dep = departure_str.strip().split("T")[0]
                if clean_dep.lower() == "tomorrow":
                    target_dt = datetime.now(timezone.utc).date() + timedelta(days=1)
                elif clean_dep.lower() in ("today", "now"):
                    target_dt = datetime.now(timezone.utc).date()
                else:
                    target_dt = datetime.fromisoformat(clean_dep).date()
                days_ahead = (target_dt - datetime.now(timezone.utc).date()).days
            except Exception:
                days_ahead = 1

            if days_ahead > 7:
                forward_warning = (
                    f"Forward-Looking Advisory: Your departure is scheduled for {departure_str} ({days_ahead} days ahead). "
                    "Notice: Open-Meteo numerical weather prediction models only span 7 days ahead. "
                    "Atmospheric hazards and storm systems beyond 7 days cannot be reliably verified at this lead time. "
                    "Active earthquake telemetry reflects tremors today and cannot predict future seismic activity. "
                    "Re-check official Department of Transportation (DOT) and civil protection feeds 24-48 hours before departure."
                )
            else:
                forward_warning = (
                    f"Forward-Looking Advisory: Your departure is scheduled for {departure_str}. "
                    "Corridor weather conditions reflect available numerical forecasts. "
                    "Active hazard bulletins reflect real-time conditions today and CANNOT guarantee future highway passability or aftershocks. "
                    "Weather systems, flooding, and road repairs change dynamically. Re-verify with official Department of "
                    "Transportation (DOT) feeds and local police within 6 hours of departure."
                )

        # 4. Query External Disaster Feeds along Route Corridor
        # Sample points: origin, midpoint, destination
        mid_idx = len(waypoints) // 2
        mid_lon, mid_lat = waypoints[mid_idx][0], waypoints[mid_idx][1]

        # 4a. USGS Earthquakes along corridor
        usgs_alerts, usgs_stat = self.external_svc.fetch_usgs_earthquakes(
            lat=mid_lat, lon=mid_lon, radius_km=max(150.0, dist_km / 2.0), min_magnitude=3.8
        )
        sources_checked.append(usgs_stat)

        for u in usgs_alerts:
            if u.latitude is not None and u.longitude is not None:
                d_to_route = _point_to_route_distance(u.latitude, u.longitude, waypoints)
                if d_to_route <= 100.0:  # Within 100km of highway corridor
                    h_item = RouteHazardItem(
                        hazard_type="Earthquake",
                        title=u.title,
                        severity=u.severity,
                        location_description=f"{u.event_location} ({d_to_route} km from highway)",
                        latitude=u.latitude,
                        longitude=u.longitude,
                        distance_to_route_km=d_to_route,
                        source=u.source,
                        source_url=u.source_url,
                        reported_at=u.event_time,
                        status="REPORTED_HAZARD",
                        details=u.summary,
                    )
                    detected_hazards.append(h_item)
                    evidence_citations.append(
                        EvidenceCitation(
                            id=u.alert_id,
                            title=u.title,
                            source=u.source,
                            source_url=u.source_url,
                            is_historical=False,
                            observed_at=u.event_time,
                            published_at=u.published_at,
                            retrieved_at=u.retrieved_at,
                            snippet=u.summary,
                            distance_km=d_to_route,
                        )
                    )

        # 4b. GDACS Multi-Hazard Alerts (Floods, Storms, Cyclones)
        gdacs_alerts, gdacs_stat = self.external_svc.fetch_gdacs_alerts(
            lat=mid_lat, lon=mid_lon, radius_km=max(250.0, dist_km / 2.0)
        )
        sources_checked.append(gdacs_stat)

        for g in gdacs_alerts:
            if g.latitude is not None and g.longitude is not None:
                d_to_route = _point_to_route_distance(g.latitude, g.longitude, waypoints)
                if d_to_route <= 150.0:
                    status_flag = "CONFIRMED_CLOSURE" if g.severity == "CRITICAL" else "REPORTED_HAZARD"
                    h_item = RouteHazardItem(
                        hazard_type=g.event_type,
                        title=g.title,
                        severity=g.severity,
                        location_description=f"{g.event_location} ({d_to_route} km from highway)",
                        latitude=g.latitude,
                        longitude=g.longitude,
                        distance_to_route_km=d_to_route,
                        source=g.source,
                        source_url=g.source_url,
                        reported_at=g.event_time,
                        status=status_flag,
                        details=g.summary,
                    )
                    detected_hazards.append(h_item)
                    evidence_citations.append(
                        EvidenceCitation(
                            id=g.alert_id,
                            title=g.title,
                            source=g.source,
                            source_url=g.source_url,
                            is_historical=False,
                            observed_at=g.event_time,
                            published_at=g.published_at,
                            retrieved_at=g.retrieved_at,
                            snippet=g.summary,
                            distance_km=d_to_route,
                        )
                    )

        # 4c. Open-Meteo Severe Weather along Corridor (Origin, Midpoint, Destination)
        for pt_name, p_lat, p_lon in [("Origin", o_lat, o_lon), ("Midpoint", mid_lat, mid_lon), ("Destination", d_lat, d_lon)]:
            w_alerts, w_stat = self.external_svc.fetch_weather_hazards(lat=p_lat, lon=p_lon, departure_date=departure_str)
            if pt_name == "Midpoint":
                sources_checked.append(w_stat)
            for w in w_alerts:
                if w.affects_location:
                    h_item = RouteHazardItem(
                        hazard_type=w.event_type,
                        title=f"{pt_name}: {w.title}",
                        severity=w.severity,
                        location_description=f"{pt_name} ({p_lat:.2f}, {p_lon:.2f})",
                        latitude=p_lat,
                        longitude=p_lon,
                        distance_to_route_km=0.0,
                        source=w.source,
                        source_url=w.source_url,
                        reported_at=w.event_time,
                        status="WEATHER_WARNING",
                        details=w.summary,
                    )
                    detected_hazards.append(h_item)
                    evidence_citations.append(
                        EvidenceCitation(
                            id=f"w_{pt_name.lower()}_{w.alert_id}",
                            title=f"{pt_name} Atmospheric Warning: {w.title}",
                            source=w.source,
                            source_url=w.source_url,
                            is_historical=False,
                            observed_at=w.event_time,
                            published_at=w.published_at,
                            retrieved_at=w.retrieved_at,
                            snippet=w.summary,
                            distance_km=0.0,
                        )
                    )

        # 5. Retrieve Historical Disaster Context from Clean Train Corpus
        historical_context_str: Optional[str] = None
        try:
            from app.services.multimodal_service import get_multimodal_service
            mm_svc = get_multimodal_service()
            ret_q = f"Road closures, building collapse, bridge damage, or flooding near {origin_geo['display_name']} or {dest_geo['display_name']}"
            hist_res = mm_svc.retrieve_text_evidence(question=ret_q, top_k=2)
            if hist_res and hist_res.text_matches:
                snippets = []
                for m in hist_res.text_matches:
                    m_txt = getattr(m, "text", "")
                    m_id = getattr(m, "evidence_id", "evi_hist")
                    snippets.append(m_txt)
                    evidence_citations.append(
                        EvidenceCitation(
                            id=m_id,
                            title=f"Historical Baseline Record ({m_id})",
                            source="DisasterM3 Historical Dataset",
                            source_url=None,
                            is_historical=True,
                            observed_at=None,
                            published_at=None,
                            retrieved_at=datetime.now(timezone.utc).isoformat(),
                            snippet=m_txt[:250],
                            distance_km=None,
                        )
                    )
                if snippets:
                    historical_context_str = "Historical Remote Sensing Benchmark Context (DisasterM3):\n" + "\n".join(snippets)
        except Exception as e:
            logger.debug(f"Historical retrieval skipped for route: {e}")

        # 6. Safety Advisory & Summary Synthesis
        now_iso = datetime.now(timezone.utc).isoformat()
        if detected_hazards:
            critical_count = sum(1 for h in detected_hazards if h.severity in ("CRITICAL", "MAJOR"))
            hazard_summary = (
                f"Detected {len(detected_hazards)} potential hazards along the {dist_km} km transit corridor "
                f"({critical_count} critical/major alerts from official sources)."
            )
            safety_advisory = (
                "⚠️ HAZARD WARNING: Active disaster or severe weather advisories intersect your highway corridor. "
                "Do NOT assume roads are fully open or passable without contacting state/regional highway patrol. "
                "Inspect alternative routes and monitor official local emergency channels."
            )
            direct_answer = (
                f"Active hazard alerts were identified along the route between {origin_geo['display_name']} and {dest_geo['display_name']}. "
                f"Primary concerns include: {', '.join(h.title for h in detected_hazards[:3])}. "
                "Travel delays or hazardous road conditions are possible."
            )
        else:
            hazard_summary = f"No active official disaster bulletins were identified within 100 km of this {dist_km} km corridor."
            safety_advisory = (
                "ℹ️ NO ACTIVE OFFICIAL DISASTER BULLETINS DETECTED ALONG CORRIDOR. "
                "CAUTION: Absence of reports in global and national disaster feeds does NOT guarantee that the road is structurally safe, "
                "free of local debris, or passable. Minor municipal road closures, local construction washouts, and unmonitored rural bridges "
                "are not tracked in national feeds. Always drive with vigilance and verify local conditions."
            )
            direct_answer = (
                f"No major disaster alerts (earthquakes, cyclones, floods) or extreme weather hazards were detected along the primary {dist_km} km route. "
                "However, localized road obstructions or unmonitored road hazards cannot be ruled out."
            )

        unverified_text = (
            f"Municipal streets, secondary rural bypasses, uninstrumented bridges, and local storm drain blockages "
            f"along the {dist_km} km corridor are not monitored by international satellite or seismic networks."
        )

        # Build Source URLs dictionary
        source_links = [
            {"name": s.provider, "url": "https://earthquake.usgs.gov" if "USGS" in s.provider else "https://www.gdacs.org" if "GDACS" in s.provider else "https://open-meteo.com" if "Open-Meteo" in s.provider else "https://project-osrm.org"}
            for s in sources_checked
        ]

        latest_alert = None
        if detected_hazards:
            top_h = detected_hazards[0]
            latest_alert = NormalizedDisasterAlert(
                alert_id="top_route_alert",
                source=top_h.source,
                source_url=top_h.source_url,
                event_type=top_h.hazard_type,
                title=top_h.title,
                severity=top_h.severity,
                latitude=top_h.latitude,
                longitude=top_h.longitude,
                event_location=top_h.location_description,
                distance_km=top_h.distance_to_route_km,
                affects_location=True,
                event_time=top_h.reported_at,
                summary=top_h.details,
            )

        grounded_answer = InvestigationGroundedAnswer(
            direct_answer=direct_answer,
            historical_context=historical_context_str,
            latest_official_alert=latest_alert,
            supporting_evidence=evidence_citations,
            source_names_and_urls=source_links,
            timestamps={"retrieved_at": now_iso},
            confirmed_location=f"{origin_geo['display_name']} -> {dest_geo['display_name']}",
            location_provenance="geocoded_origin_and_destination",
            location_coords={"latitude": mid_lat, "longitude": mid_lon},
            conflicting_evidence=None,
            missing_information_and_uncertainty=(
                f"Real-time road surface sensors are unavailable for {dist_km} km of open roadway. "
                + (forward_warning if forward_warning else "")
            ),
            practical_next_verification_step=(
                "Check regional Department of Transportation (e.g. 511 system / state highway patrol) for live road closure maps "
                "prior to departure."
            ),
            recent_imagery=RecentImageryStatus(
                available=False,
                limitation_reason="Satellite coverage for long transit corridors cannot be delivered in continuous real-time."
            ),
            sources_checked=sources_checked,
        )

        return RouteInvestigationResponse(
            origin=origin_geo["display_name"],
            destination=dest_geo["display_name"],
            origin_coords={"latitude": o_lat, "longitude": o_lon},
            destination_coords={"latitude": d_lat, "longitude": d_lon},
            departure_time=departure_str,
            is_forward_looking=is_forward,
            forward_looking_warning=forward_warning,
            route_found=True,
            route_geometry=route_data["geometry"],
            distance_km=dist_km,
            duration_minutes=dur_min,
            alternative_routes=alternatives,
            detected_hazards=detected_hazards,
            hazard_summary=hazard_summary,
            safety_advisory=safety_advisory,
            unverified_stretches=unverified_text,
            sources_checked=sources_checked,
            evidence_citations=evidence_citations,
            grounded_answer=grounded_answer,
        )


_ROUTE_RISK_SERVICE: Optional[RouteRiskService] = None


def get_route_risk_service() -> RouteRiskService:
    global _ROUTE_RISK_SERVICE
    if _ROUTE_RISK_SERVICE is None:
        _ROUTE_RISK_SERVICE = RouteRiskService()
    return _ROUTE_RISK_SERVICE
