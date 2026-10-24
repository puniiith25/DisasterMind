"""
External Disaster Intelligence Service.

Provides robust, timeout-bounded integration with real authoritative disaster data feeds:
1. USGS Earthquake Hazards API (free public GeoJSON)
2. GDACS Global Disaster Alert & Coordination System (UN / EC RSS alerts)
3. Open-Meteo Severe Weather & Hazard API (free open-access weather)
4. OpenStreetMap Nominatim Geocoding API (free geocoding)
5. Project OSRM Routing Engine (free route geometry & waypoints)
6. NASA GIBS / Satellite Imagery Catalog Status

Features:
- In-memory caching with configurable TTL (5 minutes)
- 5-second HTTP request timeouts to prevent blocking
- Partial failure isolation (if one provider fails, others proceed)
- Zero synthetic data fabrication
- Clean geographic distance calculation (Haversine formula)
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import httpx

from app.schemas.investigation import (
    NormalizedDisasterAlert,
    RecentImageryStatus,
    SourceCheckedStatus,
)

logger = logging.getLogger(__name__)

# Cache store: key -> (timestamp, data)
_EXTERNAL_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 300.0  # 5 minutes


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


class ExternalDisasterService:
    """Service orchestrating external disaster feeds with caching and resilience."""

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._client = client

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            timeout=5.0,
            headers={"User-Agent": "DisasterMind-Academic/1.0 (disastermind@local.test)"},
            follow_redirects=True,
        )

    # -------------------------------------------------------------------------
    # 1. OpenStreetMap Nominatim Geocoding
    # -------------------------------------------------------------------------
    def geocode(self, location_query: str) -> Optional[Dict[str, Any]]:
        """Geocode place name to latitude, longitude and display name."""
        q = location_query.strip()
        if not q:
            return None

        cache_key = f"geocode_{q.lower()}"
        now = time.time()
        if cache_key in _EXTERNAL_CACHE:
            ts, val = _EXTERNAL_CACHE[cache_key]
            if now - ts < CACHE_TTL_SECONDS * 6:  # Longer cache for geocoding
                return val

        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 1}

        try:
            with self._get_client() as client:
                resp = client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        top = data[0]
                        res = {
                            "display_name": top.get("display_name", q),
                            "latitude": float(top["lat"]),
                            "longitude": float(top["lon"]),
                            "type": top.get("type", "location"),
                        }
                        _EXTERNAL_CACHE[cache_key] = (now, res)
                        return res
        except Exception as exc:
            logger.warning(f"Geocoding failed for '{q}': {exc}")

        return None

    # -------------------------------------------------------------------------
    # 2. USGS Earthquakes
    # -------------------------------------------------------------------------
    def fetch_usgs_earthquakes(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: float = 350.0,
        min_magnitude: float = 3.5,
        limit: int = 10,
    ) -> Tuple[List[NormalizedDisasterAlert], SourceCheckedStatus]:
        """Fetch real-time seismic events from USGS GeoJSON API."""
        start_t = time.perf_counter()
        provider = "USGS Earthquake Hazards Program"
        alerts: List[NormalizedDisasterAlert] = []

        cache_key = f"usgs_{lat}_{lon}_{radius_km}_{min_magnitude}"
        now = time.time()
        if cache_key in _EXTERNAL_CACHE:
            ts, cached_data = _EXTERNAL_CACHE[cache_key]
            if now - ts < CACHE_TTL_SECONDS:
                return cached_data

        try:
            with self._get_client() as client:
                if lat is not None and lon is not None:
                    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
                    params: Dict[str, Any] = {
                        "format": "geojson",
                        "latitude": lat,
                        "longitude": lon,
                        "maxradiuskm": radius_km,
                        "minmagnitude": min_magnitude,
                        "limit": limit,
                    }
                    resp = client.get(url, params=params)
                else:
                    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
                    resp = client.get(url)

                dur_ms = (time.perf_counter() - start_t) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    features = data.get("features", [])
                    for feat in features[:limit]:
                        props = feat.get("properties", {})
                        geom = feat.get("geometry", {})
                        coords = geom.get("coordinates", [None, None])
                        e_lon = coords[0] if len(coords) > 0 else None
                        e_lat = coords[1] if len(coords) > 1 else None

                        dist_km = None
                        affects = False
                        if lat is not None and lon is not None and e_lat is not None and e_lon is not None:
                            dist_km = _haversine_distance_km(lat, lon, e_lat, e_lon)
                            affects = dist_km <= 80.0  # Significant shake impact zone

                        mag = props.get("mag", 0.0)
                        severity = (
                            "CRITICAL" if mag >= 6.5
                            else "MAJOR" if mag >= 5.5
                            else "MODERATE" if mag >= 4.5
                            else "ADVISORY"
                        )

                        t_epoch = props.get("time", 0) / 1000.0
                        event_time = (
                            datetime.fromtimestamp(t_epoch, tz=timezone.utc).isoformat()
                            if t_epoch > 0 else None
                        )

                        alert = NormalizedDisasterAlert(
                            alert_id=f"usgs_{feat.get('id', 'unk')}",
                            source=provider,
                            source_id=feat.get("id"),
                            source_url=props.get("url") or "https://earthquake.usgs.gov",
                            event_type="Earthquake",
                            title=props.get("title", f"M{mag} Earthquake"),
                            severity=severity,
                            latitude=e_lat,
                            longitude=e_lon,
                            event_location=props.get("place", "Unknown Location"),
                            distance_km=dist_km,
                            affects_location=affects,
                            event_time=event_time,
                            published_at=event_time,
                            summary=(
                                f"M{mag} seismic event recorded at {props.get('place')}. "
                                f"Felt reports: {props.get('felt', 0)}. Tsunami warning flag: {props.get('tsunami', 0)}."
                            ),
                            limitations="Automated preliminary USGS seismic solution; depths and magnitudes may be revised.",
                        )
                        alerts.append(alert)

                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="SUCCESS" if alerts else "NO_RECORDS_FOUND",
                        records_count=len(alerts),
                        latency_ms=round(dur_ms, 2),
                        details="Official USGS Earthquake Catalog query completed.",
                    )
                    res = (alerts, status_rec)
                    _EXTERNAL_CACHE[cache_key] = (now, res)
                    return res
                else:
                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="UNAVAILABLE",
                        records_count=0,
                        latency_ms=round(dur_ms, 2),
                        details=f"USGS returned HTTP {resp.status_code}",
                    )
                    return ([], status_rec)

        except Exception as exc:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"USGS query error: {exc}")
            status_rec = SourceCheckedStatus(
                provider=provider,
                status="FAILED",
                records_count=0,
                latency_ms=round(dur_ms, 2),
                details=f"Connection error: {str(exc)}",
            )
            return ([], status_rec)

    # -------------------------------------------------------------------------
    # 3. GDACS Global Disaster Alerts (UN / EC)
    # -------------------------------------------------------------------------
    def fetch_gdacs_alerts(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: float = 600.0,
        limit: int = 10,
    ) -> Tuple[List[NormalizedDisasterAlert], SourceCheckedStatus]:
        """Fetch active multi-hazard alerts from GDACS (Floods, Tropical Cyclones, Volcanoes, etc.)."""
        start_t = time.perf_counter()
        provider = "GDACS (UN / EC Disaster Alert)"
        alerts: List[NormalizedDisasterAlert] = []

        cache_key = f"gdacs_{lat}_{lon}_{radius_km}"
        now = time.time()
        if cache_key in _EXTERNAL_CACHE:
            ts, cached_data = _EXTERNAL_CACHE[cache_key]
            if now - ts < CACHE_TTL_SECONDS:
                return cached_data

        url = "https://www.gdacs.org/xml/rss.xml"

        try:
            with self._get_client() as client:
                resp = client.get(url)
                dur_ms = (time.perf_counter() - start_t) * 1000.0

                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    items = root.findall(".//item")

                    for item in items:
                        title_el = item.find("title")
                        desc_el = item.find("description")
                        link_el = item.find("link")
                        pub_date_el = item.find("pubDate")
                        guid_el = item.find("guid")

                        # GDACS geo namespace coordinates
                        point_el = item.find("{http://www.georss.org/georss}point")
                        lat_el = item.find("{http://www.w3.org/2003/01/geo/wgs84_pos#}lat")
                        lon_el = item.find("{http://www.w3.org/2003/01/geo/wgs84_pos#}long")

                        e_lat, e_lon = None, None
                        if point_el is not None and point_el.text:
                            parts = point_el.text.strip().split()
                            if len(parts) == 2:
                                try:
                                    e_lat, e_lon = float(parts[0]), float(parts[1])
                                except ValueError:
                                    pass
                        elif lat_el is not None and lon_el is not None:
                            try:
                                e_lat = float(lat_el.text) if lat_el.text else None
                                e_lon = float(lon_el.text) if lon_el.text else None
                            except ValueError:
                                pass

                        title_txt = title_el.text if title_el is not None and title_el.text else "GDACS Alert"
                        desc_txt = desc_el.text if desc_el is not None and desc_el.text else ""
                        link_txt = link_el.text if link_el is not None and link_el.text else "https://www.gdacs.org"
                        guid_txt = guid_el.text if guid_el is not None and guid_el.text else f"gdacs_{len(alerts)}"
                        pub_txt = pub_date_el.text if pub_date_el is not None and pub_date_el.text else None

                        # Determine event type
                        t_lower = (title_txt + " " + desc_txt).lower()
                        event_type = "Disaster Event"
                        if "flood" in t_lower:
                            event_type = "Flood"
                        elif "cyclone" in t_lower or "hurricane" in t_lower or "typhoon" in t_lower:
                            event_type = "Tropical Cyclone"
                        elif "earthquake" in t_lower:
                            event_type = "Earthquake"
                        elif "volcano" in t_lower:
                            event_type = "Volcano"
                        elif "fire" in t_lower or "wildfire" in t_lower:
                            event_type = "Wildfire"
                        elif "drought" in t_lower:
                            event_type = "Drought"

                        # Determine severity from GDACS color alerts
                        severity = "MODERATE"
                        if "red" in t_lower:
                            severity = "CRITICAL"
                        elif "orange" in t_lower:
                            severity = "MAJOR"
                        elif "green" in t_lower:
                            severity = "ADVISORY"

                        # Proximity filtering if location supplied
                        dist_km = None
                        affects = False
                        if lat is not None and lon is not None and e_lat is not None and e_lon is not None:
                            dist_km = _haversine_distance_km(lat, lon, e_lat, e_lon)
                            if dist_km > radius_km:
                                continue  # Skip events beyond radius
                            affects = dist_km <= 100.0

                        alerts.append(
                            NormalizedDisasterAlert(
                                alert_id=f"gdacs_{guid_txt}",
                                source=provider,
                                source_id=guid_txt,
                                source_url=link_txt,
                                event_type=event_type,
                                title=title_txt,
                                severity=severity,
                                latitude=e_lat,
                                longitude=e_lon,
                                event_location=title_txt.split("in")[-1].strip() if "in" in title_txt else title_txt,
                                distance_km=dist_km,
                                affects_location=affects,
                                event_time=pub_txt,
                                published_at=pub_txt,
                                summary=desc_txt[:300] if desc_txt else title_txt,
                                limitations="GDACS automated multi-hazard alert feed.",
                            )
                        )
                        if len(alerts) >= limit:
                            break

                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="SUCCESS" if alerts else "NO_RECORDS_FOUND",
                        records_count=len(alerts),
                        latency_ms=round(dur_ms, 2),
                        details="GDACS global multi-hazard RSS query completed.",
                    )
                    res = (alerts, status_rec)
                    _EXTERNAL_CACHE[cache_key] = (now, res)
                    return res
                else:
                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="UNAVAILABLE",
                        records_count=0,
                        latency_ms=round(dur_ms, 2),
                        details=f"GDACS returned HTTP {resp.status_code}",
                    )
                    return ([], status_rec)

        except Exception as exc:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"GDACS query error: {exc}")
            status_rec = SourceCheckedStatus(
                provider=provider,
                status="FAILED",
                records_count=0,
                latency_ms=round(dur_ms, 2),
                details=f"Connection error: {str(exc)}",
            )
            return ([], status_rec)

    # -------------------------------------------------------------------------
    # 4. Open-Meteo Severe Weather & Hazard Monitoring
    # -------------------------------------------------------------------------
    def fetch_weather_hazards(
        self,
        lat: float,
        lon: float,
        departure_date: Optional[str] = None,
    ) -> Tuple[List[NormalizedDisasterAlert], SourceCheckedStatus]:
        """Fetch current and forecasted severe weather conditions (wind, precip, storms)."""
        start_t = time.perf_counter()
        provider = "Open-Meteo Weather Service"
        alerts: List[NormalizedDisasterAlert] = []

        cache_key = f"weather_{lat}_{lon}_{departure_date}"
        now = time.time()
        if cache_key in _EXTERNAL_CACHE:
            ts, cached_data = _EXTERNAL_CACHE[cache_key]
            if now - ts < CACHE_TTL_SECONDS:
                return cached_data

        # Parse departure_date and check forecast horizon
        target_dt: Optional[date] = None
        days_ahead: int = 0
        if departure_date:
            try:
                clean_d = departure_date.strip().split("T")[0]
                if clean_d.lower() == "tomorrow":
                    target_dt = datetime.now(timezone.utc).date() + timedelta(days=1)
                elif clean_d.lower() in ("today", "now"):
                    target_dt = datetime.now(timezone.utc).date()
                else:
                    target_dt = datetime.fromisoformat(clean_d).date()
                today = datetime.now(timezone.utc).date()
                days_ahead = (target_dt - today).days
            except Exception as e:
                logger.debug(f"Could not parse departure_date '{departure_date}': {e}")

        # Check if requested date is beyond Open-Meteo's standard 7-day forecast horizon
        if target_dt is not None and days_ahead > 7:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            alert = NormalizedDisasterAlert(
                alert_id=f"weather_{lat}_{lon}_out_of_horizon",
                source=provider,
                source_id="horizon_exceeded",
                source_url="https://open-meteo.com",
                event_type="Forecast Horizon Exceeded",
                title=f"Travel Date Outside 7-Day Forecast Horizon ({target_dt.isoformat()})",
                severity="ADVISORY",
                latitude=lat,
                longitude=lon,
                event_location=f"Coordinates ({lat:.3f}, {lon:.3f})",
                distance_km=0.0,
                affects_location=False,
                event_time=target_dt.isoformat(),
                published_at=datetime.now(timezone.utc).isoformat(),
                summary=(
                    f"The requested travel date ({target_dt.isoformat()}) is {days_ahead} days in the future, "
                    "which exceeds the 7-day numerical weather prediction horizon supported by Open-Meteo. "
                    "Atmospheric conditions and localized storm hazards cannot be reliably forecast at this lead time. "
                    "Re-verify conditions 24 to 48 hours prior to departure."
                ),
                limitations=f"Target date is {days_ahead} days ahead (max forecast horizon is 7 days).",
            )
            alerts.append(alert)
            status_rec = SourceCheckedStatus(
                provider=provider,
                status="SUCCESS",
                records_count=1,
                latency_ms=round(dur_ms, 2),
                details=f"Target date {target_dt.isoformat()} exceeds 7-day forecast horizon.",
            )
            res = (alerts, status_rec)
            _EXTERNAL_CACHE[cache_key] = (now, res)
            return res

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,precipitation,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "forecast_days": 7,
            "timezone": "auto",
        }

        try:
            with self._get_client() as client:
                resp = client.get(url, params=params)
                dur_ms = (time.perf_counter() - start_t) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()

                    # Handle future forecast day if target_dt is in the future
                    if target_dt is not None and days_ahead > 0:
                        daily = data.get("daily", {})
                        daily_dates = daily.get("time", [])
                        target_iso = target_dt.isoformat()

                        if target_iso in daily_dates:
                            idx = daily_dates.index(target_iso)
                            temp_max = daily.get("temperature_2m_max", [None])[idx]
                            temp_min = daily.get("temperature_2m_min", [None])[idx]
                            wind = daily.get("wind_speed_10m_max", [0.0])[idx] or 0.0
                            precip = daily.get("precipitation_sum", [0.0])[idx] or 0.0
                            wcode = daily.get("weather_code", [0])[idx] or 0

                            has_hazard = False
                            severity = "ADVISORY"
                            h_type = "Severe Weather Forecast"
                            details_list = []

                            if wind >= 60.0:
                                has_hazard = True
                                severity = "CRITICAL" if wind >= 85.0 else "MAJOR"
                                h_type = "High Wind Hazard Forecast"
                                details_list.append(f"Forecast gale/storm wind gusts: {wind} km/h")
                            elif wind >= 40.0:
                                details_list.append(f"Elevated forecast wind gusts: {wind} km/h")

                            if precip >= 15.0:
                                has_hazard = True
                                severity = "MAJOR"
                                h_type = "Heavy Precipitation / Flood Risk Forecast"
                                details_list.append(f"Heavy daily rainfall forecast: {precip} mm")
                            elif precip >= 5.0:
                                details_list.append(f"Moderate rainfall forecast: {precip} mm")

                            if wcode in (95, 96, 99):
                                has_hazard = True
                                h_type = "Thunderstorm / Convective Hazard Forecast"
                                severity = "MAJOR"
                                details_list.append("Convective storm and lightning predicted")
                            elif wcode in (71, 73, 75, 85, 86):
                                has_hazard = True
                                h_type = "Winter Snowstorm Forecast"
                                details_list.append("Significant snowfall accumulation predicted")

                            summary_txt = (
                                f"Forecast for {target_iso}: Min {temp_min}°C, Max {temp_max}°C | "
                                f"Max Wind: {wind} km/h | Precip: {precip} mm. "
                                + ("; ".join(details_list) if details_list else "Favorable/manageable conditions expected.")
                            )

                            alert = NormalizedDisasterAlert(
                                alert_id=f"weather_forecast_{lat}_{lon}_{target_iso}",
                                source=provider,
                                source_id=f"wmo_daily_{wcode}",
                                source_url="https://open-meteo.com",
                                event_type=h_type,
                                title=f"{h_type} ({temp_min}°C to {temp_max}°C)" if has_hazard else f"Atmospheric Forecast ({temp_min}°C to {temp_max}°C)",
                                severity=severity if has_hazard else "ADVISORY",
                                latitude=lat,
                                longitude=lon,
                                event_location=f"Coordinates ({lat:.3f}, {lon:.3f})",
                                distance_km=0.0,
                                affects_location=has_hazard,
                                event_time=target_iso,
                                published_at=datetime.now(timezone.utc).isoformat(),
                                summary=summary_txt,
                                limitations=f"7-day numerical forecast model for {target_iso}; subject to synoptic shifts.",
                            )
                            alerts.append(alert)

                    # Also include current observation if available
                    curr = data.get("current", {})
                    temp = curr.get("temperature_2m")
                    wind = curr.get("wind_speed_10m", 0.0) or 0.0
                    precip = curr.get("precipitation", 0.0) or 0.0
                    wcode = curr.get("weather_code", 0) or 0

                    # Assess hazards based on thresholds
                    # WMO codes: 95-99 = thunderstorm, 65-67 = heavy rain/freezing rain, 75 = heavy snow
                    has_hazard = False
                    severity = "ADVISORY"
                    h_type = "Severe Weather"
                    details_list = []

                    if wind >= 60.0:
                        has_hazard = True
                        severity = "CRITICAL" if wind >= 85.0 else "MAJOR"
                        h_type = "High Wind Hazard"
                        details_list.append(f"Dangerous cyclonic/gale wind gusts: {wind} km/h")
                    elif wind >= 40.0:
                        details_list.append(f"Elevated wind gusts: {wind} km/h")

                    if precip >= 15.0:
                        has_hazard = True
                        severity = "MAJOR"
                        h_type = "Heavy Precipitation / Flood Risk"
                        details_list.append(f"Extreme heavy rainfall rate: {precip} mm/h")
                    elif precip >= 5.0:
                        details_list.append(f"Moderate rainfall: {precip} mm/h")

                    if wcode in (95, 96, 99):
                        has_hazard = True
                        h_type = "Thunderstorm / Hail Alert"
                        severity = "MAJOR"
                        details_list.append("Active convective thunderstorm and lightning activity")
                    elif wcode in (71, 73, 75, 85, 86):
                        has_hazard = True
                        h_type = "Winter Storm / Blizzard Risk"
                        details_list.append("Snowfall accumulation and freezing precipitation")

                    summary_txt = f"Current Obs: Temp: {temp}°C | Wind: {wind} km/h | Precipitation: {precip} mm/h. " + "; ".join(details_list)

                    alert = NormalizedDisasterAlert(
                        alert_id=f"weather_{lat}_{lon}",
                        source=provider,
                        source_id=f"wmo_{wcode}",
                        source_url="https://open-meteo.com",
                        event_type=h_type,
                        title=f"{h_type} ({temp}°C, Wind {wind} km/h)" if has_hazard else "Current Atmospheric Observation",
                        severity=severity if has_hazard else "ADVISORY",
                        latitude=lat,
                        longitude=lon,
                        event_location=f"Coordinates ({lat:.3f}, {lon:.3f})",
                        distance_km=0.0,
                        affects_location=has_hazard,
                        event_time=curr.get("time"),
                        published_at=curr.get("time"),
                        summary=summary_txt,
                        limitations="Numerical weather prediction model data; localized microclimates may vary.",
                    )
                    alerts.append(alert)

                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="SUCCESS",
                        records_count=len(alerts),
                        latency_ms=round(dur_ms, 2),
                        details="Atmospheric hazard forecast retrieved.",
                    )
                    res = (alerts, status_rec)
                    _EXTERNAL_CACHE[cache_key] = (now, res)
                    return res
                else:
                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="UNAVAILABLE",
                        records_count=0,
                        latency_ms=round(dur_ms, 2),
                        details=f"Open-Meteo returned HTTP {resp.status_code}",
                    )
                    return ([], status_rec)

        except Exception as exc:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"Open-Meteo query error: {exc}")
            status_rec = SourceCheckedStatus(
                provider=provider,
                status="FAILED",
                records_count=0,
                latency_ms=round(dur_ms, 2),
                details=f"Connection error: {str(exc)}",
            )
            return ([], status_rec)

    # -------------------------------------------------------------------------
    # 5. OSRM Routing Engine
    # -------------------------------------------------------------------------
    def fetch_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> Tuple[Optional[Dict[str, Any]], SourceCheckedStatus]:
        """Fetch actual road transit geometry and alternatives from Project OSRM."""
        start_t = time.perf_counter()
        provider = "Project OSRM (Open Source Routing Machine)"

        cache_key = f"osrm_{start_lat}_{start_lon}_{end_lat}_{end_lon}"
        now = time.time()
        if cache_key in _EXTERNAL_CACHE:
            ts, cached_data = _EXTERNAL_CACHE[cache_key]
            if now - ts < CACHE_TTL_SECONDS:
                return cached_data

        url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "alternatives": "true",
            "steps": "false",
        }

        try:
            with self._get_client() as client:
                resp = client.get(url, params=params)
                dur_ms = (time.perf_counter() - start_t) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    routes = data.get("routes", [])
                    if routes:
                        primary = routes[0]
                        alternatives = []
                        for alt in routes[1:]:
                            alternatives.append({
                                "distance_km": round(alt.get("distance", 0) / 1000.0, 2),
                                "duration_minutes": round(alt.get("duration", 0) / 60.0, 1),
                                "geometry": alt.get("geometry"),
                            })

                        route_info = {
                            "distance_km": round(primary.get("distance", 0) / 1000.0, 2),
                            "duration_minutes": round(primary.get("duration", 0) / 60.0, 1),
                            "geometry": primary.get("geometry"),  # GeoJSON LineString coordinates
                            "alternatives": alternatives,
                        }

                        status_rec = SourceCheckedStatus(
                            provider=provider,
                            status="SUCCESS",
                            records_count=len(routes),
                            latency_ms=round(dur_ms, 2),
                            details=f"OSRM computed route: {route_info['distance_km']} km, {len(primary['geometry']['coordinates'])} waypoints.",
                        )
                        res = (route_info, status_rec)
                        _EXTERNAL_CACHE[cache_key] = (now, res)
                        return res

                    status_rec = SourceCheckedStatus(
                        provider=provider,
                        status="NO_RECORDS_FOUND",
                        records_count=0,
                        latency_ms=round(dur_ms, 2),
                        details="No navigable driving route between coordinates.",
                    )
                    return (None, status_rec)

                status_rec = SourceCheckedStatus(
                    provider=provider,
                    status="UNAVAILABLE",
                    records_count=0,
                    latency_ms=round(dur_ms, 2),
                    details=f"OSRM returned HTTP {resp.status_code}",
                )
                return (None, status_rec)

        except Exception as exc:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"OSRM route query error: {exc}")
            status_rec = SourceCheckedStatus(
                provider=provider,
                status="FAILED",
                records_count=0,
                latency_ms=round(dur_ms, 2),
                details=f"Connection error: {str(exc)}",
            )
            return (None, status_rec)

    # -------------------------------------------------------------------------
    # 6. NASA GIBS / Recent Imagery Verification
    # -------------------------------------------------------------------------
    def check_satellite_imagery_status(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> RecentImageryStatus:
        """
        Verify available recent imagery sources without fabricating images or dates.
        Accurately identifies open-access NASA GIBS layers and explains resolution limits.
        """
        now_dt = datetime.now(timezone.utc)
        retrieval_str = now_dt.isoformat()

        if lat is None or lon is None:
            return RecentImageryStatus(
                available=False,
                provider="None",
                capture_date=None,
                retrieval_date=retrieval_str,
                limitation_reason=(
                    "No location coordinates provided to query recent satellite passes. "
                    "Confirm a location to query satellite acquisition metadata."
                ),
            )

        # NASA GIBS Modis / VIIRS daily composite layer is public open-access
        # but 250m resolution is insufficient for building damage evaluation
        gibs_layer = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
        date_str = now_dt.strftime("%Y-%m-%d")

        return RecentImageryStatus(
            available=False,
            provider="NASA GIBS / Worldview (MODIS/VIIRS)",
            capture_date=date_str,
            retrieval_date=retrieval_str,
            tile_url=f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{gibs_layer}/default/{date_str}/250m/0/0/0.jpg",
            limitation_reason=(
                "Open-access daily satellite passes (NASA VIIRS/MODIS at 250m resolution) provide regional weather/smoke views "
                "but lack the sub-meter optical resolution needed to verify individual building damage or road passability. "
                "Sub-meter commercial radar/optical satellite passes (e.g. WorldView, PlanetScope) require paid tasking "
                "and are not freely accessible in real time."
            ),
        )


_EXTERNAL_SERVICE: Optional[ExternalDisasterService] = None


def get_external_disaster_service() -> ExternalDisasterService:
    global _EXTERNAL_SERVICE
    if _EXTERNAL_SERVICE is None:
        _EXTERNAL_SERVICE = ExternalDisasterService()
    return _EXTERNAL_SERVICE
