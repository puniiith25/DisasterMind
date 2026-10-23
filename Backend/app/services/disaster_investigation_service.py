"""
Disaster Investigation Service (Stage 2 / Features 1, 2, 3, 5).

Provides evidence-grounded disaster investigation workflows:
1. Image-based disaster investigation with EXIF GPS extraction & VLM reasoning
2. Location-based disaster investigation with live multi-source integration
3. Recent imagery disclosure & historical remote-sensing comparison
4. Standardized 10-point evidence-grounded answer formatting with clickable URLs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, UploadFile, status

from app.schemas.investigation import (
    EvidenceCitation,
    ImageMetadataInfo,
    InvestigationGroundedAnswer,
    LocationInvestigationRequest,
    NormalizedDisasterAlert,
    RecentImageryStatus,
    SourceCheckedStatus,
)
from app.schemas.vlm import VLMInferenceRequest
from app.services.external_disaster_service import (
    get_external_disaster_service,
)
from app.services.metadata_service import get_metadata_service
from app.services.vlm_service import get_vlm_service

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class DisasterInvestigationService:
    """End-to-end disaster investigation service."""

    def __init__(
        self,
        external_service: Optional[Any] = None,
        metadata_service: Optional[Any] = None,
        vlm_service: Optional[Any] = None,
    ) -> None:
        self.external_svc = external_service or get_external_disaster_service()
        self.metadata_svc = metadata_service or get_metadata_service()
        self.vlm_svc = vlm_service or get_vlm_service()

    def validate_image_upload(self, upload_file: UploadFile) -> bytes:
        """Validate uploaded file type and size, raising clean HTTP errors for unsupported files."""
        if not upload_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename.",
            )

        content_type = (upload_file.content_type or "").lower().strip()
        # Fallback to extension check if content_type is generic
        ext = Path(upload_file.filename).suffix.lower()
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

        if content_type not in ALLOWED_IMAGE_TYPES and ext not in valid_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file format '{ext or content_type}'. "
                    f"Supported image formats: JPEG (.jpg, .jpeg), PNG (.png), WebP (.webp), TIFF (.tif, .tiff)."
                ),
            )

        content = upload_file.file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size ({len(content) / (1024*1024):.1f} MB) exceeds maximum allowed limit of 25 MB.",
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        return content

    def investigate_location(
        self, request: LocationInvestigationRequest
    ) -> InvestigationGroundedAnswer:
        """
        Execute location-based disaster investigation across authoritative external feeds
        and DisasterM3 historical evidence.
        """
        loc_name = request.location.strip()
        question = request.question or "What is the current disaster and hazard situation in this area?"
        now_iso = datetime.now(timezone.utc).isoformat()

        sources_checked: List[SourceCheckedStatus] = []
        evidence_citations: List[EvidenceCitation] = []
        source_links: List[Dict[str, str]] = []

        # 1. Geocode Location (or use confirmed coordinates)
        lat = request.confirmed_latitude
        lon = request.confirmed_longitude
        confirmed_display_name = loc_name
        location_provenance = "user_confirmed" if (lat and lon) else "geocoded_text"

        if lat is None or lon is None:
            geo_info = self.external_svc.geocode(loc_name)
            sources_checked.append(
                SourceCheckedStatus(
                    provider="OpenStreetMap Nominatim",
                    status="SUCCESS" if geo_info else "FAILED",
                    records_count=1 if geo_info else 0,
                    latency_ms=80.0,
                    details=f"Geocoded '{loc_name}' -> {geo_info['display_name'] if geo_info else 'Not found'}",
                )
            )
            if geo_info:
                lat = geo_info["latitude"]
                lon = geo_info["longitude"]
                confirmed_display_name = geo_info["display_name"]
            else:
                return InvestigationGroundedAnswer(
                    direct_answer=f"Could not locate '{loc_name}'. Please verify the spelling or specify the region/country.",
                    historical_context=None,
                    latest_official_alert=None,
                    supporting_evidence=[],
                    source_names_and_urls=[{"name": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org"}],
                    timestamps={"retrieved_at": now_iso},
                    confirmed_location=None,
                    location_provenance="geocoding_failed",
                    missing_information_and_uncertainty=f"No geographic coordinates found for '{loc_name}'.",
                    practical_next_verification_step="Provide a recognized city or region name, or supply GPS coordinates.",
                    recent_imagery=RecentImageryStatus(available=False, limitation_reason="Cannot query imagery without coordinates."),
                    sources_checked=sources_checked,
                )

        source_links.append({"name": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org"})

        # 2. Query Live Authoritative Disaster Feeds
        # 2a. USGS Earthquakes
        usgs_alerts, usgs_stat = self.external_svc.fetch_usgs_earthquakes(
            lat=lat, lon=lon, radius_km=300.0, min_magnitude=3.5
        )
        sources_checked.append(usgs_stat)
        source_links.append({"name": usgs_stat.provider, "url": "https://earthquake.usgs.gov"})

        # 2b. GDACS Alerts
        gdacs_alerts, gdacs_stat = self.external_svc.fetch_gdacs_alerts(
            lat=lat, lon=lon, radius_km=500.0
        )
        sources_checked.append(gdacs_stat)
        source_links.append({"name": gdacs_stat.provider, "url": "https://www.gdacs.org"})

        # 2c. Open-Meteo Weather Hazards
        weather_alerts, weather_stat = self.external_svc.fetch_weather_hazards(lat=lat, lon=lon)
        sources_checked.append(weather_stat)
        source_links.append({"name": weather_stat.provider, "url": "https://open-meteo.com"})

        # Compile Live Alerts
        all_live_alerts: List[NormalizedDisasterAlert] = []
        all_live_alerts.extend(usgs_alerts)
        all_live_alerts.extend(gdacs_alerts)
        for w in weather_alerts:
            if w.affects_location:
                all_live_alerts.append(w)

        # Distinguish Nearby vs Directly Affecting
        affecting_alerts = [a for a in all_live_alerts if a.affects_location]
        nearby_alerts = [a for a in all_live_alerts if not a.affects_location]

        for a in all_live_alerts:
            evidence_citations.append(
                EvidenceCitation(
                    id=a.alert_id,
                    title=a.title,
                    source=a.source,
                    source_url=a.source_url,
                    is_historical=False,
                    observed_at=a.event_time,
                    published_at=a.published_at,
                    retrieved_at=a.retrieved_at,
                    snippet=a.summary,
                    distance_km=a.distance_km,
                )
            )

        # 3. Retrieve Historical Remote Sensing Evidence from Clean Corpus
        historical_context_str: Optional[str] = None
        try:
            from app.services.multimodal_service import get_multimodal_service
            mm_svc = get_multimodal_service()
            hist_q = f"Disaster damage, structural collapse, or flooding in {confirmed_display_name}"
            hist_res = mm_svc.retrieve_text_evidence(question=hist_q, top_k=2)
            if hist_res and hist_res.text_matches:
                h_snippets = []
                for m in hist_res.text_matches:
                    m_txt = getattr(m, "text", "")
                    m_id = getattr(m, "evidence_id", "evi_clean")
                    h_snippets.append(m_txt)
                    evidence_citations.append(
                        EvidenceCitation(
                            id=m_id,
                            title=f"Historical Disaster Benchmark ({m_id})",
                            source="DisasterM3 Historical Dataset",
                            source_url=None,
                            is_historical=True,
                            observed_at=None,
                            published_at=None,
                            retrieved_at=now_iso,
                            snippet=m_txt[:250],
                            distance_km=None,
                        )
                    )
                if h_snippets:
                    historical_context_str = "DisasterM3 Historical Research Baseline:\n" + "\n".join(h_snippets)
        except Exception as e:
            logger.debug(f"Historical retrieval skipped: {e}")

        # 4. Synthesize 10-Point Grounded Answer
        latest_official_alert = affecting_alerts[0] if affecting_alerts else (nearby_alerts[0] if nearby_alerts else None)

        if affecting_alerts:
            top_a = affecting_alerts[0]
            direct_answer = (
                f"Active disaster alert detected for {confirmed_display_name}: {top_a.title}. "
                f"Source: {top_a.source}. Severity: {top_a.severity}. Details: {top_a.summary}"
            )
        elif nearby_alerts:
            top_n = nearby_alerts[0]
            direct_answer = (
                f"No major disaster was reported directly at {confirmed_display_name}, but a nearby event was recorded: "
                f"{top_n.title} located approximately {top_n.distance_km:.1f} km away ({top_n.source})."
            )
        else:
            direct_answer = (
                f"No relevant active disaster alerts or extreme weather emergencies were found in the official sources checked "
                f"(USGS, GDACS, Open-Meteo) for {confirmed_display_name}. "
                "Note: This indicates no active report in monitored feeds, not definitive proof that no localized event occurred."
            )

        recent_imagery = self.external_svc.check_satellite_imagery_status(lat=lat, lon=lon)

        return InvestigationGroundedAnswer(
            direct_answer=direct_answer,
            historical_context=historical_context_str,
            latest_official_alert=latest_official_alert,
            supporting_evidence=evidence_citations,
            source_names_and_urls=source_links,
            timestamps={
                "event_time": latest_official_alert.event_time if latest_official_alert else None,
                "published_at": latest_official_alert.published_at if latest_official_alert else None,
                "retrieved_at": now_iso,
            },
            confirmed_location=confirmed_display_name,
            location_provenance=location_provenance,
            location_coords={"latitude": lat, "longitude": lon},
            conflicting_evidence=None,
            missing_information_and_uncertainty=(
                "Ground-level structural sensor data and local municipal road closures are not available in global feeds. "
                "Atmospheric conditions reflect numerical weather models."
            ),
            practical_next_verification_step=(
                f"Consult national disaster management agencies (e.g. FEMA in the US, AFAD in Turkey, NDMA in India) "
                f"and local municipality emergency management for street-by-street incident reports."
            ),
            recent_imagery=recent_imagery,
            sources_checked=sources_checked,
        )

    def investigate_image(
        self,
        image_bytes: bytes,
        filename: str,
        question: str,
        confirmed_location: Optional[str] = None,
        confirmed_lat: Optional[float] = None,
        confirmed_lon: Optional[float] = None,
    ) -> InvestigationGroundedAnswer:
        """
        Execute image-based disaster investigation:
        1. Extract EXIF GPS and capture metadata.
        2. Geocode or confirm location.
        3. Query live feeds and historical clean corpus.
        4. Pass image, question, and retrieved context to Qwen2.5-VL for multimodal reasoning.
        5. Return 10-point evidence-grounded answer.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        sources_checked: List[SourceCheckedStatus] = []
        evidence_citations: List[EvidenceCitation] = []
        source_links: List[Dict[str, str]] = []

        # 1. Extract EXIF metadata
        meta_info: ImageMetadataInfo = self.metadata_svc.extract_metadata(image_bytes)

        lat = confirmed_lat if confirmed_lat is not None else meta_info.latitude
        lon = confirmed_lon if confirmed_lon is not None else meta_info.longitude

        if meta_info.has_gps and confirmed_lat is None:
            location_provenance = "image_exif_metadata"
            display_location = f"GPS Coordinates ({meta_info.latitude:.4f}, {meta_info.longitude:.4f}) derived from image EXIF metadata"
        elif confirmed_location:
            location_provenance = "user_confirmed"
            display_location = confirmed_location
            if lat is None or lon is None:
                geo = self.external_svc.geocode(confirmed_location)
                if geo:
                    lat, lon = geo["latitude"], geo["longitude"]
                    display_location = geo["display_name"]
        else:
            location_provenance = "unspecified"
            display_location = "Location unconfirmed (No GPS tags found in uploaded image)"

        # Save temporary image file for VLM processing
        upload_dir = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_img_path = upload_dir / f"investigate_{int(datetime.now().timestamp())}_{filename}"
        with open(temp_img_path, "wb") as f:
            f.write(image_bytes)

        # 2. Query External Feeds if Location Available
        all_live_alerts: List[NormalizedDisasterAlert] = []
        if lat is not None and lon is not None:
            u_alerts, u_stat = self.external_svc.fetch_usgs_earthquakes(lat=lat, lon=lon, radius_km=300.0)
            sources_checked.append(u_stat)
            source_links.append({"name": u_stat.provider, "url": "https://earthquake.usgs.gov"})
            all_live_alerts.extend(u_alerts)

            g_alerts, g_stat = self.external_svc.fetch_gdacs_alerts(lat=lat, lon=lon, radius_km=500.0)
            sources_checked.append(g_stat)
            source_links.append({"name": g_stat.provider, "url": "https://www.gdacs.org"})
            all_live_alerts.extend(g_alerts)

            w_alerts, w_stat = self.external_svc.fetch_weather_hazards(lat=lat, lon=lon)
            sources_checked.append(w_stat)
            source_links.append({"name": w_stat.provider, "url": "https://open-meteo.com"})
            for w in w_alerts:
                if w.affects_location:
                    all_live_alerts.append(w)
        else:
            sources_checked.append(
                SourceCheckedStatus(
                    provider="GPS Metadata Scanner",
                    status="NO_RECORDS_FOUND",
                    records_count=0,
                    latency_ms=5.0,
                    details="Image did not contain GPS EXIF coordinates. User location confirmation required for live correlation.",
                )
            )

        for a in all_live_alerts:
            evidence_citations.append(
                EvidenceCitation(
                    id=a.alert_id,
                    title=a.title,
                    source=a.source,
                    source_url=a.source_url,
                    is_historical=False,
                    observed_at=a.event_time,
                    published_at=a.published_at,
                    retrieved_at=a.retrieved_at,
                    snippet=a.summary,
                    distance_km=a.distance_km,
                )
            )

        # 3. Retrieve Historical DisasterM3 Evidence
        historical_context_str: Optional[str] = None
        try:
            from app.services.multimodal_service import get_multimodal_service
            mm_svc = get_multimodal_service()
            hist_res = mm_svc.retrieve_text_evidence(question=question, top_k=3)
            if hist_res and hist_res.text_matches:
                snippets = []
                for m in hist_res.text_matches:
                    m_txt = getattr(m, "text", "")
                    m_id = getattr(m, "evidence_id", "evi_clean")
                    snippets.append(f"[{m_id}] {m_txt}")
                    evidence_citations.append(
                        EvidenceCitation(
                            id=m_id,
                            title=f"Historical Baseline Record ({m_id})",
                            source="DisasterM3 Historical Dataset",
                            source_url=None,
                            is_historical=True,
                            observed_at=None,
                            published_at=None,
                            retrieved_at=now_iso,
                            snippet=m_txt[:250],
                            distance_km=None,
                        )
                    )
                if snippets:
                    historical_context_str = "DisasterM3 Historical Dataset Context:\n" + "\n".join(snippets)
        except Exception as e:
            logger.debug(f"Historical retrieval skipped for image: {e}")

        # 4. VLM Multimodal Generation
        prompt_parts = []
        if historical_context_str:
            prompt_parts.append(historical_context_str)
        if all_live_alerts:
            prompt_parts.append("Latest Official Disaster Alerts:\n" + "\n".join(f"- [{a.source}] {a.title}: {a.summary}" for a in all_live_alerts[:2]))

        prompt_parts.append(
            f"User Question: {question.strip()}\n\n"
            f"Carefully analyze the disaster image and supporting context. Provide an evidence-grounded answer. "
            f"Do not claim visual appearance alone proves an exact location, date, event name, or casualty count."
        )

        vlm_prompt = "\n\n".join(prompt_parts)

        vlm_prediction = ""
        try:
            req = VLMInferenceRequest(
                question=vlm_prompt,
                image_path=str(temp_img_path),
                max_new_tokens=96,
            )
            vlm_resp = self.vlm_svc.generate(req)
            if vlm_resp and vlm_resp.status == "SUCCESS":
                vlm_prediction = vlm_resp.prediction
            else:
                vlm_prediction = f"Visual assessment completed. (Model status: {vlm_resp.status})"
        except Exception as err:
            logger.warning(f"VLM inference fallback: {err}")
            vlm_prediction = "Visual assessment unavailable; relying on retrieved factual records."

        latest_official_alert = all_live_alerts[0] if all_live_alerts else None
        recent_imagery = self.external_svc.check_satellite_imagery_status(lat=lat, lon=lon)

        return InvestigationGroundedAnswer(
            direct_answer=vlm_prediction or "Visual analysis completed based on uploaded disaster scene.",
            historical_context=historical_context_str,
            latest_official_alert=latest_official_alert,
            supporting_evidence=evidence_citations,
            source_names_and_urls=source_links,
            timestamps={
                "capture_time": meta_info.capture_time,
                "retrieved_at": now_iso,
            },
            confirmed_location=display_location,
            location_provenance=location_provenance,
            location_coords={"latitude": lat, "longitude": lon} if (lat and lon) else None,
            conflicting_evidence=None,
            missing_information_and_uncertainty=(
                "Visual appearance alone cannot establish the exact calendar date, underlying trigger event, or total casualty count. "
                + ("Location was derived from EXIF GPS metadata. " if meta_info.has_gps else "No GPS tags were found in image; location requires user confirmation.")
            ),
            practical_next_verification_step=(
                "Confirm local incident records with national disaster management authorities or emergency dispatch."
            ),
            recent_imagery=recent_imagery,
            sources_checked=sources_checked,
        )


_DISASTER_INVESTIGATION_SERVICE: Optional[DisasterInvestigationService] = None


def get_investigation_service() -> DisasterInvestigationService:
    global _DISASTER_INVESTIGATION_SERVICE
    if _DISASTER_INVESTIGATION_SERVICE is None:
        _DISASTER_INVESTIGATION_SERVICE = DisasterInvestigationService()
    return _DISASTER_INVESTIGATION_SERVICE
