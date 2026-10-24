"""
Unit tests for Stages 2 & 3: Current Disaster Intelligence and Route Risk Investigation.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.schemas.investigation import (
    LocationInvestigationRequest,
    NormalizedDisasterAlert,
    RouteInvestigationRequest,
    SourceCheckedStatus,
)
from app.services.disaster_investigation_service import DisasterInvestigationService
from app.services.external_disaster_service import ExternalDisasterService, _haversine_distance_km
from app.services.metadata_service import ImageMetadataService
from app.services.route_risk_service import RouteRiskService


@pytest.fixture
def test_image_bytes():
    """Create a minimal PNG test image in memory."""
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMetadataService:
    def test_extract_metadata_without_gps(self, test_image_bytes):
        service = ImageMetadataService()
        meta = service.extract_metadata(test_image_bytes)
        assert meta.has_gps is False
        assert meta.location_provenance == "absent"
        assert meta.needs_location_confirmation is True
        assert meta.latitude is None
        assert meta.longitude is None

    def test_extract_metadata_with_mocked_gps(self, tmp_path):
        service = ImageMetadataService()
        test_file = tmp_path / "test_with_exif.jpg"
        test_file.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")

        # Mock PIL image returning EXIF with GPSInfo
        mock_img = MagicMock()
        # 37 deg 46 min 29.8 sec N, 122 deg 25 min 9.9 sec W (San Francisco)
        mock_img._getexif.return_value = {
            271: "Apple",
            272: "iPhone 13 Pro",
            306: "2026:10:23 10:30:00",
            34853: {
                1: "N",
                2: ((37, 1), (46, 1), (298, 10)),
                3: "W",
                4: ((122, 1), (25, 1), (99, 10)),
            },
        }

        with patch("PIL.Image.open", return_value=mock_img):
            meta = service.extract_metadata(str(test_file))

        assert meta.has_gps is True
        assert meta.latitude is not None and 37.7 <= meta.latitude <= 37.8
        assert meta.longitude is not None and -122.5 <= meta.longitude <= -122.4
        assert meta.location_provenance == "image_exif_metadata"
        assert meta.needs_location_confirmation is False
        assert meta.camera_make == "Apple"


class TestDisasterInvestigationValidation:
    def test_validate_image_upload_valid(self, test_image_bytes):
        service = DisasterInvestigationService()
        file_obj = UploadFile(
            filename="disaster_scene.png",
            file=io.BytesIO(test_image_bytes),
            headers={"content-type": "image/png"},
        )
        content = service.validate_image_upload(file_obj)
        assert len(content) == len(test_image_bytes)

    def test_validate_image_upload_unsupported_format(self):
        service = DisasterInvestigationService()
        file_obj = UploadFile(
            filename="report.pdf",
            file=io.BytesIO(b"%PDF-1.4 binary data"),
            headers={"content-type": "application/pdf"},
        )
        with pytest.raises(HTTPException) as exc_info:
            service.validate_image_upload(file_obj)
        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in exc_info.value.detail

    def test_validate_image_upload_empty_file(self):
        service = DisasterInvestigationService()
        file_obj = UploadFile(
            filename="empty.jpg",
            file=io.BytesIO(b""),
            headers={"content-type": "image/jpeg"},
        )
        with pytest.raises(HTTPException) as exc_info:
            service.validate_image_upload(file_obj)
        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()


class TestExternalDisasterService:
    def test_haversine_distance(self):
        # SF (37.77, -122.41) to San Jose (37.33, -121.89) is ~68-75 km
        d = _haversine_distance_km(37.77, -122.41, 37.33, -121.89)
        assert 60.0 <= d <= 80.0
        assert _haversine_distance_km(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_partial_failure_resilience(self):
        """Verify that when one provider fails, other providers still succeed."""
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = MagicMock(status_code=503)

        service = ExternalDisasterService(client=mock_client)
        alerts, status_rec = service.fetch_usgs_earthquakes(lat=37.77, lon=-122.41)
        assert len(alerts) == 0
        assert status_rec.status in ("UNAVAILABLE", "FAILED")


class TestInvestigationWorkflow:
    def test_location_investigation_mocked(self):
        from app.schemas.investigation import RecentImageryStatus
        mock_ext = MagicMock()
        mock_ext.geocode.return_value = {
            "display_name": "Antakya, Hatay, Turkey",
            "latitude": 36.20,
            "longitude": 36.16,
        }
        mock_ext.fetch_usgs_earthquakes.return_value = (
            [
                NormalizedDisasterAlert(
                    alert_id="usgs_123",
                    source="USGS",
                    event_type="Earthquake",
                    title="M 7.8 - Kahramanmaras, Turkey",
                    severity="CRITICAL",
                    latitude=37.17,
                    longitude=37.03,
                    event_location="Kahramanmaras, Turkey",
                    distance_km=110.0,
                    affects_location=True,
                    event_time="2023-02-06T01:17:35Z",
                    summary="Catastrophic earthquake with severe structural collapse.",
                )
            ],
            SourceCheckedStatus(provider="USGS", status="SUCCESS", records_count=1),
        )
        mock_ext.fetch_gdacs_alerts.return_value = ([], SourceCheckedStatus(provider="GDACS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_weather_hazards.return_value = ([], SourceCheckedStatus(provider="Open-Meteo", status="SUCCESS"))
        mock_ext.check_satellite_imagery_status.return_value = RecentImageryStatus(available=False)

        service = DisasterInvestigationService(external_service=mock_ext)
        req = LocationInvestigationRequest(location="Antakya, Turkey")
        ans = service.investigate_location(req)

        assert "Active disaster alert detected" in ans.direct_answer
        assert ans.confirmed_location == "Antakya, Hatay, Turkey"
        assert ans.latest_official_alert is not None
        assert ans.latest_official_alert.severity == "CRITICAL"
        assert len(ans.supporting_evidence) >= 1
        assert len(ans.sources_checked) >= 3


class TestRouteRiskService:
    def test_route_investigation_mocked(self):
        mock_ext = MagicMock()
        mock_ext.geocode.side_effect = [
            {"display_name": "San Francisco, CA", "latitude": 37.77, "longitude": -122.41},
            {"display_name": "San Jose, CA", "latitude": 37.33, "longitude": -121.89},
        ]
        mock_ext.fetch_route.return_value = (
            {
                "distance_km": 76.5,
                "duration_minutes": 55.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-122.41, 37.77], [-122.15, 37.55], [-121.89, 37.33]],
                },
                "alternatives": [],
            },
            SourceCheckedStatus(provider="OSRM", status="SUCCESS", records_count=1),
        )
        mock_ext.fetch_usgs_earthquakes.return_value = ([], SourceCheckedStatus(provider="USGS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_gdacs_alerts.return_value = ([], SourceCheckedStatus(provider="GDACS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_weather_hazards.return_value = ([], SourceCheckedStatus(provider="Open-Meteo", status="SUCCESS"))

        svc = RouteRiskService(external_service=mock_ext)
        req = RouteInvestigationRequest(
            origin="San Francisco, CA",
            destination="San Jose, CA",
            departure_time="2026-10-30",
        )
        resp = svc.investigate_route(req)

        assert resp.route_found is True
        assert resp.distance_km == 76.5
        assert resp.is_forward_looking is True
        assert resp.forward_looking_warning is not None
        # Safety enforcement: must not claim route is safe
        assert "SAFE" not in resp.safety_advisory.upper().split(" ")[0:2]
        assert "CAUTION" in resp.safety_advisory or "NO ACTIVE OFFICIAL DISASTER" in resp.safety_advisory
        assert resp.grounded_answer.direct_answer != ""

    def test_route_investigation_date_beyond_horizon(self):
        """Departure dates > 7 days in future trigger horizon warning in route advisory."""
        mock_ext = MagicMock()
        mock_ext.geocode.side_effect = [
            {"display_name": "San Francisco, CA", "latitude": 37.77, "longitude": -122.41},
            {"display_name": "San Jose, CA", "latitude": 37.33, "longitude": -121.89},
        ]
        mock_ext.fetch_route.return_value = (
            {
                "distance_km": 76.5,
                "duration_minutes": 55.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-122.41, 37.77], [-122.15, 37.55], [-121.89, 37.33]],
                },
                "alternatives": [],
            },
            SourceCheckedStatus(provider="OSRM", status="SUCCESS", records_count=1),
        )
        mock_ext.fetch_usgs_earthquakes.return_value = ([], SourceCheckedStatus(provider="USGS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_gdacs_alerts.return_value = ([], SourceCheckedStatus(provider="GDACS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_weather_hazards.return_value = ([], SourceCheckedStatus(provider="Open-Meteo", status="SUCCESS"))

        svc = RouteRiskService(external_service=mock_ext)
        # 14 days in the future
        future_date = (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d")
        req = RouteInvestigationRequest(
            origin="San Francisco, CA",
            destination="San Jose, CA",
            departure_time=future_date,
        )
        resp = svc.investigate_route(req)

        assert resp.is_forward_looking is True
        assert "7 days" in resp.forward_looking_warning or "horizon" in resp.forward_looking_warning.lower()

    def test_location_investigation_with_target_date(self):
        """Location investigation incorporates target_date into query and answer."""
        mock_ext = MagicMock()
        mock_ext.geocode.return_value = {
            "display_name": "Antakya, Hatay, Turkey",
            "latitude": 36.20,
            "longitude": 36.16,
        }
        mock_ext.fetch_usgs_earthquakes.return_value = ([], SourceCheckedStatus(provider="USGS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_gdacs_alerts.return_value = ([], SourceCheckedStatus(provider="GDACS", status="NO_RECORDS_FOUND"))
        mock_ext.fetch_weather_hazards.return_value = ([], SourceCheckedStatus(provider="Open-Meteo", status="SUCCESS"))
        from app.schemas.investigation import RecentImageryStatus
        mock_ext.check_satellite_imagery_status.return_value = RecentImageryStatus(available=False)

        service = DisasterInvestigationService(external_service=mock_ext)
        req = LocationInvestigationRequest(location="Antakya, Turkey", target_date="tomorrow")
        ans = service.investigate_location(req)

        assert ans.confirmed_location == "Antakya, Hatay, Turkey"
        assert "For intended date tomorrow" in ans.direct_answer
        mock_ext.fetch_weather_hazards.assert_called_once_with(lat=36.20, lon=36.16, departure_date="tomorrow")

    def test_open_meteo_horizon_exceeded_logic(self):
        """Weather service returns Forecast Horizon Exceeded alert when target date > 7 days."""
        ext_svc = ExternalDisasterService()
        # 12 days in future
        far_date = (datetime.now(timezone.utc) + timedelta(days=12)).strftime("%Y-%m-%d")
        alerts, status = ext_svc.fetch_weather_hazards(lat=36.2, lon=36.1, departure_date=far_date)

        assert status.status == "SUCCESS"
        assert len(alerts) == 1
        assert alerts[0].event_type == "Forecast Horizon Exceeded"
        assert "exceeds" in alerts[0].summary.lower()
