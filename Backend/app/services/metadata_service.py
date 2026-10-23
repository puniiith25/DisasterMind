"""
Image Metadata Extraction Service.

Extracts trustworthy EXIF metadata (GPS coordinates, capture date, camera make/model)
from uploaded disaster imagery using Pillow. Never fabricates coordinates.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from app.schemas.investigation import ImageMetadataInfo

logger = logging.getLogger(__name__)


def _convert_to_degrees(value: Any) -> Optional[float]:
    """Helper function to convert GPS coordinates in EXIF format to decimal degrees."""
    if not value or len(value) < 3:
        return None

    try:
        def to_float(val: Any) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            # Tuple or IFDRational (numerator, denominator)
            if hasattr(val, "numerator") and hasattr(val, "denominator"):
                return float(val.numerator) / float(val.denominator)
            if isinstance(val, tuple) and len(val) == 2:
                return float(val[0]) / float(val[1])
            return float(val)

        d = to_float(value[0])
        m = to_float(value[1])
        s = to_float(value[2])

        return d + (m / 60.0) + (s / 3600.0)
    except Exception as err:
        logger.debug(f"Error converting GPS coordinate {value}: {err}")
        return None


class ImageMetadataService:
    """Service for extracting factual metadata and provenance from image files."""

    def extract_metadata(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO],
    ) -> ImageMetadataInfo:
        """
        Extract trustworthy EXIF metadata from image file or raw bytes.
        Returns ImageMetadataInfo with explicit provenance.
        """
        try:
            if isinstance(image_input, (str, Path)):
                p = Path(image_input)
                if not p.exists():
                    return ImageMetadataInfo(
                        has_gps=False,
                        location_provenance="absent",
                        needs_location_confirmation=True,
                        notes="Image file not found on disk.",
                    )
                img = Image.open(p)
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            else:
                img = Image.open(image_input)

            exif_raw = img._getexif()
            if not exif_raw:
                return ImageMetadataInfo(
                    has_gps=False,
                    location_provenance="absent",
                    needs_location_confirmation=True,
                    notes="No EXIF metadata embedded in image.",
                )

            # Map raw numeric tag keys to human-readable names
            exif_data: Dict[str, Any] = {}
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value

            # Extract Camera Info
            camera_make = str(exif_data.get("Make", "")).strip() or None
            camera_model = str(exif_data.get("Model", "")).strip() or None

            # Extract Timestamp
            capture_time = (
                exif_data.get("DateTimeOriginal")
                or exif_data.get("DateTimeDigitized")
                or exif_data.get("DateTime")
            )
            if capture_time:
                capture_time = str(capture_time).strip()

            # Extract GPS Info
            gps_info = exif_data.get("GPSInfo")
            if not gps_info:
                return ImageMetadataInfo(
                    has_gps=False,
                    capture_time=capture_time,
                    camera_make=camera_make,
                    camera_model=camera_model,
                    location_provenance="absent",
                    needs_location_confirmation=True,
                    notes="Image contains EXIF metadata, but no GPS geolocation tags.",
                )

            # Map GPS sub-tags
            gps_data: Dict[str, Any] = {}
            for gps_tag_id, value in gps_info.items():
                sub_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_data[sub_tag_name] = value

            lat_val = gps_data.get("GPSLatitude")
            lat_ref = gps_data.get("GPSLatitudeRef", "N")
            lon_val = gps_data.get("GPSLongitude")
            lon_ref = gps_data.get("GPSLongitudeRef", "E")

            lat = _convert_to_degrees(lat_val)
            lon = _convert_to_degrees(lon_val)

            if lat is not None and lon is not None:
                if str(lat_ref).upper().startswith("S"):
                    lat = -lat
                if str(lon_ref).upper().startswith("W"):
                    lon = -lon

                # Sanity check ranges
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return ImageMetadataInfo(
                        has_gps=True,
                        latitude=round(lat, 6),
                        longitude=round(lon, 6),
                        capture_time=capture_time,
                        camera_make=camera_make,
                        camera_model=camera_model,
                        location_provenance="image_exif_metadata",
                        needs_location_confirmation=False,
                        notes="Geolocation successfully derived from image EXIF metadata.",
                    )

            return ImageMetadataInfo(
                has_gps=False,
                capture_time=capture_time,
                camera_make=camera_make,
                camera_model=camera_model,
                location_provenance="absent",
                needs_location_confirmation=True,
                notes="GPS tags present but coordinates could not be decoded.",
            )

        except Exception as exc:
            logger.warning(f"Failed to read image EXIF metadata: {exc}")
            return ImageMetadataInfo(
                has_gps=False,
                location_provenance="absent",
                needs_location_confirmation=True,
                notes=f"Metadata extraction error: {str(exc)}",
            )


_IMAGE_METADATA_SERVICE: Optional[ImageMetadataService] = None


def get_metadata_service() -> ImageMetadataService:
    global _IMAGE_METADATA_SERVICE
    if _IMAGE_METADATA_SERVICE is None:
        _IMAGE_METADATA_SERVICE = ImageMetadataService()
    return _IMAGE_METADATA_SERVICE
