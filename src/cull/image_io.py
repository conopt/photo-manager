"""Shared JPEG/HEIF pixels and EXIF for analysis, previews, and model inputs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import exifread
import numpy as np
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener

from cull.config import HEIF_EXTENSIONS

# Registration is also needed by metadata-only and draft-thumbnail readers.
register_heif_opener()


def is_heif(path: Path) -> bool:
    """Return whether the path uses a supported HEIF extension."""
    return path.suffix.lower() in HEIF_EXTENSIONS


def to_rgb_upright(image: Image.Image) -> Image.Image:
    """Apply orientation and convert an embedded ICC profile to sRGB for HEIF."""
    upright = ImageOps.exif_transpose(image).convert("RGB")
    if image.format == "HEIF" and image.info.get("icc_profile"):
        upright = ImageCms.profileToProfile(
            upright,
            ImageCms.ImageCmsProfile(BytesIO(image.info["icc_profile"])),
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
    return upright


def open_rgb_upright(path: Path) -> Image.Image:
    """Decode path as RGB with its EXIF Orientation applied.

    cv2.imread honours EXIF orientation and PIL does not, so without this the
    PIL-fed models see portrait-orientation frames sideways while the cv2
    stages see them upright.
    """
    with Image.open(path) as img:
        return to_rgb_upright(img)


def read_cv_image(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode HEIF through libheif; preserve OpenCV's existing JPEG decode path."""
    path = Path(path)
    if not is_heif(path):
        return cv2.imread(str(path), flags)
    try:
        rgb = np.asarray(open_rgb_upright(path))
    except (OSError, ValueError):
        return None
    if flags == cv2.IMREAD_GRAYSCALE:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_exif_tags(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Expose container EXIF as the same exifread tags used for JPEG metadata."""
    if is_heif(path):
        with Image.open(path) as image:
            exif = image.info.get("exif")
        if not exif:
            return {}
        if exif.startswith(b"Exif\x00\x00"):
            exif = exif[6:]
        return exifread.process_file(BytesIO(exif), **kwargs)
    with path.open("rb") as stream:
        return exifread.process_file(stream, **kwargs)
