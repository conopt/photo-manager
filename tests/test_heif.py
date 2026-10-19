"""Generated HEIF fixtures exercise the real decoder without personal images."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms
import pillow_heif

from cull.image_io import open_rgb_upright, read_cv_image
from cull.disk import scan_jpegs
from cull.pipeline import _scan_with_dashboard
from cull.search import _scan_jpegs
from cull.stage1.burst import _read_exif_capture, _dhash
from cull.stage1.worker import _decode_once, _read_exif_summary
from cull.tui.previews import PixelBox, PreviewSpec, decode_fitted


def _write_heif(path: Path, *, orientation: int = 1) -> None:
    pixels = np.zeros((64, 96, 3), dtype=np.uint8)
    pixels[:, :48] = (190, 30, 20)
    pixels[:, 48:] = (20, 70, 190)
    exif = Image.Exif()
    exif[274] = orientation
    exif[34665] = {36867: "2026:08:01 12:34:56", 37521: "125", 34855: 200}
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(pixels).save(
        path, format="HEIF", quality=-1, exif=exif.tobytes(), icc_profile=profile,
    )


@pytest.mark.parametrize("suffix", [".hif", ".HEIF", ".HeIc"])
def test_heif_discovery_pixels_metadata_and_original_preservation(tmp_path: Path, suffix: str) -> None:
    photo = tmp_path / ("sample" + suffix)
    _write_heif(photo)
    before = hashlib.sha256(photo.read_bytes()).digest()
    assert photo.read_bytes()[4:8] == b"ftyp"
    assert _scan_with_dashboard(tmp_path, MagicMock()) == [photo]
    assert _scan_jpegs(tmp_path) == [photo]
    assert scan_jpegs(tmp_path) == [photo]

    rgb = np.asarray(open_rgb_upright(photo))
    assert rgb.shape == (64, 96, 3)
    assert np.max(np.abs(rgb[30, 20].astype(int) - (190, 30, 20))) < 8
    bgr = read_cv_image(photo)
    assert np.array_equal(bgr, rgb[:, :, ::-1])
    decoded = _decode_once(photo)
    assert decoded.full_gray.shape == (64, 96)
    assert np.array_equal(decoded.full_gray, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    capture = _read_exif_capture(photo)
    assert capture is not None and capture.has_subsec
    assert capture.time == datetime(2026, 8, 1, 12, 34, 56, 125000)
    time, summary = _read_exif_summary(photo)
    assert time == datetime(2026, 8, 1, 12, 34, 56)
    assert summary is not None and summary.iso == 200
    assert _dhash(photo) is not None
    assert hashlib.sha256(photo.read_bytes()).digest() == before


def test_heif_orientation_matches_analysis_and_preview(tmp_path: Path) -> None:
    photo = tmp_path / "rotated.hif"
    _write_heif(photo, orientation=6)
    upright = open_rgb_upright(photo)
    assert upright.size == (64, 96)
    assert read_cv_image(photo).shape[:2] == (96, 64)
    assert read_cv_image(photo, cv2.IMREAD_GRAYSCALE).shape == (96, 64)
    # Preview construction is kept independent of any real library or session.
    spec = PreviewSpec(source=photo, box=PixelBox(width=128, height=128))
    preview = decode_fitted(spec)
    assert preview.size == upright.size
    assert np.array_equal(np.asarray(preview), np.asarray(upright))


def test_corrupt_heif_does_not_decode(tmp_path: Path) -> None:
    path = tmp_path / "broken.hif"
    path.write_bytes(b"not an image")
    assert read_cv_image(path) is None


def test_jpeg_opencv_pixels_are_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), (30, 70, 120)).save(path)
    assert np.array_equal(read_cv_image(path), cv2.imread(str(path)))
    assert np.array_equal(
        read_cv_image(path, cv2.IMREAD_GRAYSCALE),
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE),
    )


def test_ten_bit_heif_is_decoded_for_analysis(tmp_path: Path) -> None:
    path = tmp_path / "ten-bit.hif"
    pixels = np.full((32, 48, 3), 32768, dtype=np.uint16)
    image = pillow_heif.from_bytes("RGB;16", (48, 32), pixels.tobytes())
    image.save(path, quality=-1, bit_depth=10)
    source = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)
    assert source.info["bit_depth"] == 10
    decoded = np.asarray(open_rgb_upright(path))
    assert decoded.dtype == np.uint8 and decoded.shape == (32, 48, 3)
    assert np.max(np.abs(decoded.astype(int) - 128)) < 3


def test_mixed_heif_and_jpeg_reach_duplicate_encoder(tmp_path: Path) -> None:
    from cull.stage1.duplicate import _run_cnn_encoding

    _write_heif(tmp_path / "sample.hif", orientation=6)
    Image.new("RGB", (48, 32), "green").save(tmp_path / "sample.jpg")
    excluded = tmp_path / "_review"
    excluded.mkdir()
    _write_heif(excluded / "old.hif")
    cnn = MagicMock()
    cnn.encode_images.side_effect = AssertionError("directory loader does not accept HEIF")
    seen = []
    def encode(*, image_array):
        seen.append(image_array.shape)
        return np.array([1.0, 0.0])
    cnn.encode_image.side_effect = encode
    cnn.find_duplicates.return_value = {}
    encodings, _ = _run_cnn_encoding(cnn, tmp_path)
    assert set(encodings) == {"sample.hif", "sample.jpg"}
    assert seen == [(96, 64, 3), (32, 48, 3)]
