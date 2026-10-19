"""Deterministic stand-ins for the model-backed parts of Stages 1-2, plus a local VLM endpoint.

Only the per-photo model calls are replaced: the Stage 1 worker pool, the
duplicate detector, burst confirmation, Stage 2 model warm-up and the
batched CLIP/IQA forward. Everything around them -- dashboard calls, stack
resolution, the Stage 2 reducer, routing, the search-cache write, Stage 3
over HTTP, decisions, Stage 4 and file routing -- is the real code.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel, Field

from cull._pipeline.stage1_runner import _sort_by_path, _Stage1LoopInput, _Stage1Output
from cull._pipeline.stage2_runner import (
    _BatchCtx,
    _DualPrefetch,
    _emit_batch_results,
    _EmitInput,
)
from cull.config import IQA_EXPOSURE_DEFAULT, CullConfig
from cull.models import BlurScores, ExifSummary, ExposureScores, Stage1Result
from cull.stage1.duplicate import DuplicateGroup, DuplicateResult
from cull.stage2.fusion import FusionResult, IqaScores, compute_composite

ENCODING_DIM: int = 16
CLIP_DIM: int = 12
IMAGE_SIZE: tuple[int, int] = (64, 48)
SHOOT_START: datetime = datetime(2026, 5, 1, 14, 0, 0)
ENDPOINT_MODEL: str = "example-vision-model"


def _digest(text: str) -> int:
    """Return a process-stable integer derived from text."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _unit(text: str, salt: str) -> float:
    """Return a stable value in [0, 1) for (text, salt)."""
    return (_digest(f"{salt}:{text}") % 10_000) / 10_000


def _vector(name: str, dim: int) -> np.ndarray:
    """Return a stable float32 vector for a photo name."""
    rng = np.random.default_rng(_digest(name) % (2**32))
    return rng.standard_normal(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Generated photos
# ---------------------------------------------------------------------------


def _write_image(path: Path, index: int) -> None:
    """Write a small gradient image whose colours depend on index."""
    width, height = IMAGE_SIZE
    xs = np.linspace(0, 255, width, dtype=np.float32)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = (xs[None, :] * (index + 1) / 9) % 256
    pixels[..., 1] = (index * 37) % 256
    pixels[..., 2] = np.linspace(255, 0, height, dtype=np.float32)[:, None]
    fmt = "HEIF" if path.suffix.lower() == ".heic" else "JPEG"
    Image.fromarray(pixels).save(path, format=fmt)


def write_staging_corpus(root: Path) -> Path:
    """Create originals outside a staging SOURCE that links to them plus holds previews.

    Returns SOURCE: symlinks to JPEG/HEIF originals, regular JPEG previews,
    and one nested directory.
    """
    originals = root / "originals"
    source = root / "staging"
    originals.mkdir()
    (source / "day2").mkdir(parents=True)
    names = ["a01.jpg", "a02.jpg", "a03.jpg", "a04.heic", "a05.jpg", "a06.jpg"]
    for index, name in enumerate(names):
        _write_image(originals / name, index)
        (source / name).symlink_to(originals / name)
    for index, name in enumerate(["day2/b01.jpg", "day2/b02.jpg", "preview03.jpg"], start=len(names)):
        _write_image(source / name, index)
    return source


# ---------------------------------------------------------------------------
# Fake Stage 1 / Stage 2 model calls
# ---------------------------------------------------------------------------


class FakeCalls(BaseModel):
    """How often each faked model-backed call ran."""

    stage1_loops: int = 0
    duplicate_scans: int = 0
    stage2_prewarms: int = 0
    stage2_chunks: int = 0
    is_forbidden: bool = False

    def hit(self, name: str) -> None:
        """Count one call, or fail when model-backed stages are forbidden."""
        if self.is_forbidden:
            raise AssertionError(f"{name} ran although Stages 1-2 must not run")
        setattr(self, name, getattr(self, name) + 1)


def _stage1_result(path: Path, index: int) -> Stage1Result:
    """Return a deterministic Stage 1 result; the third photo is a blur reject."""
    name = path.name
    is_reject = index == 2
    return Stage1Result(
        photo_path=path,
        blur=BlurScores(tenengrad=100 + 900 * _unit(name, "ten"), fft_ratio=_unit(name, "fft"), blur_tier=1),
        exposure=ExposureScores(
            dr_score=0.5 + _unit(name, "dr") / 2, clipping_highlight=0.01, clipping_shadow=0.02,
            midtone_pct=0.6, color_cast_score=_unit(name, "cast") / 10,
        ),
        noise_score=_unit(name, "noise") / 10,
        is_pass=not is_reject,
        reject_reason="blur" if is_reject else None,
        capture_time=SHOOT_START + timedelta(seconds=index * 2),
        exif=ExifSummary(iso=200 + 100 * (index % 3), shutter="1/250", aperture="f/2.8", focal_length_mm=35.0),
    )


class _FakeStage1(BaseModel):
    """Callable replacements for the Stage 1 worker pool, duplicate detector and burst check."""

    calls: FakeCalls

    def loop(self, loop_in: _Stage1LoopInput, dashboard: Any) -> _Stage1Output:
        """Assess every photo like the worker pool, in path order."""
        self.calls.hit("stage1_loops")
        output = _Stage1Output()
        dashboard.start_stage1(len(loop_in.paths))
        for index, path in enumerate(loop_in.paths):
            result = _stage1_result(path, index)
            output.results[str(path)] = result
            (output.survivors if result.is_pass else output.rejected).append(path)
            dashboard.update_stage1(path, result)
        _sort_by_path(output)
        return output

    def duplicates(self, loop_in: _Stage1LoopInput) -> DuplicateResult:
        """Group the first two photos as near-duplicates; embed every photo."""
        self.calls.hit("duplicate_scans")
        paths = loop_in.paths
        return DuplicateResult(
            duplicate_groups=[DuplicateGroup(paths=paths[:2])],
            encodings={str(p): _vector(p.name, ENCODING_DIM) for p in paths},
        )

    def bursts(self, loop_in: _Stage1LoopInput, output: _Stage1Output) -> list[list[str]]:
        """Confirm one burst among the survivors."""
        survivors = [str(p) for p in output.survivors]
        return [survivors[3:5]]


def _fusion(path: Path, config: CullConfig) -> FusionResult:
    """Fuse deterministic metric scores with the real fusion; composites span all three routes."""
    name = path.name
    scores = IqaScores(
        photo_path=path, topiq=0.80 + 0.18 * _unit(name, "topiq"),
        laion_aesthetic=0.80 + 0.18 * _unit(name, "laion"), clipiqa=0.80 + 0.18 * _unit(name, "clipiqa"),
        exposure=IQA_EXPOSURE_DEFAULT,
        palette_lab=(50 + 10 * _unit(name, "L"), _unit(name, "a"), -_unit(name, "b")),
    )
    return compute_composite(scores, config)


class _FakeStage2(BaseModel):
    """Callable replacements for Stage 2 model warm-up and the batched forward."""

    calls: FakeCalls

    def prewarm(self, loop_in: Any, dashboard: Any) -> None:
        """Report model loading like the real warm-up."""
        self.calls.hit("stage2_prewarms")
        dashboard.start_stage2_loading()
        dashboard.clear_stage2_loading()

    def score_chunks(self, prefetch: _DualPrefetch, emit_in: _EmitInput) -> _BatchCtx:
        """Score each chunk and collect CLIP rows, emitting results like the real loop."""
        batch_ctx = _BatchCtx(loop_in=emit_in.loop_in, cache=emit_in.output.search_cache)
        for chunk in prefetch.chunks:
            self.calls.hit("stage2_chunks")
            pairs = [(path, _fusion(path, emit_in.loop_in.config)) for path in chunk]
            for path in chunk:
                row = _vector(path.name, CLIP_DIM)
                batch_ctx.embedding_rows.append((path, row / np.linalg.norm(row)))
            _emit_batch_results(emit_in.model_copy(update={"pairs": pairs}))
        return batch_ctx


def install_fake_local_stages(monkeypatch: pytest.MonkeyPatch) -> FakeCalls:
    """Replace the model-backed Stage 1-2 calls; return their call counter."""
    calls = FakeCalls()
    stage1 = _FakeStage1(calls=calls)
    stage2 = _FakeStage2(calls=calls)
    monkeypatch.setattr("cull._pipeline.stage1_runner._run_stage1_loop", stage1.loop)
    monkeypatch.setattr("cull._pipeline.stage1_runner._find_duplicates_for", stage1.duplicates)
    monkeypatch.setattr("cull._pipeline.stage1_runner._detect_burst_groups", stage1.bursts)
    monkeypatch.setattr("cull._pipeline.stage2_runner._prewarm_stage2_models", stage2.prewarm)
    monkeypatch.setattr("cull._pipeline.stage2_runner._score_chunks", stage2.score_chunks)
    return calls


# ---------------------------------------------------------------------------
# Local OpenAI-compatible endpoint
# ---------------------------------------------------------------------------


class EndpointLog(BaseModel):
    """Requests the local endpoint received."""

    models: list[str] = Field(default_factory=list)


def _reply_for(body: dict) -> str:
    """Return a deterministic VLM reply derived from the request's image bytes."""
    content = body["messages"][0]["content"]
    images = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
    seed = _digest("".join(images))
    return json.dumps({
        "sharpness": 0.7, "exposure": 0.6, "composition": 0.8, "expression": 0.5,
        "keeper": seed % 2 == 0, "confidence": 0.8, "flags": [],
        "rating": 1 + seed % 5,
    })


@contextmanager
def vlm_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[EndpointLog]:
    """Serve deterministic chat completions on 127.0.0.1 and point cull at them.

    Any local-model registry lookup or MLX load fails the test.
    """
    log = EndpointLog()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.models.append(body["model"])
            reply = {"choices": [{"message": {"content": _reply_for(body)}}]}
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PHOTO_MANAGER_VLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.delenv("PHOTO_MANAGER_VLM_API_KEY", raising=False)
    monkeypatch.delenv("PHOTO_MANAGER_VLM_TIMEOUT", raising=False)

    def local_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("local VLM registry or MLX access attempted")

    monkeypatch.setattr("cull.vlm_registry.discover_vlms", local_forbidden)
    monkeypatch.setattr("cull.vlm_session._get_mlx_vlm", local_forbidden)
    try:
        yield log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

