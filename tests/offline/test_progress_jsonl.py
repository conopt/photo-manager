"""JsonlProgressDashboard: every dashboard hook works, output is JSON lines on stderr only."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from cull.dashboard import (
    Dashboard,
    DashboardLaunchInfo,
    _S4UpdateInput,
    _Stage2ReducerUpdateInput,
    _Stage2UpdateInput,
)
from cull.models import (
    BlurScores,
    ExposureScores,
    Stage1Result,
    Stage2Result,
    Stage3Result,
)
from cull.progress_jsonl import JsonlProgressDashboard
from cull.stage2.fusion import FusionResult

PHOTO: Path = Path("/photos/a.jpg")
STAGE1 = Stage1Result(
    photo_path=PHOTO,
    blur=BlurScores(tenengrad=500.0, fft_ratio=0.3, blur_tier=1),
    exposure=ExposureScores(
        dr_score=0.8, clipping_highlight=0.0, clipping_shadow=0.0, midtone_pct=0.6, color_cast_score=0.0,
    ),
    noise_score=0.01,
)
FUSION = FusionResult(
    stage2=Stage2Result(photo_path=PHOTO, topiq=0.9, laion_aesthetic=0.9, clipiqa=0.9, composite=0.9),
    routing="AMBIGUOUS",
)
STAGE3 = Stage3Result(photo_path=PHOTO, is_keeper=True, confidence=0.9)

# One call per public Dashboard method, in pipeline order.
CALLS: list[tuple[str, tuple]] = [
    ("begin_scan", ()),
    ("update_scan_progress", (1, 1000)),
    ("end_scan", ()),
    ("set_photo_count", (1,)),
    ("start_scanning", ()),
    ("start_stage1", (1,)),
    ("update_stage1", (PHOTO, STAGE1)),
    ("set_dupe_count", (0,)),
    ("set_burst_count", (0,)),
    ("refresh", ()),
    ("stop_scanning", ()),
    ("complete_stage1", (0.5,)),
    ("start_stage2_loading", ()),
    ("clear_stage2_loading", ()),
    ("start_stage2", (1,)),
    ("update_stage2", (_Stage2UpdateInput(path=PHOTO, fusion=FUSION, routing="AMBIGUOUS"),)),
    ("complete_stage2", (0.5,)),
    ("start_stage2_reducer", (1,)),
    ("update_stage2_reducer", (_Stage2ReducerUpdateInput(),)),
    ("complete_stage2_reducer", (0.1,)),
    ("start_stage3", (1, "vision-model")),
    ("start_analysis", (PHOTO,)),
    ("complete_analysis", (PHOTO, STAGE3)),
    ("complete_stage3", (2.0,)),
    ("start_stage4", (1,)),
    ("start_stage4_peak", (1,)),
    ("update_stage4_peak", (1,)),
    ("complete_stage4_peak", (0.1,)),
    ("start_stage4_diversity", (1,)),
    ("update_stage4_diversity", (1,)),
    ("complete_stage4_diversity", (0.1,)),
    ("start_stage4_tournament", (1,)),
    ("update_stage4_tournament", (1,)),
    ("complete_stage4_tournament", (0.1,)),
    ("start_stage4_narrative", (1,)),
    ("update_stage4_narrative", (1,)),
    ("complete_stage4_narrative", (0.1,)),
    ("update_stage4", (_S4UpdateInput(selected=1),)),
    ("complete_stage4", (0.4,)),
    ("show_results", (None,)),
]


def _launch() -> DashboardLaunchInfo:
    """Return minimal launch info."""
    return DashboardLaunchInfo(source_path="/photos", photo_count=0, preset="general", file_size_gb=0.0)


def test_calls_cover_every_public_dashboard_method() -> None:
    """The exercised hooks are exactly the live dashboard's public methods."""
    public = {name for name, _ in inspect.getmembers(Dashboard, inspect.isfunction) if not name.startswith("_")}
    assert public == {name for name, _ in CALLS}


def test_every_hook_emits_json_lines_on_stderr_without_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """All hooks run; stdout stays empty and every stderr line is a progress object."""
    started = time.monotonic()
    with JsonlProgressDashboard(_launch()) as dashboard:
        for name, args in CALLS:
            getattr(dashboard, name)(*args)
    assert time.monotonic() - started < 1.0, "the live display's per-photo gauge hold must not apply"
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [json.loads(line) for line in captured.err.splitlines()]
    assert {line["stage"] for line in lines} == {"scan", "stage1", "stage2", "stage3", "stage4"}
    assert all({"stage", "done", "total", "event"} <= set(line) for line in lines)
    assert {"stage": "stage3", "done": 1, "total": 1, "event": "update"} in lines
    assert {"stage": "stage4", "done": 1, "total": 1, "event": "complete", "substage": "narrative"} in lines


def test_escaping_exception_is_reported_as_error_line(capsys: pytest.CaptureFixture[str]) -> None:
    """A failure inside the run becomes a final error line at the last reported stage."""
    with pytest.raises(RuntimeError), JsonlProgressDashboard(_launch()) as dashboard:
        dashboard.start_stage1(3)
        raise RuntimeError("decoder failed")
    last = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert last == {"stage": "stage1", "done": 0, "total": 3, "event": "error", "message": "decoder failed"}
