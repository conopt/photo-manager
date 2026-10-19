"""Machine-readable progress: one JSON object per line on stderr.

``JsonlProgressDashboard`` keeps the live ``Dashboard``'s full method surface
and bookkeeping (it subclasses it) but never starts the Rich display. Each
progress event is written as one flushed line, for example::

    {"stage": "stage2", "done": 12, "total": 40, "event": "update"}

``stage`` is one of ``scan``, ``stage1``, ``stage2``, ``stage3``, ``stage4``;
``total`` is ``null`` when unknown. ``event`` is ``start``, ``update``,
``complete`` or ``error``; sub-steps add a ``substage`` key, and some events
add ``seconds``, ``model``, ``target`` or ``message``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from cull.dashboard import (
    Dashboard,
    DashboardLaunchInfo,
    _S4UpdateInput,
    _Stage2ReducerUpdateInput,
    _Stage2UpdateInput,
    _StageState,
)
from cull.models import Stage1Result, Stage3Result

STAGE_SCAN: str = "scan"
STAGE_1: str = "stage1"
STAGE_2: str = "stage2"
STAGE_3: str = "stage3"
STAGE_4: str = "stage4"
EVENT_START: str = "start"
EVENT_UPDATE: str = "update"
EVENT_COMPLETE: str = "complete"
EVENT_ERROR: str = "error"


class ProgressEvent(BaseModel):
    """One progress line; ``extra`` keys are merged into the emitted object."""

    stage: str
    done: int
    total: int | None
    event: str
    extra: dict[str, Any] = {}


def write_progress_line(stream: TextIO, event: ProgressEvent) -> None:
    """Write one event as a single flushed JSON line."""
    payload = {"stage": event.stage, "done": event.done, "total": event.total, "event": event.event}
    payload.update(event.extra)
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def emit_progress_error(message: str) -> None:
    """Report a failure outside a dashboard's lifetime as a JSON error line on stderr."""
    write_progress_line(sys.stderr, ProgressEvent(
        stage=STAGE_SCAN, done=0, total=None, event=EVENT_ERROR, extra={"message": message},
    ))


class JsonlProgressDashboard(Dashboard):
    """Dashboard that reports progress as JSON lines instead of a live display."""

    def __init__(self, launch_info: DashboardLaunchInfo) -> None:
        """Pin the real stderr now: model loaders redirect sys.stderr while they run."""
        super().__init__(launch_info)
        self._stream: TextIO = sys.stderr
        self._last: ProgressEvent = ProgressEvent(stage=STAGE_SCAN, done=0, total=None, event=EVENT_START)

    def __enter__(self) -> JsonlProgressDashboard:
        """Silence library logging (it would interleave with the JSON lines)."""
        self._suppress_library_logs()
        return self

    def __exit__(self, *args: object) -> None:
        """Report an escaping exception, then restore logging state."""
        exc = args[1] if len(args) > 1 else None
        if isinstance(exc, Exception):
            self.report_error(str(exc) or type(exc).__name__)
        self._restore_library_logs()

    def _emit(self, event: ProgressEvent) -> None:
        """Write one line and remember it as the latest position."""
        self._last = event
        write_progress_line(self._stream, event)

    def _emit_stage(self, stage: str, state: _StageState, event: str) -> None:
        """Emit a stage's tracked done/total."""
        self._emit(ProgressEvent(stage=stage, done=state.done, total=state.total, event=event))

    def _emit_sub(self, substage: str, state: _StageState, event: str) -> None:
        """Emit a Stage 4 sub-step's done/total."""
        self._emit(ProgressEvent(
            stage=STAGE_4, done=state.done, total=state.total, event=event, extra={"substage": substage},
        ))

    def report_error(self, message: str) -> None:
        """Emit an error line at the last reported stage."""
        last = self._last
        self._emit(ProgressEvent(
            stage=last.stage, done=last.done, total=last.total, event=EVENT_ERROR, extra={"message": message},
        ))

    # --- Scan ------------------------------------------------------------

    def begin_scan(self) -> None:
        """Report the start of the file scan."""
        super().begin_scan()
        self._emit(ProgressEvent(stage=STAGE_SCAN, done=0, total=None, event=EVENT_START))

    def update_scan_progress(self, count: int, total_bytes: int) -> None:
        """Report one more photo found."""
        super().update_scan_progress(count, total_bytes)
        self._emit(ProgressEvent(stage=STAGE_SCAN, done=count, total=None, event=EVENT_UPDATE))

    def end_scan(self) -> None:
        """Report the number of photos found."""
        super().end_scan()
        count = self._scan_count
        self._emit(ProgressEvent(stage=STAGE_SCAN, done=count, total=count, event=EVENT_COMPLETE))

    # --- Stage 1 ---------------------------------------------------------

    def start_stage1(self, total: int) -> None:
        """Report the Stage 1 photo count."""
        super().start_stage1(total)
        self._emit_stage(STAGE_1, self._s1, EVENT_START)

    def update_stage1(self, path: Path, result: Stage1Result) -> None:
        """Report one Stage 1 result."""
        super().update_stage1(path, result)
        self._emit_stage(STAGE_1, self._s1, EVENT_UPDATE)

    def complete_stage1(self, elapsed: float) -> None:
        """Report Stage 1 completion."""
        super().complete_stage1(elapsed)
        self._emit_complete(STAGE_1, self._s1, elapsed)

    def _emit_complete(self, stage: str, state: _StageState, elapsed: float) -> None:
        """Emit a stage completion with its elapsed seconds."""
        self._emit(ProgressEvent(
            stage=stage, done=state.done, total=state.total, event=EVENT_COMPLETE,
            extra={"seconds": round(elapsed, 3)},
        ))

    # --- Stage 2 ---------------------------------------------------------

    def start_stage2_loading(self) -> None:
        """Report that Stage 2 models are loading."""
        super().start_stage2_loading()
        self._emit(ProgressEvent(stage=STAGE_2, done=0, total=None, event=EVENT_START, extra={"substage": "loading"}))

    def start_stage2(self, total: int) -> None:
        """Report the Stage 2 photo count."""
        super().start_stage2(total)
        self._emit_stage(STAGE_2, self._s2, EVENT_START)

    def update_stage2(self, update_in: _Stage2UpdateInput) -> None:
        """Report one Stage 2 result."""
        super().update_stage2(update_in)
        self._emit_stage(STAGE_2, self._s2, EVENT_UPDATE)

    def complete_stage2(self, elapsed: float) -> None:
        """Report Stage 2 completion."""
        super().complete_stage2(elapsed)
        self._emit_complete(STAGE_2, self._s2, elapsed)

    def start_stage2_reducer(self, total: int) -> None:
        """Report the start of the shoot-level reducer pass."""
        super().start_stage2_reducer(total)
        self._emit_reducer(EVENT_START)

    def update_stage2_reducer(self, update_in: _Stage2ReducerUpdateInput) -> None:
        """Report one reducer update."""
        super().update_stage2_reducer(update_in)
        self._emit_reducer(EVENT_UPDATE)

    def complete_stage2_reducer(self, elapsed: float) -> None:
        """Report reducer completion."""
        super().complete_stage2_reducer(elapsed)
        self._emit_reducer(EVENT_COMPLETE)

    def _emit_reducer(self, event: str) -> None:
        """Emit the reducer's done/total as a Stage 2 sub-step."""
        state = self._s2_reducer
        self._emit(ProgressEvent(
            stage=STAGE_2, done=state.done, total=state.total, event=event, extra={"substage": "reducer"},
        ))

    # --- Stage 3 ---------------------------------------------------------

    def start_stage3(self, total: int, model_name: str) -> None:
        """Report the Stage 3 queue length and model."""
        super().start_stage3(total, model_name)
        state = self._s3
        self._emit(ProgressEvent(
            stage=STAGE_3, done=state.done, total=state.total, event=EVENT_START, extra={"model": model_name},
        ))

    def complete_analysis(self, path: Path, result: Stage3Result) -> None:
        """Report one Stage 3 result, without the live display's gauge hold."""
        self._s3_tracker._complete_analysis(path, result)
        self._s3_tracker._clear_analysis()
        self._emit_stage(STAGE_3, self._s3, EVENT_UPDATE)

    def complete_stage3(self, elapsed: float) -> None:
        """Report Stage 3 completion."""
        super().complete_stage3(elapsed)
        self._emit_complete(STAGE_3, self._s3, elapsed)

    # --- Stage 4 ---------------------------------------------------------

    def start_stage4(self, target: int) -> None:
        """Report the start of curation and its target."""
        super().start_stage4(target)
        self._emit(ProgressEvent(stage=STAGE_4, done=0, total=None, event=EVENT_START, extra={"target": target}))

    def update_stage4(self, tally_update: _S4UpdateInput) -> None:
        """Report the selections made so far."""
        super().update_stage4(tally_update)
        self._emit(ProgressEvent(stage=STAGE_4, done=self._s4.done, total=None, event=EVENT_UPDATE))

    def complete_stage4(self, elapsed: float) -> None:
        """Report curation completion."""
        super().complete_stage4(elapsed)
        self._emit(ProgressEvent(
            stage=STAGE_4, done=self._s4.done, total=None, event=EVENT_COMPLETE,
            extra={"seconds": round(elapsed, 3)},
        ))

    def start_stage4_peak(self, total: int) -> None:
        """Report the start of peak-moment selection."""
        super().start_stage4_peak(total)
        self._emit_sub("peak", self._s4_peak, EVENT_START)

    def update_stage4_peak(self, done: int) -> None:
        """Report peak-moment progress."""
        super().update_stage4_peak(done)
        self._emit_sub("peak", self._s4_peak, EVENT_UPDATE)

    def complete_stage4_peak(self, elapsed: float) -> None:
        """Report peak-moment completion."""
        super().complete_stage4_peak(elapsed)
        self._emit_sub("peak", self._s4_peak, EVENT_COMPLETE)

    def start_stage4_diversity(self, total: int) -> None:
        """Report the start of diversity selection."""
        super().start_stage4_diversity(total)
        self._emit_sub("diversity", self._s4_diversity, EVENT_START)

    def update_stage4_diversity(self, done: int) -> None:
        """Report diversity progress."""
        super().update_stage4_diversity(done)
        self._emit_sub("diversity", self._s4_diversity, EVENT_UPDATE)

    def complete_stage4_diversity(self, elapsed: float) -> None:
        """Report diversity completion."""
        super().complete_stage4_diversity(elapsed)
        self._emit_sub("diversity", self._s4_diversity, EVENT_COMPLETE)

    def start_stage4_tournament(self, total: int) -> None:
        """Report the start of the pairwise tournament."""
        super().start_stage4_tournament(total)
        self._emit_sub("tournament", self._s4_tournament, EVENT_START)

    def update_stage4_tournament(self, done: int) -> None:
        """Report tournament progress."""
        super().update_stage4_tournament(done)
        self._emit_sub("tournament", self._s4_tournament, EVENT_UPDATE)

    def complete_stage4_tournament(self, elapsed: float) -> None:
        """Report tournament completion."""
        super().complete_stage4_tournament(elapsed)
        self._emit_sub("tournament", self._s4_tournament, EVENT_COMPLETE)

    def start_stage4_narrative(self, total: int) -> None:
        """Report the start of the narrative-flow pass."""
        super().start_stage4_narrative(total)
        self._emit_sub("narrative", self._s4_narrative, EVENT_START)

    def update_stage4_narrative(self, done: int) -> None:
        """Report narrative-flow progress."""
        super().update_stage4_narrative(done)
        self._emit_sub("narrative", self._s4_narrative, EVENT_UPDATE)

    def complete_stage4_narrative(self, elapsed: float) -> None:
        """Report narrative-flow completion."""
        super().complete_stage4_narrative(elapsed)
        self._emit_sub("narrative", self._s4_narrative, EVENT_COMPLETE)

    # --- Results ---------------------------------------------------------

    def show_results(self, session_result: Any) -> None:
        """Print nothing: the JSON report carries the results."""
