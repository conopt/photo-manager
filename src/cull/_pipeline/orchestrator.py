"""Pipeline orchestrator — runs Stage 1, Stage 2, Stage 3 in sequence."""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cull.config import (
    CullConfig,
    IMAGE_EXTENSIONS,
    STAGE_IQA,
    STAGE_VLM,
)
from cull.vlm_registry import run_vlm_preflight
from cull.vlm_session import vlm_session
from cull.dashboard import (
    Dashboard,
    DashboardLaunchInfo,
)
from cull.decision_reason import annotate_reasons
from cull.progress_jsonl import JsonlProgressDashboard
from cull.models import (
    CurationResult,
    DecisionLabel,
    PhotoDecision,
    Stage3Result,
)
from cull.router import CURATED_DIR, REVIEW_DIR
from cull.stage1 import duplicate as duplicate_module
from cull import clip_loader
from cull import emotieff_loader
from cull.stage2.aesthetic import unload_predictor
from cull.stage2.composition import unload_topiq_iaa
from cull.stage2.iqa import unload_metrics
from cull.stage2.portrait import unload_face_landmarker

from cull._pipeline.stage1_runner import _run_s1, _Stage1Output
from cull._pipeline.stage2_runner import (
    _run_s2,
    _S2RunInput,
    _Stage2Output,
)
from cull._pipeline.stage2_reducer import (
    _run_s2_reducer,
    _S2ReducerRunInput,
)
from cull._pipeline.stage2_scoring import _SearchCache
from cull._pipeline.event_routing import EventRouteInput, collect_event_scores, route_event
from cull._pipeline.stack_resolution import resolve_event_stacks, resolve_moment_stacks
from cull._pipeline.stage3_runner import (
    _run_s3_if_configured,
    _S3MaybeRunInput,
)
from cull._pipeline.stage4_curator import (
    _run_s4,
    _S4RunInput,
)
from cull._pipeline.decision_assembly import (
    _build_all_decisions,
    _build_summary,
    _DecisionCtx,
)
from cull._pipeline.checkpoint import (
    CheckpointData,
    CheckpointLoadInput,
    CheckpointSaveInput,
    CheckpointTimings,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session result models
# ---------------------------------------------------------------------------


class SessionTiming(BaseModel):
    """Wall-clock timing per pipeline stage."""

    stage1_seconds: float = 0.0
    stage2_seconds: float = 0.0
    reducer_seconds: float = 0.0
    stage3_seconds: float = 0.0
    stage4_seconds: float = 0.0
    total_seconds: float = 0.0  # Stages 1-3, reducer, decisions and Stage 4


class SessionSummary(BaseModel):
    """Decision counts across all photos."""

    keepers: int = 0
    rejected: int = 0
    duplicates: int = 0
    uncertain: int = 0
    selected: int = 0


class SessionResult(BaseModel):
    """Full pipeline run output — decisions, timing, and summary."""

    run_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source_path: str = ""
    model: str = ""
    preset: str = "general"
    is_portrait: bool = False
    total_photos: int = 0
    stages_run: list[int] = Field(default_factory=list)
    summary: SessionSummary = Field(default_factory=SessionSummary)
    timing: SessionTiming = Field(default_factory=SessionTiming)
    decisions: list[PhotoDecision] = Field(default_factory=list)
    curation: CurationResult | None = None


# ---------------------------------------------------------------------------
# Internal parameter bundles (max 2 params rule)
# ---------------------------------------------------------------------------


class _StageTimings(BaseModel):
    """Mutable timing collector for pipeline stages."""

    stage1: float = 0.0
    stage2: float = 0.0
    reducer: float = 0.0
    stage3: float = 0.0
    stage4: float = 0.0


class _PipelineCtx(BaseModel):
    """Top-level context carrying config and paths through the pipeline."""

    config: CullConfig
    paths: list[Path]


class _SessionInput(BaseModel):
    """Input bundle for assembling the final SessionResult."""

    config: CullConfig
    source_path: str
    total_photos: int
    decisions: list[PhotoDecision]
    total_seconds: float
    curation: CurationResult | None = None


# ---------------------------------------------------------------------------
# Stage execution wrappers
# ---------------------------------------------------------------------------


class _StageRunCtx(BaseModel):
    """Shared context for running pipeline stages."""

    model_config = {"arbitrary_types_allowed": True}

    config: CullConfig
    paths: list[Path]
    source_path: Path | None = None
    timings: _StageTimings = Field(default_factory=_StageTimings)
    dashboard: Any = None  # Dashboard instance
    vlm_session: Any = None  # VlmSession — Any avoids import cycle
    vlm_stack: Any = None  # ExitStack owning the lazy VLM load/unload for this run


# `_S2RunInput` uses a forward reference to `_StageRunCtx`, and
# `_S2ReducerRunInput` uses forward references to `_Stage2Output` and
# `_StageRunCtx`. Rebuild both once the referenced classes are defined in
# this module's namespace.
_S2RunInput.model_rebuild(
    _types_namespace={"_StageRunCtx": _StageRunCtx}
)
_S2ReducerRunInput.model_rebuild(
    _types_namespace={
        "_Stage2Output": _Stage2Output,
        "_StageRunCtx": _StageRunCtx,
    }
)


class _StagesResult(BaseModel):
    """Collected outputs from all pipeline stages."""

    model_config = {"arbitrary_types_allowed": True}

    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None
    s3_results: dict[str, Stage3Result] = Field(default_factory=dict)
    search_cache: _SearchCache | None = None


def _unload_stage2_models() -> None:
    """Free Stage 2 neural network models before Stage 3."""
    unload_metrics()
    unload_predictor()
    unload_face_landmarker()
    unload_topiq_iaa()
    clip_loader.unload()
    emotieff_loader.unload()
    gc.collect()


def _unload_imagededup_cnn() -> None:
    """Free Stage 1 duplicate-detection models (CNN + DINOv2) before Stage 2 loads its models."""
    duplicate_module.unload_models()


def _assemble_session(session_in: _SessionInput, ctx: _StageRunCtx) -> SessionResult:
    """Build final SessionResult from pipeline outputs."""
    return SessionResult(
        source_path=session_in.source_path,
        model=session_in.config.model,
        preset=session_in.config.preset,
        is_portrait=session_in.config.is_portrait,
        total_photos=session_in.total_photos,
        stages_run=session_in.config.stages,
        summary=_build_summary(session_in.decisions),
        timing=SessionTiming(
            stage1_seconds=ctx.timings.stage1,
            stage2_seconds=ctx.timings.stage2,
            reducer_seconds=ctx.timings.reducer,
            stage3_seconds=ctx.timings.stage3,
            stage4_seconds=ctx.timings.stage4,
            total_seconds=session_in.total_seconds,
        ),
        decisions=session_in.decisions,
        curation=session_in.curation,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _PipelineRunInput(BaseModel):
    """Input bundle for run_pipeline.

    ``stop_after_stage=2`` runs the scan and Stages 1-2 only, writes a
    checkpoint to ``save_state`` (when set), and returns the Stage 1-2
    decisions without calling the VLM. ``resume_state`` loads such a
    checkpoint and runs Stage 3, Stage 4 and decisions as an uninterrupted
    run would, without running Stages 1-2 again.
    """

    config: CullConfig
    source_path: Path
    file_size_gb: float = 0.0
    progress_jsonl: bool = False
    stop_after_stage: int | None = None
    save_state: Path | None = None
    resume_state: Path | None = None


def run_pipeline(run_in: _PipelineRunInput) -> SessionResult:
    """Orchestrate Stage 1, 2, 3 and return SessionResult."""
    return _run_pipeline_impl(run_in)


class _RunState(BaseModel):
    """Mutable state captured during a pipeline run, used by finalize."""

    model_config = {"arbitrary_types_allowed": True}

    ctx: _StageRunCtx
    stages: _StagesResult
    paths: list[Path]
    total_seconds: float


def _make_dashboard(run_in: _PipelineRunInput) -> Dashboard:
    """Create a Dashboard instance for the given run input."""
    launch_info = DashboardLaunchInfo(
        source_path=str(run_in.source_path),
        photo_count=0,
        preset=run_in.config.preset,
        file_size_gb=run_in.file_size_gb,
    )
    if run_in.progress_jsonl:
        return JsonlProgressDashboard(launch_info)
    return Dashboard(launch_info)


def _build_run_ctx(
    run_in: _PipelineRunInput, dashboard: Dashboard
) -> _StageRunCtx:
    """Scan source, update dashboard photo count, and build a fresh ctx."""
    paths = _scan_with_dashboard(run_in.source_path, dashboard)
    dashboard.set_photo_count(len(paths))
    return _StageRunCtx(
        config=run_in.config, paths=paths,
        source_path=run_in.source_path, dashboard=dashboard,
    )


def _execute_run(ctx: _StageRunCtx) -> _RunState:
    """Run pipeline stages against an existing ctx and return run state."""
    t_start = time.monotonic()
    stages = _execute_stages_inline(ctx)
    return _RunState(
        ctx=ctx, stages=stages, paths=ctx.paths,
        total_seconds=time.monotonic() - t_start,
    )


def _event_route_input(stages: _StagesResult) -> EventRouteInput:
    """Bundle the stage outputs the event router and scorer read."""
    return EventRouteInput(
        s1_out=stages.s1_out, s2_out=stages.s2_out, s3_results=stages.s3_results,
    )


def _event_labels(state: _RunState) -> dict[str, DecisionLabel] | None:
    """Return event-preset labels for representatives, or None for other presets."""
    if not state.ctx.config.is_event:
        return None
    return route_event(_event_route_input(state.stages))


def _finalize_run(state: _RunState, run_in: _PipelineRunInput) -> SessionResult:
    """Build decisions, run Stage 4, and assemble the SessionResult."""
    t_finalize = time.monotonic()
    dec_ctx = _DecisionCtx(
        paths=state.paths, s1_out=state.stages.s1_out,
        s2_out=state.stages.s2_out, s3_results=state.stages.s3_results,
        event_labels=_event_labels(state),
    )
    decisions = _build_all_decisions(dec_ctx)
    curation = _run_s4(_S4RunInput(stages=state.stages, decisions=decisions, ctx=state.ctx))
    annotate_reasons(decisions)
    session_in = _SessionInput(
        config=run_in.config, source_path=str(run_in.source_path),
        total_photos=len(state.paths), decisions=decisions,
        total_seconds=state.total_seconds + time.monotonic() - t_finalize,
        curation=curation,
    )
    return _assemble_session(session_in, state.ctx)


def _needs_vlm(config: CullConfig) -> bool:
    """Return True if Stage 3 or the Stage 4 curator will need a VLM session.

    The event curator never calls the VLM, so only Stage 3 needs it there.
    """
    if config.is_event:
        return STAGE_VLM in config.stages
    return STAGE_VLM in config.stages or config.curate_target is not None


def _resolve_vlm_session_scope(
    config: CullConfig,
) -> AbstractContextManager:
    """Return vlm_session CM if Stage 3/4 will run, else nullcontext.

    Kept as a standalone helper for the frozen cull_fast import contract
    (cull_fast.cli_hook uses its own single-shot scope); `_run_with_session`
    below no longer calls this — it opens the VLM lazily via `vlm_stack`
    instead, so weights load only after Stage 2 unloads.
    """
    if _needs_vlm(config):
        return vlm_session(config.model)
    return contextlib.nullcontext()


def _preflight_vlm(config: CullConfig) -> None:
    """Fail fast on an unresolvable VLM alias before any stage runs.

    Resolves the alias and checks the on-disk directory structure without
    calling mlx_vlm.load(), so a bad --model value is caught before an
    expensive Stage 1-2 run instead of after it.
    """
    if _needs_vlm(config):
        run_vlm_preflight(config.model)


def _load_vlm_if_needed(ctx: _StageRunCtx) -> None:
    """Enter the VLM session CM into ctx.vlm_stack once Stage 2 memory is freed."""
    if ctx.vlm_stack is None or not _needs_vlm(ctx.config):
        return
    ctx.vlm_session = ctx.vlm_stack.enter_context(vlm_session(ctx.config.model))


def _run_with_session(
    run_in: _PipelineRunInput, dashboard: Any
) -> SessionResult:
    """Preflight the VLM alias, then run stages and finalize.

    The VLM itself loads only after Stage 2 models unload (see
    `_execute_stages_inline`) and stays resident through Stage 4 curation;
    it unloads when the ExitStack closes at the end of this function.
    """
    ctx = _build_run_ctx(run_in, dashboard)
    if run_in.stop_after_stage == STAGE_IQA:
        return _run_to_checkpoint(ctx, run_in)
    resumed = _load_resume_state(ctx, run_in) if run_in.resume_state is not None else None
    if resumed is not None:
        run_in = run_in.model_copy(update={"source_path": ctx.source_path})
    _preflight_vlm(run_in.config)
    with contextlib.ExitStack() as stack:
        ctx.vlm_stack = stack
        state = _execute_run(ctx) if resumed is None else _execute_resumed(ctx, resumed)
        return _finalize_run(state, run_in)


# ---------------------------------------------------------------------------
# Stage checkpoint: stop after Stage 2, or resume from it
# ---------------------------------------------------------------------------


def _checkpoint_timings(ctx: _StageRunCtx, total: float) -> CheckpointTimings:
    """Return the Stage 1-2 timings recorded on ctx."""
    return CheckpointTimings(
        stage1=ctx.timings.stage1, stage2=ctx.timings.stage2,
        reducer=ctx.timings.reducer, total=total,
    )


def _run_to_checkpoint(ctx: _StageRunCtx, run_in: _PipelineRunInput) -> SessionResult:
    """Run Stages 1-2, save the checkpoint, and return the Stage 1-2 decisions.

    No VLM is preflighted, loaded or called. The returned session is what a
    run without Stage 3 and Stage 4 would decide; it is assembled only after
    the checkpoint is written, so it cannot change what a resume sees.
    """
    t_start = time.monotonic()
    local = _run_local_stages(ctx)
    total = time.monotonic() - t_start
    if run_in.save_state is not None:
        save_checkpoint(CheckpointSaveInput(
            path=run_in.save_state, source_path=run_in.source_path, paths=ctx.paths,
            config=ctx.config, timings=_checkpoint_timings(ctx, total),
            s1_out=local.s1_out, s2_out=local.s2_out,
        ))
    config = ctx.config.model_copy(update={
        "stages": [s for s in ctx.config.stages if s != STAGE_VLM], "curate_target": None,
    })
    ctx.config = config
    state = _RunState(ctx=ctx, stages=_run_vlm_stages(ctx, local), paths=ctx.paths, total_seconds=total)
    return _finalize_run(state, run_in.model_copy(update={"config": config}))


def _load_resume_state(ctx: _StageRunCtx, run_in: _PipelineRunInput) -> CheckpointData:
    """Load and validate the checkpoint, then use its source path spelling.

    Stage 1-2 outputs are keyed by the photo paths of the checkpointed run;
    the scanned photos were just checked to be the same files.
    """
    data = load_checkpoint(CheckpointLoadInput(
        path=run_in.resume_state, source_path=run_in.source_path,
        paths=ctx.paths, config=run_in.config,
    ))
    source = Path(data.state.source_path)
    ctx.source_path = source
    ctx.paths = [source / photo.path for photo in data.state.photos]
    return data


def _execute_resumed(ctx: _StageRunCtx, resumed: CheckpointData) -> _RunState:
    """Run Stage 3 onwards on checkpointed Stage 1-2 outputs."""
    t_start = time.monotonic()
    timings = resumed.state.timings
    ctx.timings.stage1 = timings.stage1
    ctx.timings.stage2 = timings.stage2
    ctx.timings.reducer = timings.reducer
    local = _LocalStages(s1_out=resumed.s1_out, s2_out=resumed.s2_out)
    stages = _run_vlm_stages(ctx, local)
    return _RunState(
        ctx=ctx, stages=stages, paths=ctx.paths,
        total_seconds=timings.total + time.monotonic() - t_start,
    )


def _run_pipeline_impl(run_in: _PipelineRunInput) -> SessionResult:
    """Implementation of the pipeline execution."""
    with _make_dashboard(run_in) as dashboard:
        result = _run_with_session(run_in, dashboard)
        dashboard.show_results(result)
        return result


class _LocalStages(BaseModel):
    """Stage 1-2 outputs: everything a run has before the VLM is involved."""

    model_config = {"arbitrary_types_allowed": True}

    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None


def _execute_stages_inline(ctx: _StageRunCtx) -> _StagesResult:
    """Run stages without opening a new dashboard context.

    The VLM loads (via `_load_vlm_if_needed`) after Stage 2 models unload,
    so Stage 1-2's torch/CLIP stack and the VLM are never resident together.
    """
    return _run_vlm_stages(ctx, _run_local_stages(ctx))


def _run_local_stages(ctx: _StageRunCtx) -> _LocalStages:
    """Run Stage 1, Stage 2 with its reducer, and non-event moment stacks; free their models."""
    s2_out: _Stage2Output | None = None
    s1_out = _run_s1(ctx)
    if STAGE_IQA in ctx.config.stages:
        s2_out = _run_s2(_S2RunInput(s1_out=s1_out, ctx=ctx))
        _run_s2_reducer(_S2ReducerRunInput(s2_out=s2_out, s1_out=s1_out, ctx=ctx))
        _unload_stage2_models()
    if not ctx.config.is_event:
        resolve_moment_stacks(s1_out, s2_out)
    return _LocalStages(s1_out=s1_out, s2_out=s2_out)


def _run_vlm_stages(ctx: _StageRunCtx, local: _LocalStages) -> _StagesResult:
    """Load the VLM if needed, run Stage 3, and resolve event stacks."""
    s3_results: dict[str, Stage3Result] = {}
    _load_vlm_if_needed(ctx)
    if STAGE_IQA in ctx.config.stages:
        s3_results = _run_s3_if_configured(
            _S3MaybeRunInput(ctx=ctx, s2_out=local.s2_out, s1_out=local.s1_out),
        )
    s2_out = local.s2_out
    search_cache = s2_out.search_cache if s2_out is not None else None
    stages = _StagesResult(
        s1_out=local.s1_out, s2_out=s2_out,
        s3_results=s3_results, search_cache=search_cache,
    )
    if ctx.config.is_event:
        resolve_event_stacks(local.s1_out, collect_event_scores(_event_route_input(stages)))
    return stages


# Prior pipeline output dirs must never be re-ingested on a re-run — canonical
# names live in cull.router (single source of truth).
_EXCLUDED_SCAN_DIRS: frozenset[str] = frozenset({REVIEW_DIR, CURATED_DIR})


def _is_pipeline_output_path(path: Path, source_path: Path) -> bool:
    """Return True if path lives under a prior pipeline output dir (_review/_curated)."""
    relative_parts = path.relative_to(source_path).parts
    return any(part in _EXCLUDED_SCAN_DIRS for part in relative_parts)


def _scan_with_dashboard(source_path: Path, dashboard: Dashboard) -> list[Path]:
    """Scan for supported images while updating the dashboard's scan state."""
    paths: list[Path] = []
    total_bytes = 0
    dashboard.begin_scan()
    for p in source_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if _is_pipeline_output_path(p, source_path):
            continue
        paths.append(p)
        total_bytes += p.stat().st_size
        dashboard.update_scan_progress(len(paths), total_bytes)
    dashboard.end_scan()
    return sorted(paths)
