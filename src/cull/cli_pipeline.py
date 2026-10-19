"""Pipeline runners and post-processing helpers — extracted from cli.py in the 600-series split."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from pydantic import BaseModel, ConfigDict

from cull.cli_config import _build_config, _build_stages
from cull.cli_help import FAST_MODE_AVAILABLE
from cull.cli_results import (
    _execute_moves,
    _launch_review_after,
    _move_files,
    _show_dry_run,
    _show_results,
    _write_report,
)
from cull.config import CullConfig, STAGE_IQA, VLM_MODELS_ROOT
from cull.dashboard import show_general_error, show_vlm_load_error
from cull.pipeline import SessionResult, _PipelineRunInput, run_pipeline
from cull.progress_jsonl import emit_progress_error
from cull.report import write_json_report, write_report
from cull._pipeline.checkpoint import CheckpointError
from cull.vlm_session import VlmLoadError

BYTES_PER_GB: float = 1024 * 1024 * 1024
VALIDATION_EXIT_CODE: int = 1
SUBCOMMAND_FLAGS: tuple[str, ...] = (
    "overrides", "search", "similar", "explain", "report_card", "calibrate",
    "bake_manifest",
)
REVIEW_FLAGS: tuple[str, ...] = ("review", "review_all", "review_after")
CURATE_REQUIRED_STAGES: tuple[int, ...] = (1, 2)
CHECKPOINT_STOP_STAGE: int = STAGE_IQA
CHECKPOINT_FLAGS: tuple[str, ...] = ("stop_after_stage", "save_state", "resume_state")


class PostPipelineInput(BaseModel):
    """Inputs for post-pipeline handling (results, moves, report, TUI)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: SessionResult
    config: CullConfig
    report_flag: bool = False
    review_after: bool = False
    json_report: Path | None = None
    progress_jsonl: bool = False
    saved_state: Path | None = None  # Set when the run stopped after Stage 2.


def _require_source(kwargs: dict) -> None:
    """Exit with error if no source argument was supplied."""
    if not kwargs.get("source"):
        show_general_error(
            "Missing source",
            "cull requires a SOURCE directory argument (or a subcommand flag).",
        )
        sys.exit(VALIDATION_EXIT_CODE)


def _reject_subcommand_conflict(active_review: list[str]) -> None:
    """Emit review-vs-subcommand conflict error and exit."""
    show_general_error(
        "Conflicting flags",
        f"--{active_review[0].replace('_', '-')} cannot be combined with subcommand flags.",
    )
    sys.exit(VALIDATION_EXIT_CODE)


def _validate_subcommand_flags(kwargs: dict) -> None:
    """Ensure at most one subcommand flag is set; exit with error if more."""
    active = [f for f in SUBCOMMAND_FLAGS if kwargs.get(f)]
    if len(active) > 1:
        show_general_error(
            "Conflicting flags",
            f"Only one subcommand allowed at a time. Got: {active}",
        )
        sys.exit(VALIDATION_EXIT_CODE)
    active_review = [f for f in REVIEW_FLAGS if kwargs.get(f)]
    if active_review and active:
        _reject_subcommand_conflict(active_review)


def _validate_fast_conflicts(kwargs: dict) -> None:
    """Refuse --fast combined with any subcommand or review flag (spec §9 Q2)."""
    if not kwargs.get("fast"):
        return
    conflicts = [f for f in SUBCOMMAND_FLAGS if kwargs.get(f)]
    conflicts.extend(f for f in REVIEW_FLAGS if kwargs.get(f))
    if conflicts:
        show_general_error(
            "Conflicting flags",
            f"--fast cannot be combined with: {conflicts}",
        )
        sys.exit(VALIDATION_EXIT_CODE)


def _validate_curate_stages(config: CullConfig) -> None:
    """Stage 4 requires Stages 1 and 2 to run — fail fast with a clear error."""
    if config.curate_target is None:
        return
    missing = [s for s in CURATE_REQUIRED_STAGES if s not in config.stages]
    if not missing:
        return
    show_general_error(
        "Invalid stage combination",
        f"--curate requires Stages 1 and 2 to run. Missing: {missing}. "
        f"Remove --stage, or include at least stages 1 and 2.",
    )
    sys.exit(VALIDATION_EXIT_CODE)


def _fail(kwargs: dict, message: str) -> None:
    """Report a validation error (as a JSON line in --progress-jsonl mode) and exit."""
    if kwargs.get("progress_jsonl"):
        emit_progress_error(message)
    else:
        show_general_error("Invalid options", message)
    sys.exit(VALIDATION_EXIT_CODE)


def _output_path_problem(flag: str, path: Path | None) -> str | None:
    """Return why an output file path is unusable, or None when it is fine."""
    if path is None:
        return None
    if path.is_dir():
        return f"{flag} {path} is a directory; give a file path"
    if not path.parent.is_dir():
        return f"{flag} {path}: directory {path.parent} does not exist"
    return None


def _checkpoint_problem(kwargs: dict) -> str | None:
    """Return why the stage-checkpoint flags are inconsistent, or None."""
    stop, save, resume = (kwargs.get(f) for f in CHECKPOINT_FLAGS)
    if stop is not None and stop != CHECKPOINT_STOP_STAGE:
        return f"--stop-after-stage supports only {CHECKPOINT_STOP_STAGE}"
    if (stop is None) != (save is None):
        return f"--stop-after-stage {CHECKPOINT_STOP_STAGE} and --save-state must be given together"
    if resume is not None and stop is not None:
        return "--resume-state cannot be combined with --stop-after-stage/--save-state"
    others = [f for f in (*SUBCOMMAND_FLAGS, "review", "review_all", "fast") if kwargs.get(f)]
    if stop is not None and kwargs.get("review_after"):
        others.append("review_after")
    if others:
        return f"stage checkpoint flags cannot be combined with: {others}"
    stages = _build_stages(kwargs["stage"], kwargs["no_vlm"])
    if any(s not in stages for s in CURATE_REQUIRED_STAGES):
        return "stage checkpoints need Stages 1 and 2; remove --stage or include 1 and 2"
    return None


def _validate_run_flags(kwargs: dict) -> None:
    """Validate --json-report and the stage-checkpoint flags before any work starts."""
    is_checkpoint = any(kwargs.get(f) is not None for f in CHECKPOINT_FLAGS)
    problems = [
        _checkpoint_problem(kwargs) if is_checkpoint else None,
        _output_path_problem("--json-report", kwargs.get("json_report")),
        _output_path_problem("--save-state", kwargs.get("save_state")),
    ]
    for problem in problems:
        if problem is not None:
            _fail(kwargs, problem)


def _build_run_input(kwargs: dict, config: CullConfig) -> _PipelineRunInput:
    """Build the pipeline input; checkpoint runs record an absolute source path."""
    source = Path(kwargs["source"])
    is_checkpoint = any(kwargs.get(f) is not None for f in CHECKPOINT_FLAGS)
    return _PipelineRunInput(
        config=config,
        source_path=source.absolute() if is_checkpoint else source,
        file_size_gb=_compute_file_size_gb(source),
        progress_jsonl=bool(kwargs.get("progress_jsonl")),
        stop_after_stage=kwargs.get("stop_after_stage"),
        save_state=kwargs.get("save_state"),
        resume_state=kwargs.get("resume_state"),
    )


def _compute_file_size_gb(source: Path) -> float:
    """Sum the size of all files under source in gigabytes."""
    total = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    return total / BYTES_PER_GB


def _run_with_dashboard(run_in: _PipelineRunInput) -> SessionResult:
    """Run pipeline with error handling and dashboard panels.

    In --progress-jsonl mode the dashboard has already reported the error as
    a JSON line, so no panel is printed.
    """
    is_panel = not run_in.progress_jsonl
    try:
        return run_pipeline(run_in)
    except VlmLoadError as exc:
        if is_panel:
            show_vlm_load_error(getattr(exc, "alias", str(exc)), VLM_MODELS_ROOT)
        sys.exit(1)
    except CheckpointError as exc:
        if is_panel:
            show_general_error("Stage checkpoint", str(exc))
        sys.exit(1)
    except Exception as exc:
        if is_panel:
            show_general_error("Pipeline Error", str(exc))
        raise


def _run_fast_dispatch(kwargs: dict) -> None:
    """Lazy-import and dispatch to the fast-mode pipeline (Pattern B)."""
    if not FAST_MODE_AVAILABLE:
        show_general_error(
            "Fast mode unavailable",
            "The cull_fast package is not installed. Install it to enable --fast.",
        )
        sys.exit(1)
    from cull_fast.cli_hook import run_fast_pipeline  # noqa: PLC0415

    _require_source(kwargs)
    config = _build_config(kwargs)
    source = Path(kwargs["source"])
    run_in = _PipelineRunInput(
        config=config, source_path=source, file_size_gb=_compute_file_size_gb(source),
    )
    result = run_fast_pipeline(run_in)
    post_in = PostPipelineInput(
        result=result,
        config=config,
        report_flag=bool(kwargs["report"]),
        review_after=bool(kwargs.get("review_after", False)),
    )
    _post_pipeline(post_in)


def _write_json_report(post_in: PostPipelineInput) -> None:
    """Write the full SessionResult to --json-report, when requested."""
    if post_in.json_report is not None:
        write_json_report(post_in.result, post_in.json_report)


def _finish_stopped_run(post_in: PostPipelineInput) -> None:
    """After a run that stopped at its checkpoint: no routing, moves or session report."""
    _write_json_report(post_in)
    if not post_in.progress_jsonl:
        click.echo(f"Stage {CHECKPOINT_STOP_STAGE} checkpoint written to {post_in.saved_state}")


def _route_quietly(post_in: PostPipelineInput) -> None:
    """--progress-jsonl: move files and write the session report without console output."""
    _execute_moves(post_in.result, post_in.config)
    if post_in.report_flag:
        write_report(post_in.result)


def _post_pipeline(post_in: PostPipelineInput) -> None:
    """Handle results display, file moves, report, and TUI handoff.

    A dry run writes nothing into the source tree; --json-report is written
    in every mode, after moves so destinations are recorded.
    """
    result = post_in.result
    config = post_in.config
    if post_in.saved_state is not None:
        _finish_stopped_run(post_in)
        return
    if config.is_dry_run:
        if not post_in.progress_jsonl:
            _show_dry_run(result)
        _write_json_report(post_in)
        return
    if post_in.progress_jsonl:
        _route_quietly(post_in)
    else:
        _show_results(result, config)
        _move_files(result, config)
        if post_in.report_flag:
            _write_report(result)
    _write_json_report(post_in)
    if post_in.review_after:
        _launch_review_after(result, config)
