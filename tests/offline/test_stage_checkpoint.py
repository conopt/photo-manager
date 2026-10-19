"""Endpoint runs, dry-run isolation, JSON reports, JSONL progress and stage checkpoints.

Generated photos run through the real pipeline; only the model-backed
Stage 1-2 calls are faked (tests/offline/_pipeline_fakes.py) and the VLM is
a local HTTP endpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner, Result

from cull import cli
from cull._pipeline.checkpoint import (
    CheckpointError,
    CheckpointLoadInput,
    CheckpointSaveInput,
    CheckpointTimings,
    load_checkpoint,
    save_checkpoint,
)
from cull._pipeline.stage1_runner import _Stage1Output
from cull._pipeline.stage2_runner import _Stage2Output
from cull._pipeline.stage2_scoring import _SearchCache
from cull.config import EMBEDDING_CACHE_FILENAME, CullConfig
from cull.model_cache import BootstrapStatus, PreflightResult
from cull.pipeline import SessionResult, _PipelineRunInput, run_pipeline
from cull.vlm_registry import vlm_display_name
from tests.offline._pipeline_fakes import (
    ENDPOINT_MODEL,
    install_fake_local_stages,
    vlm_endpoint,
    write_staging_corpus,
)

PROGRESS_STAGES: set[str] = {"scan", "stage1", "stage2", "stage3", "stage4"}
ANSI_ESCAPE: str = "\x1b"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A staging SOURCE of symlinked originals and previews."""
    return write_staging_corpus(tmp_path)


@pytest.fixture
def cull_cli(monkeypatch: pytest.MonkeyPatch):
    """Invoke the cull command in-process with the offline model cache treated as ready."""
    ready = PreflightResult(status=BootstrapStatus(state="ok", cache_root=Path("/unused")), message="ok")
    monkeypatch.setattr(cli, "require_bootstrap_valid", lambda cache: ready)

    def invoke(*args: object) -> Result:
        return CliRunner().invoke(cli._cull_pipeline_command, [str(a) for a in args])

    return invoke


def _snapshot(root: Path) -> dict[str, tuple]:
    """Return every entry under root: type, symlink target or file bytes, and mtime."""
    entries: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            stat = path.lstat()
            if path.is_symlink():
                entries[str(path.relative_to(root))] = ("link", os.readlink(path), stat.st_mtime_ns)
            elif path.is_dir():
                entries[str(path.relative_to(root))] = ("dir",)
            else:
                entries[str(path.relative_to(root))] = ("file", path.read_bytes(), stat.st_mtime_ns)
    return entries


def _report(path: Path) -> SessionResult:
    """Load a --json-report file."""
    return SessionResult.model_validate_json(path.read_text(encoding="utf-8"))


def _outcome(result: SessionResult) -> dict:
    """Return everything a run decided, without run timestamps and timings."""
    curation = result.curation.model_dump(exclude={"elapsed_seconds"}) if result.curation else None
    return {
        "decisions": [d.model_dump(mode="json") for d in result.decisions],
        "summary": result.summary.model_dump(),
        "curation": curation,
        "stages_run": result.stages_run,
        "source_path": result.source_path,
    }


def _progress_lines(stderr: str) -> list[dict]:
    """Parse every stderr line as one JSON progress object."""
    return [json.loads(line) for line in stderr.splitlines() if line.strip()]


def _forbid_local_stages(monkeypatch: pytest.MonkeyPatch, calls) -> None:
    """Make any Stage 1/2 execution fail the test."""
    calls.is_forbidden = True

    def forbidden(*args: object) -> None:
        raise AssertionError("Stage 1/2 ran during a resume")

    monkeypatch.setattr("cull._pipeline.orchestrator._run_s1", forbidden)
    monkeypatch.setattr("cull._pipeline.orchestrator._run_s2", forbidden)


# ---------------------------------------------------------------------------
# Endpoint display name
# ---------------------------------------------------------------------------


def test_endpoint_display_name_skips_local_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """With an endpoint configured, the display name is the model ID and no registry is read."""
    with vlm_endpoint(monkeypatch):
        assert vlm_display_name(ENDPOINT_MODEL) == ENDPOINT_MODEL


def test_endpoint_full_pipeline_uses_endpoint_model_name(
    source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full endpoint run reaches Stage 3 instead of failing on a local display-name lookup."""
    install_fake_local_stages(monkeypatch)
    config = CullConfig(model=ENDPOINT_MODEL, is_portrait=False, is_dry_run=True)
    with vlm_endpoint(monkeypatch) as log:
        result = run_pipeline(_PipelineRunInput(config=config, source_path=source, progress_jsonl=True))
    rated = [d.stage3 for d in result.decisions if d.stage3 is not None]
    assert rated, "the fixture must send at least one photo to Stage 3"
    assert {s.model_used for s in rated} == {ENDPOINT_MODEL}
    assert log.models == [ENDPOINT_MODEL] * len(rated)


# ---------------------------------------------------------------------------
# Dry run, --json-report, --progress-jsonl
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_into_source_and_writes_json_report(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cull_cli,
) -> None:
    """--dry-run leaves SOURCE byte-for-byte unchanged; --json-report still has every decision."""
    install_fake_local_stages(monkeypatch)
    before = _snapshot(source)
    report_path = tmp_path / "report.json"
    with vlm_endpoint(monkeypatch):
        run = cull_cli(source, "--model", ENDPOINT_MODEL, "--no-portrait", "--dry-run",
                       "--json-report", report_path)
    assert run.exit_code == 0, run.output
    assert _snapshot(source) == before
    report = _report(report_path)
    assert report.total_photos == len(report.decisions) == 9
    assert all(d.reason for d in report.decisions)
    assert {d.reason.split(" · ")[0] for d in report.decisions} >= {"REJECT", "DUPLICATE"}


def test_routed_run_persists_search_cache_and_reports(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cull_cli,
) -> None:
    """Without --dry-run the same run writes the embeddings cache and session report."""
    install_fake_local_stages(monkeypatch)
    report_path = tmp_path / "report.json"
    with vlm_endpoint(monkeypatch):
        run = cull_cli(source, "--model", ENDPOINT_MODEL, "--no-portrait", "--no-sidecars",
                       "--json-report", report_path)
    assert run.exit_code == 0, run.output
    assert (source / EMBEDDING_CACHE_FILENAME).is_file()
    assert (source / "session_report.json").is_file()
    moved = [d for d in _report(report_path).decisions if d.destination is not None]
    assert moved and all(d.destination.exists() for d in moved)


def test_progress_jsonl_writes_json_lines_to_stderr_only(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cull_cli,
) -> None:
    """--progress-jsonl emits parseable stage/done/total lines and keeps stdout clean."""
    install_fake_local_stages(monkeypatch)
    with vlm_endpoint(monkeypatch):
        run = cull_cli(source, "--model", ENDPOINT_MODEL, "--no-portrait", "--dry-run", "--curate", "50",
                       "--progress-jsonl", "--json-report", tmp_path / "report.json")
    assert run.exit_code == 0, run.output
    assert ANSI_ESCAPE not in run.stdout
    assert run.stdout.strip() == ""
    lines = _progress_lines(run.stderr)
    assert {"scan", "stage1", "stage2", "stage3"} <= {line["stage"] for line in lines}
    for line in lines:
        assert line["stage"] in PROGRESS_STAGES
        assert isinstance(line["done"], int)
        assert line["total"] is None or (isinstance(line["total"], int) and line["done"] <= line["total"])
    stage1 = [line for line in lines if line["stage"] == "stage1" and "substage" not in line]
    assert stage1[0] == {"stage": "stage1", "done": 0, "total": 9, "event": "start"}
    assert [line["done"] for line in stage1 if line["event"] == "update"] == list(range(1, 10))


# ---------------------------------------------------------------------------
# Stage checkpoint: phase 1 / phase 2 against an uninterrupted run
# ---------------------------------------------------------------------------


def test_checkpoint_resume_matches_uninterrupted_run(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cull_cli,
) -> None:
    """Stop after Stage 2, resume, and decide exactly what one uninterrupted run decides."""
    calls = install_fake_local_stages(monkeypatch)
    common = (source, "--model", ENDPOINT_MODEL, "--no-portrait", "--dry-run", "--curate", "50")
    state = tmp_path / "state.npz"
    with vlm_endpoint(monkeypatch) as log:
        full = cull_cli(*common, "--json-report", tmp_path / "full.json")
        assert full.exit_code == 0, full.output
        full_requests = len(log.models)
        phase1 = cull_cli(*common, "--stop-after-stage", 2, "--save-state", state,
                          "--progress-jsonl", "--json-report", tmp_path / "phase1.json")
        assert phase1.exit_code == 0, phase1.output
        assert len(log.models) == full_requests, "phase 1 must not call the VLM"
        _forbid_local_stages(monkeypatch, calls)
        phase2 = cull_cli(*common, "--resume-state", state,
                          "--progress-jsonl", "--json-report", tmp_path / "phase2.json")
        assert phase2.exit_code == 0, phase2.output
    assert len(log.models) == 2 * full_requests > 0
    assert _outcome(_report(tmp_path / "phase2.json")) == _outcome(_report(tmp_path / "full.json"))
    provisional = _report(tmp_path / "phase1.json")
    assert provisional.stages_run == [1, 2] and provisional.curation is None
    assert all(d.stage3 is None and d.reason for d in provisional.decisions)
    assert phase1.stdout.strip() == phase2.stdout.strip() == ""
    assert "stage3" not in {line["stage"] for line in _progress_lines(phase1.stderr)}
    assert {"stage1", "stage2"}.isdisjoint(line["stage"] for line in _progress_lines(phase2.stderr))


def test_event_checkpoint_resume_matches_uninterrupted_curation(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event curation (MMR over checkpointed CLIP rows) picks the same photos after a resume."""
    calls = install_fake_local_stages(monkeypatch)
    config = CullConfig(model=ENDPOINT_MODEL, preset="event", is_portrait=False, is_dry_run=True, curate_target=3)
    state = tmp_path / "event.npz"

    def run(**options: object) -> SessionResult:
        return run_pipeline(_PipelineRunInput(config=config, source_path=source, progress_jsonl=True, **options))

    with vlm_endpoint(monkeypatch):
        full = run()
        run(stop_after_stage=2, save_state=state)
        _forbid_local_stages(monkeypatch, calls)
        resumed = run(resume_state=state)
    assert full.curation is not None and full.curation.actual_count == 3
    assert _outcome(resumed) == _outcome(full)


# ---------------------------------------------------------------------------
# Checkpoint validation and format
# ---------------------------------------------------------------------------


def _save_phase1(source: Path, state: Path, cull_cli) -> None:
    """Run phase 1 into state."""
    run = cull_cli(source, "--no-portrait", "--stop-after-stage", 2, "--save-state", state, "--progress-jsonl")
    assert run.exit_code == 0, run.output


def test_resume_rejects_changed_photos_and_options(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cull_cli,
) -> None:
    """A resume fails clearly when photos or Stage 1-2 options changed since phase 1."""
    install_fake_local_stages(monkeypatch)
    state = tmp_path / "state.npz"
    _save_phase1(source, state, cull_cli)
    preset = cull_cli(source, "--no-portrait", "--no-vlm", "--preset", "wedding",
                      "--resume-state", state, "--progress-jsonl")
    assert preset.exit_code == 1
    error = _progress_lines(preset.stderr)[-1]
    assert error["event"] == "error" and "preset" in error["message"]
    (source / "day2" / "b01.jpg").unlink()
    changed = cull_cli(source, "--no-portrait", "--no-vlm", "--resume-state", state, "--progress-jsonl")
    assert changed.exit_code == 1
    error = _progress_lines(changed.stderr)[-1]
    assert "changed since the checkpoint" in error["message"] and "b01.jpg" in error["message"]


def test_load_rejects_foreign_and_future_files(source: Path, tmp_path: Path) -> None:
    """Non-checkpoint files and other format versions are refused with a clear message."""
    config = CullConfig(is_portrait=False)
    load_in = CheckpointLoadInput(path=tmp_path / "x.npz", source_path=source, paths=[], config=config)
    (tmp_path / "x.npz").write_bytes(b"not an archive")
    with pytest.raises(CheckpointError, match="not a cull stage checkpoint"):
        load_checkpoint(load_in)
    save_checkpoint(CheckpointSaveInput(
        path=tmp_path / "x.npz", source_path=source, paths=[], config=config,
        timings=CheckpointTimings(), s1_out=_Stage1Output(),
    ))
    with np.load(tmp_path / "x.npz") as archive:
        members = {name: archive[name] for name in archive.files}
    state = json.loads(members["state"].tobytes())
    state["version"] = 99
    members["state"] = np.frombuffer(json.dumps(state).encode(), dtype=np.uint8)
    with (tmp_path / "x.npz").open("wb") as handle:
        np.savez(handle, **members)
    with pytest.raises(CheckpointError, match="version 99"):
        load_checkpoint(load_in)


def test_checkpoint_round_trips_scores_and_arrays_exactly(source: Path, tmp_path: Path) -> None:
    """Scores (including NaN/inf), duplicate sets, CNN encodings and CLIP rows survive unchanged."""
    from tests.offline._pipeline_fakes import _fusion, _stage1_result  # noqa: PLC0415

    paths = sorted(p for p in source.rglob("*.jpg"))
    s1 = _Stage1Output(
        results={str(p): _stage1_result(p, i) for i, p in enumerate(paths)},
        survivors=paths, duplicate_paths={str(paths[1])}, stacks=[[str(paths[0]), str(paths[1])]],
        encodings={str(p): np.arange(4, dtype=np.float32) + i for i, p in enumerate(paths)},
    )
    s1.results[str(paths[0])].noise_score = float("nan")
    config = CullConfig(is_portrait=False)
    fusions = {str(p): _fusion(p, config) for p in paths}
    fusions[str(paths[2])].stage2.composite = float("-inf")
    rows = np.random.default_rng(0).standard_normal((len(paths), 5)).astype(np.float32)
    s2 = _Stage2Output(results=fusions, keepers=paths[:2], search_cache=_SearchCache(
        embeddings=rows, path_to_row={str(p): i for i, p in enumerate(paths)},
    ))
    state = tmp_path / "rt.npz"
    save_checkpoint(CheckpointSaveInput(
        path=state, source_path=source, paths=paths, config=config,
        timings=CheckpointTimings(stage1=1.5, total=2.0), s1_out=s1, s2_out=s2,
    ))
    loaded = load_checkpoint(CheckpointLoadInput(path=state, source_path=source, paths=paths, config=config))
    assert np.isnan(loaded.s1_out.results[str(paths[0])].noise_score)
    loaded.s1_out.results[str(paths[0])].noise_score = s1.results[str(paths[0])].noise_score = 0.0
    assert loaded.s1_out.model_dump(exclude={"encodings"}) == s1.model_dump(exclude={"encodings"})
    assert all(np.array_equal(loaded.s1_out.encodings[k], v) for k, v in s1.encodings.items())
    assert loaded.s2_out.model_dump(exclude={"search_cache"}) == s2.model_dump(exclude={"search_cache"})
    assert np.array_equal(loaded.s2_out.search_cache.embeddings, rows)
    assert loaded.s2_out.search_cache.path_to_row == s2.search_cache.path_to_row
    assert loaded.state.timings.stage1 == 1.5


@pytest.mark.parametrize("args,message", [
    (["--stop-after-stage", "3", "--save-state", "s.npz"], "supports only 2"),
    (["--stop-after-stage", "2"], "must be given together"),
    (["--save-state", "s.npz"], "must be given together"),
    (["--stop-after-stage", "2", "--save-state", "s.npz", "--stage", "1"], "need Stages 1 and 2"),
    (["--stop-after-stage", "2", "--save-state", "s.npz", "--review-after"], "cannot be combined"),
    (["--json-report", "missing-dir/r.json"], "does not exist"),
])
def test_invalid_checkpoint_and_report_flags_fail_before_running(
    source: Path, monkeypatch: pytest.MonkeyPatch, cull_cli, args: list[str], message: str,
) -> None:
    """Inconsistent flags exit 1 with a JSON error line before any stage runs."""
    calls = install_fake_local_stages(monkeypatch)
    monkeypatch.chdir(source.parent)
    run = cull_cli(source, "--progress-jsonl", *args)
    assert run.exit_code == 1
    error = _progress_lines(run.stderr)[-1]
    assert error["event"] == "error" and message in error["message"]
    assert calls.stage1_loops == 0
