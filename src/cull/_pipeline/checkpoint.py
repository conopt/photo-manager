"""Stage checkpoint: persist Stages 1-2 so Stage 3, Stage 4 and routing run later.

A checkpoint is one uncompressed NumPy ``.npz`` archive, loaded with
``allow_pickle=False``, holding three members:

- ``state``: UTF-8 JSON (``CheckpointState``) with the format name and
  version, the source directory, a fingerprint of every scanned photo, the
  config fields that shaped Stages 1-2, their timings, and the Stage 1 and
  Stage 2 outputs (moment stacks, duplicate groups, routing lists, scores).
- ``encodings``: the duplicate detector's per-photo CNN embeddings, one row
  per ``state.encoding_keys`` entry (Stage 4 clustering reads them).
- ``search_embeddings``: the L2-normalised CLIP rows, one per
  ``state.search_paths`` entry (Stage 4 diversity reads them).

JSON floats keep NaN and infinities as ``NaN``/``Infinity`` tokens so every
score round-trips exactly.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic_core
from pydantic import BaseModel, Field, ValidationError

from cull._pipeline.stage1_runner import _Stage1Output
from cull._pipeline.stage2_runner import _Stage2Output
from cull._pipeline.stage2_scoring import _SearchCache
from cull.config import CullConfig

CHECKPOINT_FORMAT: str = "cull-stage-checkpoint"
CHECKPOINT_VERSION: int = 1
CHECKPOINT_STAGE: int = 2
_STATE_MEMBER: str = "state"
_ENCODINGS_MEMBER: str = "encodings"
_SEARCH_MEMBER: str = "search_embeddings"
_MISMATCH_EXAMPLES: int = 3
# CullConfig fields that change Stage 1-2 output; a resume must use the same values.
STAGE12_CONFIG_FIELDS: tuple[str, ...] = ("preset", "burst_gap", "is_portrait")


class CheckpointError(RuntimeError):
    """A checkpoint cannot be written, read, or does not match this run."""


class PhotoFingerprint(BaseModel):
    """Identity of one scanned photo, relative to the source directory."""

    path: str
    size: int
    mtime_ns: int


class CheckpointTimings(BaseModel):
    """Wall-clock seconds spent in the checkpointed stages."""

    stage1: float = 0.0
    stage2: float = 0.0
    reducer: float = 0.0
    total: float = 0.0


class CheckpointState(BaseModel):
    """JSON part of a checkpoint."""

    format: Literal["cull-stage-checkpoint"] = CHECKPOINT_FORMAT
    version: int = CHECKPOINT_VERSION
    stage: int = CHECKPOINT_STAGE
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source_path: str
    photos: list[PhotoFingerprint]
    config: dict[str, object]
    timings: CheckpointTimings
    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None
    encoding_keys: list[str] = Field(default_factory=list)
    search_paths: list[str] | None = None


class CheckpointData(BaseModel):
    """A loaded checkpoint: Stage 1-2 outputs with their arrays restored."""

    model_config = {"arbitrary_types_allowed": True}

    state: CheckpointState
    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None


class CheckpointSaveInput(BaseModel):
    """Everything a checkpoint records about a finished Stage 1-2 run."""

    model_config = {"arbitrary_types_allowed": True}

    path: Path
    source_path: Path
    paths: list[Path]
    config: CullConfig
    timings: CheckpointTimings
    s1_out: _Stage1Output
    s2_out: _Stage2Output | None = None


class CheckpointLoadInput(BaseModel):
    """What a resumed run must match: its source, scanned photos and config."""

    path: Path
    source_path: Path
    paths: list[Path]
    config: CullConfig


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def fingerprint_photos(source_path: Path, paths: list[Path]) -> list[PhotoFingerprint]:
    """Return path (relative to source), size and mtime for every photo."""
    fingerprints: list[PhotoFingerprint] = []
    for path in paths:
        stat = path.stat()
        fingerprints.append(PhotoFingerprint(
            path=path.relative_to(source_path).as_posix(), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        ))
    return fingerprints


def _stage12_config(config: CullConfig) -> dict[str, object]:
    """Return the config fields that shaped Stage 1-2 output."""
    return {name: getattr(config, name) for name in STAGE12_CONFIG_FIELDS}


def _encoding_matrix(encodings: dict[str, object]) -> tuple[list[str], np.ndarray]:
    """Stack the duplicate detector's per-photo vectors in key order."""
    keys = list(encodings)
    if not keys:
        return keys, np.zeros((0, 0), dtype=np.float32)
    rows = [np.asarray(encodings[key]) for key in keys]
    if any(row.ndim != 1 or row.shape != rows[0].shape for row in rows):
        raise CheckpointError("duplicate-detector encodings are not equal-length vectors")
    return keys, np.stack(rows)


def _search_arrays(cache: _SearchCache | None) -> tuple[list[str] | None, np.ndarray]:
    """Return the CLIP search rows and their paths in row order."""
    if cache is None:
        return None, np.zeros((0, 0), dtype=np.float32)
    ordered = sorted(cache.path_to_row.items(), key=lambda item: item[1])
    if [row for _, row in ordered] != list(range(len(cache.embeddings))):
        raise CheckpointError("search cache rows do not match its path index")
    return [path for path, _ in ordered], np.asarray(cache.embeddings)


def _state_json(state: CheckpointState) -> bytes:
    """Serialise the state, keeping NaN/Infinity floats exact."""
    data = state.model_dump(
        mode="python", exclude={"s1_out": {"encodings"}, "s2_out": {"search_cache"}},
    )
    return pydantic_core.to_json(data, inf_nan_mode="constants")


def _build_state(save_in: CheckpointSaveInput) -> tuple[CheckpointState, np.ndarray, np.ndarray]:
    """Return the JSON state plus the encodings and search-embedding matrices."""
    encoding_keys, encodings = _encoding_matrix(save_in.s1_out.encodings)
    cache = save_in.s2_out.search_cache if save_in.s2_out is not None else None
    search_paths, search = _search_arrays(cache)
    state = CheckpointState(
        source_path=str(save_in.source_path),
        photos=fingerprint_photos(save_in.source_path, save_in.paths),
        config=_stage12_config(save_in.config), timings=save_in.timings,
        s1_out=save_in.s1_out, s2_out=save_in.s2_out,
        encoding_keys=encoding_keys, search_paths=search_paths,
    )
    return state, encodings, search


def save_checkpoint(save_in: CheckpointSaveInput) -> None:
    """Write the checkpoint atomically: a sibling temp file, then os.replace."""
    state, encodings, search = _build_state(save_in)
    payload = np.frombuffer(_state_json(state), dtype=np.uint8)
    tmp_path = save_in.path.with_name(f".{save_in.path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez(handle, **{_STATE_MEMBER: payload, _ENCODINGS_MEMBER: encodings, _SEARCH_MEMBER: search})
        os.replace(tmp_path, save_in.path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise CheckpointError(f"cannot write checkpoint {save_in.path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _read_members(path: Path) -> dict[str, np.ndarray]:
    """Read the archive's three members without unpickling anything."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in (_STATE_MEMBER, _ENCODINGS_MEMBER, _SEARCH_MEMBER)}
    except FileNotFoundError:
        raise CheckpointError(f"checkpoint not found: {path}") from None
    except (OSError, ValueError, KeyError) as exc:
        raise CheckpointError(f"{path} is not a cull stage checkpoint ({exc})") from None


def _parse_state(raw: np.ndarray, path: Path) -> CheckpointState:
    """Check format and version before validating the full state."""
    try:
        header = json.loads(raw.tobytes())
    except ValueError:
        raise CheckpointError(f"{path} has an unreadable checkpoint state") from None
    if not isinstance(header, dict) or header.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path} is not a cull stage checkpoint")
    if header.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"{path} has checkpoint version {header.get('version')}; this cull reads version "
            f"{CHECKPOINT_VERSION}. Re-run the Stage 1-2 phase with this version."
        )
    try:
        return CheckpointState.model_validate_json(raw.tobytes())
    except ValidationError as exc:
        raise CheckpointError(f"{path} has an invalid checkpoint state: {exc}") from None


def _restore_arrays(state: CheckpointState, members: dict[str, np.ndarray]) -> None:
    """Put the encodings and the search cache back on the Stage 1/2 outputs."""
    encodings = members[_ENCODINGS_MEMBER]
    if len(state.encoding_keys) != len(encodings):
        raise CheckpointError("checkpoint encodings do not match their key list")
    state.s1_out.encodings = {key: encodings[i] for i, key in enumerate(state.encoding_keys)}
    if state.s2_out is None or state.search_paths is None:
        return
    search = members[_SEARCH_MEMBER]
    if len(search) != len(state.search_paths):
        raise CheckpointError("checkpoint search embeddings do not match their path list")
    state.s2_out.search_cache = _SearchCache(
        embeddings=search, path_to_row={path: i for i, path in enumerate(state.search_paths)},
    )


def _describe_photo_mismatch(saved: list[PhotoFingerprint], current: list[PhotoFingerprint]) -> str:
    """Summarise which photos were added, removed or changed since the checkpoint."""
    saved_by_path = {p.path: p for p in saved}
    current_by_path = {p.path: p for p in current}
    added = sorted(set(current_by_path) - set(saved_by_path))
    removed = sorted(set(saved_by_path) - set(current_by_path))
    changed = sorted(k for k in set(saved_by_path) & set(current_by_path) if saved_by_path[k] != current_by_path[k])
    parts = [
        f"{len(names)} {label} (e.g. {', '.join(names[:_MISMATCH_EXAMPLES])})"
        for label, names in (("added", added), ("removed", removed), ("changed", changed)) if names
    ]
    return "; ".join(parts)


def _check_source(state: CheckpointState, load_in: CheckpointLoadInput) -> None:
    """Require the same source directory and the same, unchanged photos."""
    if Path(state.source_path).resolve() != load_in.source_path.resolve():
        raise CheckpointError(
            f"checkpoint was made for {state.source_path}, not {load_in.source_path}"
        )
    current = fingerprint_photos(load_in.source_path, load_in.paths)
    if current != state.photos:
        detail = _describe_photo_mismatch(state.photos, current)
        raise CheckpointError(f"photos in {load_in.source_path} changed since the checkpoint: {detail}")


def _check_config(state: CheckpointState, config: CullConfig) -> None:
    """Require the config fields that shaped Stage 1-2 to be unchanged."""
    current = _stage12_config(config)
    differing = [
        f"{name} (checkpoint {state.config.get(name)!r}, now {current[name]!r})"
        for name in STAGE12_CONFIG_FIELDS if state.config.get(name) != current[name]
    ]
    if differing:
        raise CheckpointError(f"options differ from the checkpointed run: {', '.join(differing)}")


def load_checkpoint(load_in: CheckpointLoadInput) -> CheckpointData:
    """Read and validate a checkpoint against this run's source, photos and config."""
    members = _read_members(load_in.path)
    state = _parse_state(members[_STATE_MEMBER], load_in.path)
    if state.stage != CHECKPOINT_STAGE:
        raise CheckpointError(f"{load_in.path} stops after stage {state.stage}; only stage 2 can resume")
    _check_source(state, load_in)
    _check_config(state, load_in.config)
    _restore_arrays(state, members)
    return CheckpointData(state=state, s1_out=state.s1_out, s2_out=state.s2_out)
