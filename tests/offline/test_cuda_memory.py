"""PHOTO_MANAGER_CUDA_MEMORY_FRACTION parsing, validation and application."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from cull import cli, cuda_memory
from cull.cuda_memory import (
    CUDA_MEMORY_FRACTION_ENV,
    apply_cuda_memory_fraction,
    cuda_memory_fraction_from_env,
)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no fraction applied."""
    monkeypatch.setattr(cuda_memory, "_applied_fraction", None)


@pytest.mark.parametrize("raw,expected", [("0.5", 0.5), (" 1 ", 1.0), ("1e-3", 0.001), ("", None)])
def test_valid_values(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float | None) -> None:
    """Numbers in (0, 1] parse; an empty value means unset."""
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, raw)
    assert cuda_memory_fraction_from_env() == expected


def test_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No variable, no cap."""
    monkeypatch.delenv(CUDA_MEMORY_FRACTION_ENV, raising=False)
    assert cuda_memory_fraction_from_env() is None


@pytest.mark.parametrize("raw", ["0", "-0.2", "1.01", "nan", "inf", "half", "50%"])
def test_invalid_values_fail_clearly(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Out-of-range or non-numeric values name the variable and the value."""
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, raw)
    with pytest.raises(ValueError, match=CUDA_MEMORY_FRACTION_ENV) as failure:
        cuda_memory_fraction_from_env()
    assert repr(raw) in str(failure.value)


def _record_fraction_calls(monkeypatch: pytest.MonkeyPatch, device: str) -> list[float]:
    """Pretend the selected device is device and record set_per_process_memory_fraction calls."""
    calls: list[float] = []
    monkeypatch.setattr("cull.stage2.iqa.select_device", lambda: device)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device=None: calls.append(fraction))
    return calls


def test_applied_once_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """On CUDA the cap is set exactly once, however often apply is called."""
    calls = _record_fraction_calls(monkeypatch, "cuda")
    assert apply_cuda_memory_fraction(0.4) is True
    assert apply_cuda_memory_fraction(0.4) is True
    assert calls == [0.4]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_not_applied_without_cuda(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """Without CUDA (e.g. CUDA hidden from a CPU-only process) nothing is set."""
    calls = _record_fraction_calls(monkeypatch, device)
    assert apply_cuda_memory_fraction(0.4) is False
    assert calls == []


def test_unset_never_touches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No fraction, no device probing."""
    calls = _record_fraction_calls(monkeypatch, "cuda")
    assert apply_cuda_memory_fraction(None) is False
    assert calls == []


def test_cli_rejects_invalid_fraction_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command exits 1 with the reason on stderr before any other work."""
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "1.5")
    monkeypatch.setattr(cli, "_enforce_bootstrap_gate", lambda: pytest.fail("startup validation must run first"))
    run = CliRunner().invoke(cli._cull_pipeline_command, [str(tmp_path)])
    assert run.exit_code == 1
    assert CUDA_MEMORY_FRACTION_ENV in run.stderr
