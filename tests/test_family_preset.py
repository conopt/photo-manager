"""Family preset: rated like event, with a baby-first prompt and a face-neutral tiebreak."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cull._pipeline.event_routing import (
    EventRouteInput,
    collect_event_scores,
    route_event,
)
from cull._pipeline.stage3_runner import _rate_single_event, _Stage3LoopInput
from cull.cli_config import PRESET_CHOICES
from cull.config import (
    CLUSTER_THRESHOLD,
    EVENT_VLM_ALIAS,
    GENRE_WEIGHTS,
    PRESET_QUALITY_POLICY,
    CullConfig,
)
from cull.models import PortraitScores, Stage2Result, Stage3Result
from cull.pipeline import _Stage1Output, _Stage2Output
from cull.stage2.fusion import FusionResult
from cull.stage3.event_prompt import (
    EVENT_PROMPT,
    FAMILY_PROMPT,
    parse_event_response,
    rating_prompt,
)
from cull.stage3.event_score import cheap_prob, event_score
from cull.stage3.event_scoring import rate_photo
from cull.stage3.prompt import PromptContext
from cull.stage3.vlm_scoring import VlmRequest, VlmScoreCallInput
from cull.vlm_registry import VLMEntry
from cull.vlm_session import VlmSession

FAMILY_REPLY = '{"rating": 5, "keep": true, "flags": ["child", "tender_moment", "eyes_closed"]}'


class FakeVlmSession(VlmSession):
    """VlmSession whose generate() returns one canned reply and records prompts."""

    _reply: str = ""
    _prompts: list[str] = []

    def configure(self, reply: str) -> None:
        """Set the canned reply."""
        object.__setattr__(self, "_reply", reply)
        object.__setattr__(self, "_prompts", [])

    def generate(self, call_in: Any) -> str:
        """Return the canned reply."""
        self._prompts.append(call_in.prompt)
        return self._reply


def _fake_session(reply: str) -> FakeVlmSession:
    """Return a FakeVlmSession primed with reply."""
    entry = VLMEntry(alias="fake", directory=Path(tempfile.gettempdir()), display_name="Fake")
    session = FakeVlmSession(entry=entry)
    session.configure(reply)
    return session


def _stage2(path: Path, **overrides: Any) -> Stage2Result:
    """Return a Stage2Result with fixed IQA scores."""
    values: dict[str, Any] = {"topiq": 0.60, "clipiqa": 0.65, "laion_aesthetic": 0.50, "composite": 0.5}
    values.update(overrides)
    return Stage2Result(photo_path=path, **values)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_family_is_a_rated_preset() -> None:
    """Family routes and curates by the VLM rating, with event's default model."""
    config = CullConfig(preset="family")
    assert config.is_event is True
    assert config.model == EVENT_VLM_ALIAS
    assert "family" in PRESET_CHOICES


def test_family_has_every_per_preset_table() -> None:
    """Stage 2 and Stage 4 tables cover family; closed eyes cost a family photo nothing."""
    assert "family" in GENRE_WEIGHTS
    assert "family" in CLUSTER_THRESHOLD
    assert PRESET_QUALITY_POLICY["family"]["eyes_closed_penalty"] == 0.0


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_rating_prompt_by_preset() -> None:
    """Family gets its own prompt; event and anything unknown keep the event prompt."""
    assert rating_prompt("family") == FAMILY_PROMPT
    assert rating_prompt("event") == EVENT_PROMPT
    assert rating_prompt(None) == EVENT_PROMPT


def test_family_prompt_puts_the_child_first() -> None:
    """Baby moments the event prompt rejects are not faults, but only standouts rate 5."""
    assert "closed eyes or an open mouth on the child is never a fault" in FAMILY_PROMPT
    assert "A sleeping, yawning, crying or feeding child" in FAMILY_PROMPT
    assert "Most good photos of the child are a 4; keep 5 for the best of the shoot." in FAMILY_PROMPT
    assert "the child small in the frame, or a frame without the child" in FAMILY_PROMPT
    assert "mid-chew" not in FAMILY_PROMPT
    assert "back_of_head" not in FAMILY_PROMPT


def test_parser_reads_family_flags() -> None:
    """The family flags survive parsing."""
    result = parse_event_response(FAMILY_REPLY)
    assert result.rating == 5
    assert result.flags == ["child", "tender_moment", "eyes_closed"]


def test_rate_photo_sends_the_family_prompt(tmp_path: Path) -> None:
    """The request's preset selects the prompt sent to the VLM."""
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    session = _fake_session(FAMILY_REPLY)
    request = VlmRequest(image_path=image, context=PromptContext(preset="family"), model="fake")
    with patch("cull.stage3.vlm_scoring.time.sleep"):
        result = rate_photo(VlmScoreCallInput(request=request, session=session))
    assert result.is_keeper is True
    assert session._prompts == [FAMILY_PROMPT]


def test_stage3_rates_with_the_run_preset(tmp_path: Path) -> None:
    """Stage 3 carries the run's preset into the rating request."""
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    session = _fake_session(FAMILY_REPLY)
    loop_in = _Stage3LoopInput(
        ambiguous=[image], config=CullConfig(preset="family"), model_name="fake",
        s1_results={}, s2_results={}, session=session,
    )
    _rate_single_event(image, loop_in)
    assert session._prompts == [FAMILY_PROMPT]


# ---------------------------------------------------------------------------
# Tiebreak and routing
# ---------------------------------------------------------------------------


def test_family_tiebreak_ignores_face_terms() -> None:
    """A crying, eyes-closed face scores the same as no face; event still penalises it."""
    plain = _stage2(Path("a.jpg"))
    crying = _stage2(
        Path("a.jpg"),
        portrait=PortraitScores(valence=-0.8, arousal=0.6, eye_sharpness_left=0.0, is_eyes_closed=True),
    )
    assert cheap_prob(crying, "family") == pytest.approx(cheap_prob(plain, "family"))
    assert cheap_prob(crying) < cheap_prob(plain)


def test_family_tiebreak_follows_image_quality() -> None:
    """Higher TOPIQ, CLIPIQA and LAION rank higher."""
    sharp = _stage2(Path("a.jpg"), topiq=0.70)
    soft = _stage2(Path("a.jpg"), topiq=0.40)
    assert cheap_prob(sharp, "family") > cheap_prob(soft, "family")


def test_family_event_score_is_rating_plus_family_tiebreak() -> None:
    """The rating dominates; the family tiebreak adds 0.01 of itself."""
    stage2 = _stage2(Path("a.jpg"))
    rated = Stage3Result(photo_path=Path("a.jpg"), rating=5)
    assert event_score(stage2, rated, "family") == pytest.approx(5 + 0.01 * cheap_prob(stage2, "family"))


def _route_input(tmp_path: Path, preset: str) -> EventRouteInput:
    """Return two unrated singletons: a crying baby and a smiling adult of equal quality."""
    baby, adult = tmp_path / "baby.jpg", tmp_path / "adult.jpg"
    faces = {
        baby: PortraitScores(valence=-0.8, arousal=0.6, is_eyes_closed=True),
        adult: PortraitScores(valence=0.9, arousal=0.3),
    }
    stage2 = {p: _stage2(p, topiq=0.62 if p == baby else 0.60, portrait=face) for p, face in faces.items()}
    s1_out = _Stage1Output(results={}, survivors=list(faces))
    s2_out = _Stage2Output(results={
        str(p): FusionResult(stage2=s2, routing="AMBIGUOUS") for p, s2 in stage2.items()
    })
    return EventRouteInput(s1_out=s1_out, s2_out=s2_out, preset=preset)


def test_routing_uses_the_preset_tiebreak(tmp_path: Path) -> None:
    """Without ratings, family ranks the sharper crying baby first; event ranks the smiling adult."""
    family = collect_event_scores(_route_input(tmp_path, "family"))
    event = collect_event_scores(_route_input(tmp_path, "event"))
    baby, adult = str(tmp_path / "baby.jpg"), str(tmp_path / "adult.jpg")
    assert family[baby] > family[adult]
    assert event[adult] > event[baby]
    assert set(route_event(_route_input(tmp_path, "family"))) == {baby, adult}
