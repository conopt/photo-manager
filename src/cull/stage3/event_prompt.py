"""Rated-preset prompts and their response parser: one 1-5 rating per photo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cull.config import EVENT_PRESET, EVENT_RATING_MAX, EVENT_RATING_MIN, FAMILY_PRESET
from cull.models import Stage3Result
from cull.stage3.parser import _clean_json_text, _extract_json_text

logger = logging.getLogger(__name__)

# Rejection criteria and reply format are verbatim from the prompt measured on
# the 488-photo singles-mixer shoot (keeper AUC 0.72 from `rating` alone with
# qwen3.8-27b). Only the opening names a generic event and client.
EVENT_PROMPT: str = """\
You are culling photos from an event for the client.
They want flattering, lively candids: people engaged in conversation, genuine
smiles and laughter, pairs and small groups connecting, a few wide room shots.
Guests will see these, so an unflattering frame of a person is a reject.

Reject for: missed focus or blur on the main face, closed or half-closed eyes
on a main subject (a squint from genuine laughter is fine), mid-chew or
mid-word grimace, the main subject cut off badly at the frame edge, backs of
heads dominating the frame, a blurry foreground object blocking the subject,
no clear subject.

Reply with ONLY one JSON object, no other text:
{"rating": <1-5, 5 = hero shot>, "keep": <true|false>, "flags": [<zero or more of
"eyes_closed","blur","awkward_expression","mid_chew","back_of_head","face_cut_off",
"obstructed_fg","cluttered","no_clear_subject","great_expression","connection">]}"""

# A family's own photos of their baby or young child. The event prompt rejects
# what such an archive treasures (a sleeping, yawning, crying or feeding baby
# reads as closed eyes, awkward expression or mid-chew) and caps a solo child
# below group "connection" frames; on a 330-photo family shoot every rating-5
# frame but two carried "connection" and no frame of the baby alone scored 5.
# The grade ladder keeps 5 for standouts: without it every clear baby frame
# rated 5 (111 of 330); with a "one in ten" quota, 1 of 330.
FAMILY_PROMPT: str = """\
You are culling a family's own photos for their private archive. Their baby or
young child is the main subject. The parents keep only the best frames.

Judge the child, not the adults. A sleeping, yawning, crying or feeding child
is a real moment: closed eyes or an open mouth on the child is never a fault
by itself. An adult's expression, closed eyes or turned back does not matter.

Most good photos of the child are a 4; keep 5 for the best of the shoot.
5: a standout the parents would print. The child's face is sharp and clearly
   seen, with a strong expression or moment (eye contact, a smile, a big yawn,
   a peaceful sleeping face, a tender touch), in pleasing light and a clean frame.
4: a good, sharp photo of the child that is a little ordinary: a neutral
   expression, plain framing or flat light.
3: an ordinary record: soft focus, a dull expression, awkward or cluttered
   framing, the child small in the frame, or a frame without the child.
1-2: technical failure on the child: missed focus or motion blur on the face,
   a foreground object blocking the child, the face badly cut off at the frame
   edge, severe over- or underexposure, no clear subject.

Reply with ONLY one JSON object, no other text:
{"rating": <1-5>, "keep": <true|false>, "flags": [<zero or more of
"child","tender_moment","great_expression","connection","eyes_closed","blur",
"face_cut_off","obstructed_fg","cluttered","no_clear_subject">]}"""

# Every flag either prompt may return; the parser drops anything else.
EVENT_FLAGS: frozenset[str] = frozenset({
    "eyes_closed", "blur", "awkward_expression", "mid_chew", "back_of_head",
    "face_cut_off", "obstructed_fg", "cluttered", "no_clear_subject",
    "great_expression", "connection", "child", "tender_moment",
})

_RATING_PROMPTS: dict[str, str] = {EVENT_PRESET: EVENT_PROMPT, FAMILY_PRESET: FAMILY_PROMPT}


def rating_prompt(preset: str | None) -> str:
    """Return the rating prompt for a rated preset; event for anything else."""
    return _RATING_PROMPTS.get(preset or EVENT_PRESET, EVENT_PROMPT)

_UNKNOWN_PATH = Path("unknown")


def parse_rating(value: object) -> int | None:
    """Return value as an int rating in 1-5, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not number.is_integer():
        return None
    rating = int(number)
    return rating if EVENT_RATING_MIN <= rating <= EVENT_RATING_MAX else None


def _parse_flags(value: object) -> list[str]:
    """Return the known flags in reply order; ignore anything else."""
    if not isinstance(value, list):
        return []
    return [flag for flag in value if isinstance(flag, str) and flag in EVENT_FLAGS]


def _load_reply(text: str) -> dict[str, object] | None:
    """Return the first JSON object in text, or None."""
    json_text = _extract_json_text(text)
    if json_text is None:
        return None
    try:
        data = json.loads(_clean_json_text(json_text))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_event_response(text: str) -> Stage3Result:
    """Return a Stage3Result carrying rating and flags, or a parse-error result."""
    data = _load_reply(text)
    rating = parse_rating(data.get("rating")) if data is not None else None
    if data is None or rating is None:
        logger.warning("parse_event_response: no 1-5 rating in reply: %.120r", text)
        return Stage3Result(photo_path=_UNKNOWN_PATH, is_parse_error=True)
    return Stage3Result(
        photo_path=_UNKNOWN_PATH, rating=rating, flags=_parse_flags(data.get("flags")),
    )
