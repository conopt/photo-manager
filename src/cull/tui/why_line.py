"""One plain-language line stating the AI decision on a photo and why."""

from __future__ import annotations

from rich.text import Text

from cull.decision_reason import DECISION_WORDS, WhyContext, build_why_text

__all__ = ["DECISION_COLOURS", "DECISION_WORDS", "WhyContext", "build_why_line", "build_why_text"]

DECISION_COLOURS: dict[str, str] = {
    "keeper": "bold green",
    "rejected": "bold red",
    "uncertain": "bold yellow",
    "duplicate": "bold magenta",
    "select": "bold cyan",
}


def build_why_line(ctx: WhyContext) -> Text:
    """Return the why text coloured by the AI decision."""
    return Text(build_why_text(ctx), style=DECISION_COLOURS[ctx.ai_label], no_wrap=True, overflow="ellipsis")
