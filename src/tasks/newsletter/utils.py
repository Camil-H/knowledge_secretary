"""Text helpers for the newsletter."""

import re


def clean(text: str) -> str:
    """Collapse runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()
