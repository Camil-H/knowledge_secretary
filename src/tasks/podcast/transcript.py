# src/tasks/podcast/transcript.py
"""Longform episode transcript generation.

The episode is written part by part against our own LLM seam: each part gets its own slice of
the research, a role instruction, a word target, and a bounded tail of the transcript so far.
Owning turn length here is what keeps the audio layer inside Cloud TTS's per-turn byte limit."""

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent / "transcript_prompt.md").read_text()
MAX_SOURCE_CHARS = 12000
PARTS = 8  # 1 intro + 6 body + 1 outro
MAX_TURN_CHARS = 1200
CONTEXT_TAIL_CHARS = 2500
INTRO_SOURCE_CHARS = 1200
PERSON1, PERSON2 = "Person1", "Person2"
TURN_PATTERN = re.compile(r"<(Person[12])>(.*?)</\1>", re.DOTALL)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# == Exceptions ===============================================================


class TranscriptError(Exception):
    """A part came back unusable after repair (carries part_idx in the message)."""


# == Generation ===============================================================


@dataclass(frozen=True)
class PartBudget:
    words: int  # soft target, written into the part instruction
    max_tokens: int  # hard ceiling passed to call()


# Word targets sum to 5,550 — set by the Cloud TTS monthly character budget at 30 episodes a
# month, not by taste. Raising them costs money.
PART_BUDGETS: dict[str, PartBudget] = {
    "intro": PartBudget(words=250, max_tokens=600),
    "body": PartBudget(words=800, max_tokens=1600),
    "outro": PartBudget(words=500, max_tokens=1100),
}
_PART_INSTRUCTIONS: dict[str, str] = {
    "intro": (
        "This is the opening part. Greet the audience once, hook them on why this topic matters, "
        "and hand over to the first substantive explanation. Do not summarize the whole episode."
    ),
    "body": (
        "This is a middle part. Continue the conversation mid-flow, with no greeting and no "
        "sign-off, and spend it entirely on the source material below."
    ),
    "outro": (
        "This is the closing part. Cover the remaining material, land on where the topic is "
        "heading and what is still open, then sign off once, briefly."
    ),
}


def generate(topic: str, research: str, *, call: Callable[..., str]) -> str:
    """Full episode transcript in <Person1>/<Person2> markup.

    Truncates research to MAX_SOURCE_CHARS, generates PARTS parts with a rolling context
    tail, repairs and stitches. Raises TranscriptError when a part is unusable."""
    if not topic or not research:
        raise ValueError("topic and research are required")

    # one chunk per part that carries new material: the body parts plus the outro
    chunks = _chunk_research(research[:MAX_SOURCE_CHARS], PARTS - 1)
    parts: list[str] = []
    for index in range(PARTS):
        role = _part_role(index, PARTS)
        context = "\n".join(parts)[-CONTEXT_TAIL_CHARS:]
        part = call(
            system=_part_system(topic, role, context),
            user=_part_source(role, index, chunks),
            max_tokens=PART_BUDGETS[role].max_tokens,
        )
        parts.append(_repair(part, index))
    return _stitch(parts)


# == Helper Functions =========================================================


def _part_role(i: int, n: int) -> str:
    """The role tag keying PART_BUDGETS / _PART_INSTRUCTIONS for part i of n."""
    if i == 0:
        return "intro"
    if i == n - 1:
        return "outro"
    return "body"


def _part_system(topic: str, role: str, context: str) -> str:
    """The system prompt for one part: contract, role instruction, budget, rolling context."""
    sections = [
        SYSTEM_PROMPT,
        f"Episode topic: {topic}",
        _PART_INSTRUCTIONS[role],
        f"Write about {PART_BUDGETS[role].words} words for this part.",
        f"Keep every turn under {MAX_TURN_CHARS} characters.",
    ]
    if context:
        sections.append(f"The transcript so far ends with:\n{context}")
    return "\n\n".join(sections)


def _part_source(role: str, index: int, chunks: list[str]) -> str:
    """The research slice handed to part `index`: an opening taste, one body chunk, or the last."""
    if role == "intro":
        return _truncate_at_sentence(chunks[0], INTRO_SOURCE_CHARS)
    if role == "outro":
        return chunks[-1]
    return chunks[index - 1]


def _chunk_research(research: str, n_chunks: int) -> list[str]:
    """Split research into n_chunks roughly equal chunks, preferring sentence boundaries."""
    target = max(1, math.ceil(len(research) / n_chunks))
    pieces = [
        slice_ for s in _SENTENCE_BOUNDARY.split(research) if s for slice_ in _slices(s, target)
    ]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > target and len(chunks) < n_chunks - 1:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}" if current else piece
    chunks.append(current)
    return chunks + [""] * (n_chunks - len(chunks))


def _slices(text: str, size: int) -> list[str]:
    """`text` cut into size-bounded slices, so one runaway sentence can't own a whole chunk."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


def _repair(part_text: str, part_idx: int) -> str:
    """Only well-formed, capped, paired turns from one part's raw output.

    Raises TranscriptError when nothing usable survives."""
    turns = [(m.group(1), m.group(2).strip()) for m in TURN_PATTERN.finditer(part_text)]
    turns = [(speaker, text) for speaker, text in turns if text]
    # a max_tokens cut mid-turn leaves an unpaired opener, which breaks downstream Q/A pairing
    if turns and turns[-1][0] == PERSON1:
        turns.pop()
    if not turns:
        raise TranscriptError(f"no usable turns in part {part_idx}")
    return "\n".join(
        f"<{speaker}>{_cap_turn(text, part_idx)}</{speaker}>" for speaker, text in turns
    )


def _cap_turn(text: str, part_idx: int) -> str:
    """A turn trimmed to MAX_TURN_CHARS at a sentence boundary; ⚠️ when content is lost."""
    if len(text) <= MAX_TURN_CHARS:
        return text
    capped = _truncate_at_sentence(text, MAX_TURN_CHARS)
    logger.warning(
        "⚠️ transcript: part %d turn over cap, truncated %d chars", part_idx, len(text) - len(capped)
    )
    return capped


def _truncate_at_sentence(text: str, limit: int) -> str:
    """`text` cut to at most `limit` chars, at the last sentence end if there is one."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [head.rfind(mark) for mark in ".!?"]
    cut = max(ends)
    return head[: cut + 1].strip() if cut > 0 else head.strip()


def _stitch(parts: list[str]) -> str:
    """The parts as one strictly alternating transcript opened by Person1; ⚠️ on any drop."""
    kept: list[str] = []
    expected = PERSON1
    dropped = 0
    for part in parts:
        for match in TURN_PATTERN.finditer(part):
            speaker, text = match.group(1), match.group(2)
            if speaker != expected:
                dropped += 1
                continue
            kept.append(f"<{speaker}>{text}</{speaker}>")
            expected = PERSON2 if speaker == PERSON1 else PERSON1
    if dropped:
        logger.warning("⚠️ transcript: dropped %d out-of-sequence turn(s) while stitching", dropped)
    return "\n".join(kept)
