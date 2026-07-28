"""Transcript generation: what each part asks the model for, and how malformed output is
repaired and stitched. The LLM seam is a recording stub — nothing here talks to a model."""

import pytest

from src.tasks.podcast.transcript import (
    _PART_INSTRUCTIONS,
    CONTEXT_TAIL_CHARS,
    MAX_SOURCE_CHARS,
    MAX_TURN_CHARS,
    PART_BUDGETS,
    PARTS,
    PERSON1,
    PERSON2,
    TURN_PATTERN,
    TranscriptError,
    _part_role,
    _repair,
    _stitch,
    generate,
)

_TOPIC = "PROTACs"
_CONTEXT_MARKER = "The transcript so far ends with:\n"


# == Doubles ==================================================================


class _Call:
    """Records (system, user, max_tokens) per invocation and returns a canned part."""

    def __init__(self, turn_chars: int = 40) -> None:
        self.calls: list[dict] = []
        self.turn_chars = turn_chars

    def __call__(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        index = len(self.calls)
        filler = f"Part {index} sentence. " * max(1, self.turn_chars // 20)
        return f"<{PERSON1}>{filler}</{PERSON1}><{PERSON2}>{filler}</{PERSON2}>"


def _research(chars: int = 6000, sentence: str = "The mechanism matters here. ") -> str:
    return (sentence * (chars // len(sentence) + 1))[:chars]


# == Generation ===============================================================


def test_generate_returns_an_alternating_transcript():
    transcript = generate(_TOPIC, _research(), call=_Call())
    speakers = [m.group(1) for m in TURN_PATTERN.finditer(transcript)]
    assert speakers == [PERSON1, PERSON2] * PARTS


@pytest.mark.parametrize(
    "topic, research", [("", "material"), (_TOPIC, "")], ids=["no_topic", "no_research"]
)
def test_generate_requires_a_topic_and_research(topic, research):
    with pytest.raises(ValueError, match="required"):
        generate(topic, research, call=_Call())


def test_generate_composes_one_request_per_part_from_the_role_registry():
    call = _Call()
    generate(_TOPIC, _research(), call=call)

    assert len(call.calls) == PARTS
    for index, recorded in enumerate(call.calls):
        budget = PART_BUDGETS[_part_role(index, PARTS)]
        assert recorded["max_tokens"] == budget.max_tokens
        assert f"{budget.words} words" in recorded["system"]
        assert _TOPIC in recorded["system"]
        assert str(MAX_TURN_CHARS) in recorded["system"]


def test_generate_gives_each_part_only_its_own_role_instruction():
    call = _Call()
    generate(_TOPIC, _research(), call=call)

    for index, recorded in enumerate(call.calls):
        role = _part_role(index, PARTS)
        assert _PART_INSTRUCTIONS[role] in recorded["system"]
        others = [text for key, text in _PART_INSTRUCTIONS.items() if key != role]
        assert all(text not in recorded["system"] for text in others)


def test_generate_bounds_the_rolling_context_handed_to_each_part():
    """podcastfy accumulated the whole conversation into every part's prompt; we pass a tail."""
    call = _Call(turn_chars=MAX_TURN_CHARS - 100)
    generate(_TOPIC, _research(), call=call)

    contexts = [
        recorded["system"].split(_CONTEXT_MARKER)[1]
        for recorded in call.calls
        if _CONTEXT_MARKER in recorded["system"]
    ]
    assert contexts  # later parts do get the conversation so far
    assert _CONTEXT_MARKER not in call.calls[0]["system"]  # nothing precedes the intro
    assert max(len(context) for context in contexts) == CONTEXT_TAIL_CHARS


def test_generate_caps_the_research_it_sends():
    tail = "TAILMARKER. " * 42
    call = _Call()
    generate(_TOPIC, _research(MAX_SOURCE_CHARS) + tail[:500], call=call)

    sent = "".join(recorded["user"] for recorded in call.calls)
    assert "TAILMARKER" not in sent
    assert "The mechanism matters here." in sent


# == Helper Functions =========================================================


# ----- _repair -----


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            f"<{PERSON1}>Hi.</{PERSON1}><{PERSON2}>Hello.</{PERSON2}>",
            [(PERSON1, "Hi."), (PERSON2, "Hello.")],
        ),
        (
            f"Here is the transcript:\n<{PERSON1}>Hi.</{PERSON1}>\nthanks!"
            f"<{PERSON2}>Hello.</{PERSON2}>\n(end of part)",
            [(PERSON1, "Hi."), (PERSON2, "Hello.")],
        ),
        (
            f"<{PERSON1}>Hi.</{PERSON1}><{PERSON2}>Hello.</{PERSON2}><{PERSON1}>And so",
            [(PERSON1, "Hi."), (PERSON2, "Hello.")],
        ),
        (
            f"<{PERSON1}>Hi.</{PERSON1}><{PERSON2}>Hello.</{PERSON2}><{PERSON1}>Bye.</{PERSON1}>",
            [(PERSON1, "Hi."), (PERSON2, "Hello.")],
        ),
        (
            f"<{PERSON1}>Hi.</{PERSON1}><{PERSON2}>   </{PERSON2}><{PERSON2}>Hello.</{PERSON2}>",
            [(PERSON1, "Hi."), (PERSON2, "Hello.")],
        ),
    ],
    ids=["clean", "prose_outside_tags", "truncated_opener", "unpaired_last_turn", "empty_turn"],
)
def test_repair_keeps_only_well_formed_paired_turns(raw, expected):
    assert [(m.group(1), m.group(2)) for m in TURN_PATTERN.finditer(_repair(raw, 0))] == expected


def test_repair_truncates_an_over_cap_turn_at_a_sentence_boundary(caplog):
    long_turn = "This sentence is long enough to matter. " * 60
    raw = f"<{PERSON1}>{long_turn}</{PERSON1}><{PERSON2}>Right.</{PERSON2}>"

    with caplog.at_level("WARNING"):
        repaired = _repair(raw, 2)

    first = TURN_PATTERN.findall(repaired)[0][1]
    assert len(first) <= MAX_TURN_CHARS
    assert first.endswith(".")
    assert any("⚠️" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "raw",
    ["", "Sorry, I cannot help with that.", f"<{PERSON1}>Only one opener.</{PERSON1}>"],
    ids=["empty", "refusal_prose", "single_unpaired_turn"],
)
def test_repair_raises_when_no_turn_survives(raw):
    with pytest.raises(TranscriptError, match="part 5"):
        _repair(raw, 5)


# ----- _stitch -----


@pytest.mark.parametrize(
    "parts, expected",
    [
        (
            [f"<{PERSON1}>A</{PERSON1}><{PERSON2}>B</{PERSON2}>"] * 2,
            [PERSON1, PERSON2, PERSON1, PERSON2],
        ),
        (
            [
                f"<{PERSON1}>A</{PERSON1}><{PERSON2}>B</{PERSON2}>",
                f"<{PERSON2}>C</{PERSON2}><{PERSON1}>D</{PERSON1}><{PERSON2}>E</{PERSON2}>",
            ],
            [PERSON1, PERSON2, PERSON1, PERSON2],
        ),
        (
            [f"<{PERSON1}>A</{PERSON1}><{PERSON1}>B</{PERSON1}><{PERSON2}>C</{PERSON2}>"],
            [PERSON1, PERSON2],
        ),
        (
            [f"<{PERSON2}>A</{PERSON2}><{PERSON1}>B</{PERSON1}><{PERSON2}>C</{PERSON2}>"],
            [PERSON1, PERSON2],
        ),
    ],
    ids=["already_alternating", "person2_opens_a_part", "double_person1", "person2_opens_episode"],
)
def test_stitch_enforces_strict_alternation_from_person1(parts, expected):
    assert [m.group(1) for m in TURN_PATTERN.finditer(_stitch(parts))] == expected


def test_stitch_warns_when_it_drops_a_turn(caplog):
    parts = [f"<{PERSON1}>A</{PERSON1}><{PERSON1}>B</{PERSON1}><{PERSON2}>C</{PERSON2}>"]
    with caplog.at_level("WARNING"):
        _stitch(parts)
    assert any("⚠️" in record.message for record in caplog.records)
