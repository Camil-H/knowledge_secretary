"""Cloud TTS orchestration: turn parsing, the voice registry, PCM stitching, the ffmpeg encode
and the character meter. The TTS transport, ffmpeg and the ledger are all faked — no request,
key or binary is real."""

import pytest

import src.tasks.podcast.audio as audio
from src import config
from src.clients.cloud_tts import AudioError
from src.tasks.podcast.audio import _turns, synthesize

_TRANSCRIPT = (
    "<Person1>Hello and welcome to the show.</Person1>\n"
    "<Person2>Glad to be here today.</Person2>\n"
    "<Person1>Let us get into it.</Person1>"
)
_OUT = "/tmp/does-not-matter/episode.mp3"


# ----- test doubles -----


class _FakeTurnSynthesizer:
    """Stands in for cloud_tts.synthesize_turn: records each turn, returns distinct frames."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.raises = raises

    def __call__(self, text: str, voice: str) -> bytes:
        self.calls.append((text, voice))
        if self.raises:
            raise self.raises
        return f"pcm{len(self.calls)}".encode()


class _Completed:
    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


class _FfmpegRecorder:
    def __init__(self, returncode: int = 0) -> None:
        self.argv: list[str] = []
        self.pcm = b""
        self.returncode = returncode

    def __call__(self, argv, *, input, capture_output):
        self.argv = argv
        self.pcm = input
        return _Completed(self.returncode, stderr=b"x" * 10 + b"unsupported sample format")


def _install(monkeypatch, *, turn=None, ffmpeg=None, totals=None, has_ffmpeg=True):
    """Fake every boundary; returns (turn synthesizer, ffmpeg recorder, metered char counts)."""
    turn = turn or _FakeTurnSynthesizer()
    ffmpeg = ffmpeg or _FfmpegRecorder()
    metered: list[int] = []
    totals = list(totals or [])

    def _consume(ledger, chars, **kwargs):
        metered.append(chars)
        return totals.pop(0) if totals else chars

    monkeypatch.setattr(audio.cloud_tts, "synthesize_turn", turn)
    monkeypatch.setattr(
        audio.shutil, "which", lambda name: "/usr/bin/ffmpeg" if has_ffmpeg else None
    )
    monkeypatch.setattr(audio.subprocess, "run", ffmpeg)
    monkeypatch.setattr(audio.ledger_mod, "consume_tts_chars", _consume)
    return turn, ffmpeg, metered


# ===== Synthesis =====


def test_synthesize_returns_the_out_path(monkeypatch):
    _install(monkeypatch)
    assert synthesize(_TRANSCRIPT, _OUT, ledger={}) == _OUT


def test_synthesize_maps_each_speaker_to_its_registered_voice(monkeypatch):
    turn, _, _ = _install(monkeypatch)
    synthesize(_TRANSCRIPT, _OUT, ledger={})

    assert turn.calls == [
        (text, config.TTS_VOICES[speaker]) for speaker, text in _turns(_TRANSCRIPT)
    ]


def test_synthesize_hands_ffmpeg_the_turn_payloads_in_order(monkeypatch):
    _, ffmpeg, _ = _install(monkeypatch)
    synthesize(_TRANSCRIPT, _OUT, ledger={})

    assert ffmpeg.pcm == b"pcm1pcm2pcm3"


def test_synthesize_encodes_mono_pcm_at_the_pinned_bitrate(monkeypatch):
    """The episode duration is read downstream as bytes / 4000, so the bitrate is load-bearing."""
    _, ffmpeg, _ = _install(monkeypatch)
    synthesize(_TRANSCRIPT, _OUT, ledger={})

    argv = ffmpeg.argv
    assert argv[0] == "ffmpeg"
    assert argv[-1] == _OUT
    rate = str(config.TTS_PCM_RATE_HZ)
    for pair in (["-f", "s16le"], ["-ar", rate], ["-ac", "1"], ["-b:a", config.TTS_MP3_BITRATE]):
        index = argv.index(pair[0])
        assert argv[index : index + 2] == pair


def test_synthesize_propagates_a_turn_failure(monkeypatch):
    boom = AudioError("cloud-tts", detail="turn retries exhausted")
    _install(monkeypatch, turn=_FakeTurnSynthesizer(raises=boom))

    with pytest.raises(AudioError) as ei:
        synthesize(_TRANSCRIPT, _OUT, ledger={})
    assert ei.value is boom


@pytest.mark.parametrize(
    "transcript, ffmpeg, has_ffmpeg, expected",
    [
        pytest.param(
            _TRANSCRIPT, _FfmpegRecorder(returncode=1), True, "ffmpeg", id="encode_failed"
        ),
        pytest.param(_TRANSCRIPT, None, False, "ffmpeg", id="no_ffmpeg_binary"),
        pytest.param(
            f"<Person1>{'x' * (config.TTS_MAX_TURN_BYTES + 1)}</Person1>",
            None,
            True,
            "bytes",
            id="turn_too_big",
        ),
        pytest.param("just prose, no markup", None, True, "no Person1", id="no_turns"),
    ],
)
def test_synthesize_raises_audio_error(monkeypatch, transcript, ffmpeg, has_ffmpeg, expected):
    _install(monkeypatch, ffmpeg=ffmpeg, has_ffmpeg=has_ffmpeg)
    with pytest.raises(AudioError, match=expected):
        synthesize(transcript, _OUT, ledger={})


def test_synthesize_meters_the_spoken_characters(monkeypatch):
    _, _, metered = _install(monkeypatch)
    synthesize(_TRANSCRIPT, _OUT, ledger={})

    assert metered == [sum(len(text) for _, text in _turns(_TRANSCRIPT))]


def test_synthesize_warns_but_proceeds_past_the_month_budget(monkeypatch, caplog):
    _, ffmpeg, _ = _install(monkeypatch, totals=[config.TTS_MONTH_CHAR_BUDGET + 1])

    with caplog.at_level("WARNING"):
        assert synthesize(_TRANSCRIPT, _OUT, ledger={}) == _OUT
    assert any("free tier exceeded" in record.getMessage() for record in caplog.records)
    assert ffmpeg.pcm


def test_synthesize_stays_silent_under_the_month_budget(monkeypatch, caplog):
    _install(monkeypatch, totals=[config.TTS_MONTH_CHAR_BUDGET])

    with caplog.at_level("WARNING"):
        synthesize(_TRANSCRIPT, _OUT, ledger={})
    assert caplog.records == []


# ----- _turns -----


def test_turns_pairs_each_speaker_with_its_text():
    assert _turns(_TRANSCRIPT) == [
        ("Person1", "Hello and welcome to the show."),
        ("Person2", "Glad to be here today."),
        ("Person1", "Let us get into it."),
    ]


@pytest.mark.parametrize(
    "transcript, expected",
    [
        ("<Person1>   </Person1><Person2>Real turn.</Person2>", [("Person2", "Real turn.")]),
        ("<Person3>Not a host.</Person3>", []),
        ("", []),
    ],
    ids=["empty_turn_skipped", "unknown_speaker_ignored", "empty_transcript"],
)
def test_turns_edge_cases(transcript, expected):
    assert _turns(transcript) == expected
