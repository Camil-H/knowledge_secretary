"""Cloud TTS orchestration: turn parsing, the voice registry, PCM stitching, the ffmpeg
encode and the character meter. The TTS client, ffmpeg, the ledger and sleep are all faked —
no request, key, binary or wait is real."""

import pytest

import src.tasks.podcast.audio as audio
from src.tasks.podcast.audio import (
    MAX_TURN_BYTES,
    MONTH_CHAR_BUDGET,
    MP3_BITRATE,
    PCM_RATE_HZ,
    VOICES,
    AudioError,
    _strip_wav_header,
    _turns,
    synthesize,
)

_TRANSCRIPT = (
    "<Person1>Hello and welcome to the show.</Person1>\n"
    "<Person2>Glad to be here today.</Person2>\n"
    "<Person1>Let us get into it.</Person1>"
)
_OUT = "/tmp/does-not-matter/episode.mp3"


# ----- test doubles -----


def _wav(payload: bytes) -> bytes:
    header = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 16
    return header + b"data" + len(payload).to_bytes(4, "little") + payload


class _Response:
    def __init__(self, audio_content: bytes) -> None:
        self.audio_content = audio_content


class _FakeTTSClient:
    """Records the request each turn composes and hands back distinguishable payloads."""

    def __init__(self, raises: list[Exception] | None = None) -> None:
        self.calls: list[dict] = []
        self.raises = list(raises or [])

    def synthesize_speech(self, *, input, voice, audio_config):
        self.calls.append(
            {
                "text": input.text,
                "voice_name": voice.name,
                "language_code": voice.language_code,
                "encoding": audio_config.audio_encoding,
                "sample_rate": audio_config.sample_rate_hertz,
            }
        )
        if self.raises:
            raise self.raises.pop(0)
        return _Response(_wav(f"pcm{len(self.calls)}".encode()))


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


def _install(monkeypatch, *, client=None, ffmpeg=None, totals=None, has_ffmpeg=True):
    """Fake every boundary; returns (client, ffmpeg recorder, list of metered char counts)."""
    client = client or _FakeTTSClient()
    ffmpeg = ffmpeg or _FfmpegRecorder()
    metered: list[int] = []
    totals = list(totals or [])

    def _consume(ledger, chars, **kwargs):
        metered.append(chars)
        return totals.pop(0) if totals else chars

    monkeypatch.setattr(audio, "_client", lambda: client)
    monkeypatch.setattr(
        audio.shutil, "which", lambda name: "/usr/bin/ffmpeg" if has_ffmpeg else None
    )
    monkeypatch.setattr(audio.subprocess, "run", ffmpeg)
    monkeypatch.setattr(audio.ledger_mod, "consume_tts_chars", _consume)
    return client, ffmpeg, metered


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(audio.time, "sleep", lambda seconds: None)


# ===== Synthesis =====


def test_synthesize_returns_the_out_path(monkeypatch):
    _install(monkeypatch)
    assert synthesize(_TRANSCRIPT, _OUT, ledger={}) == _OUT


def test_synthesize_maps_each_speaker_to_its_registered_voice(monkeypatch):
    client, _, _ = _install(monkeypatch)
    synthesize(_TRANSCRIPT, _OUT, ledger={})

    assert [call["voice_name"] for call in client.calls] == [
        VOICES["Person1"],
        VOICES["Person2"],
        VOICES["Person1"],
    ]
    assert [call["text"] for call in client.calls] == [text for _, text in _turns(_TRANSCRIPT)]


def test_synthesize_requests_linear16_at_the_pcm_rate(monkeypatch):
    client, _, _ = _install(monkeypatch)
    synthesize(_TRANSCRIPT, _OUT, ledger={})

    expected = audio.texttospeech.AudioEncoding.LINEAR16
    for call in client.calls:
        assert call["language_code"] == audio._LANGUAGE_CODE
        assert call["encoding"] == expected
        assert call["sample_rate"] == PCM_RATE_HZ


def test_synthesize_hands_ffmpeg_the_stripped_payloads_in_turn_order(monkeypatch):
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
    for pair in (["-f", "s16le"], ["-ar", str(PCM_RATE_HZ)], ["-ac", "1"], ["-b:a", MP3_BITRATE]):
        index = argv.index(pair[0])
        assert argv[index : index + 2] == pair


def test_synthesize_retries_a_transient_refusal_then_succeeds(monkeypatch):
    failures: list[Exception] = [RuntimeError("429 quota exceeded")] * (audio._TTS_RETRIES - 1)
    client, ffmpeg, _ = _install(monkeypatch, client=_FakeTTSClient(raises=failures))

    assert synthesize("<Person1>One line only.</Person1>", _OUT, ledger={}) == _OUT
    assert len(client.calls) == audio._TTS_RETRIES


def test_synthesize_gives_up_once_the_retries_are_spent(monkeypatch):
    failures: list[Exception] = [RuntimeError("503 backend unavailable")] * audio._TTS_RETRIES
    client, _, _ = _install(monkeypatch, client=_FakeTTSClient(raises=failures))

    with pytest.raises(AudioError, match="503"):
        synthesize("<Person1>One line only.</Person1>", _OUT, ledger={})
    assert len(client.calls) == audio._TTS_RETRIES


def test_synthesize_does_not_retry_a_non_transient_failure(monkeypatch):
    client, _, _ = _install(monkeypatch, client=_FakeTTSClient(raises=[ValueError("bad voice")]))

    with pytest.raises(AudioError, match="bad voice"):
        synthesize(_TRANSCRIPT, _OUT, ledger={})
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "transcript, ffmpeg, has_ffmpeg, expected",
    [
        pytest.param(
            _TRANSCRIPT, _FfmpegRecorder(returncode=1), True, "ffmpeg", id="encode_failed"
        ),
        pytest.param(_TRANSCRIPT, None, False, "ffmpeg", id="no_ffmpeg_binary"),
        pytest.param(
            f"<Person1>{'x' * (MAX_TURN_BYTES + 1)}</Person1>",
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
    _, ffmpeg, _ = _install(monkeypatch, totals=[MONTH_CHAR_BUDGET + 1])

    with caplog.at_level("WARNING"):
        assert synthesize(_TRANSCRIPT, _OUT, ledger={}) == _OUT
    assert any("free tier exceeded" in record.getMessage() for record in caplog.records)
    assert ffmpeg.pcm


def test_synthesize_stays_silent_under_the_month_budget(monkeypatch, caplog):
    _install(monkeypatch, totals=[MONTH_CHAR_BUDGET])

    with caplog.at_level("WARNING"):
        synthesize(_TRANSCRIPT, _OUT, ledger={})
    assert caplog.records == []


def test_client_without_a_key_is_an_audio_error(monkeypatch):
    monkeypatch.delenv(audio._TTS_KEY_LABEL, raising=False)
    with pytest.raises(AudioError, match=audio._TTS_KEY_LABEL):
        audio._client()


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


# ----- _strip_wav_header -----


def test_strip_wav_header_returns_the_data_chunk():
    assert _strip_wav_header(_wav(b"frames")) == b"frames"


@pytest.mark.parametrize(
    "bare",
    [b"\x01\x02\x03\x04", b"\x01\x02data\x03\x04"],
    ids=["no_marker", "samples_containing_the_marker"],
)
def test_strip_wav_header_passes_through_bare_pcm(bare):
    """A non-WAV LINEAR16 response must survive intact. The second row is the one that bites:
    "data" occurs freely in real samples, so scanning unconditionally would truncate a turn."""
    assert _strip_wav_header(bare) == bare
