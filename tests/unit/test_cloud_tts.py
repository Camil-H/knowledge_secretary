"""The Cloud TTS transport: the request each turn composes, its retry loop, transient
classification, key handling and the WAV unwrap. The TTS client and sleep are faked — no real
request, key or wait."""

import pytest

import src.clients.cloud_tts as cloud_tts
from src import config
from src.clients.cloud_tts import AudioError, _is_transient, _strip_wav_header, synthesize_turn

_VOICE = "en-US-Chirp3-HD-Iapetus"


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


def _fake_client(monkeypatch, raises: list[Exception] | None = None) -> _FakeTTSClient:
    client = _FakeTTSClient(raises)
    monkeypatch.setattr(cloud_tts, "_CLIENT", client)
    return client


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No sleeps, no memoized client and no key carried between tests."""
    monkeypatch.setattr(cloud_tts.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cloud_tts, "_CLIENT", None)
    monkeypatch.delenv(config.TTS_KEY_LABEL, raising=False)


# ===== Primitive =====


def test_synthesize_turn_requests_the_text_and_voice_as_linear16_at_the_pcm_rate(monkeypatch):
    client = _fake_client(monkeypatch)

    synthesize_turn("Hello and welcome.", _VOICE)

    assert client.calls == [
        {
            "text": "Hello and welcome.",
            "voice_name": _VOICE,
            "language_code": config.TTS_LANGUAGE_CODE,
            "encoding": cloud_tts.texttospeech.AudioEncoding.LINEAR16,
            "sample_rate": config.TTS_PCM_RATE_HZ,
        }
    ]


def test_synthesize_turn_returns_the_frames_without_the_wav_container(monkeypatch):
    _fake_client(monkeypatch)
    assert synthesize_turn("One line only.", _VOICE) == b"pcm1"


def test_synthesize_turn_retries_a_transient_refusal_then_succeeds(monkeypatch):
    failures: list[Exception] = [RuntimeError("429 quota exceeded")] * (config.TTS_RETRIES - 1)
    client = _fake_client(monkeypatch, failures)

    assert synthesize_turn("One line only.", _VOICE)
    assert len(client.calls) == config.TTS_RETRIES


def test_synthesize_turn_gives_up_once_the_retries_are_spent(monkeypatch):
    failures: list[Exception] = [RuntimeError("503 backend unavailable")] * config.TTS_RETRIES
    client = _fake_client(monkeypatch, failures)

    with pytest.raises(AudioError, match="503"):
        synthesize_turn("One line only.", _VOICE)
    assert len(client.calls) == config.TTS_RETRIES


def test_synthesize_turn_does_not_retry_a_non_transient_failure(monkeypatch):
    client = _fake_client(monkeypatch, [ValueError("bad voice")])

    with pytest.raises(AudioError, match="bad voice"):
        synthesize_turn("One line only.", _VOICE)
    assert len(client.calls) == 1


def test_synthesize_turn_backs_off_with_a_capped_delay(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(cloud_tts.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(config, "TTS_RETRIES", 6)
    monkeypatch.setattr(config, "BACKOFF_CAP_S", 4)
    _fake_client(monkeypatch, [RuntimeError("429 quota")] * 6)

    with pytest.raises(AudioError):
        synthesize_turn("One line only.", _VOICE)
    assert slept == [2, 4, 4, 4, 4]


def test_client_without_a_key_is_an_audio_error():
    with pytest.raises(AudioError, match=config.TTS_KEY_LABEL):
        cloud_tts._client()


def test_client_reuses_the_memoized_client(monkeypatch):
    client = _fake_client(monkeypatch)
    assert cloud_tts._client() is client


# ----- _is_transient -----


@pytest.mark.parametrize(
    "message, expected",
    [
        ("429 Too Many Requests", True),
        ("Quota exceeded for characters", True),
        ("503 Service Unavailable", True),
        ("Deadline Exceeded", True),
        ("invalid voice name", False),
    ],
)
def test_is_transient_classifies_the_refusal(message, expected):
    assert _is_transient(RuntimeError(message)) is expected


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
