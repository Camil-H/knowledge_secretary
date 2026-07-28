# src/tasks/podcast/audio.py
"""Episode audio: the <Person1>/<Person2> transcript rendered by Google Cloud TTS.

One request per turn (LINEAR16), raw-PCM concatenation, then a single ffmpeg encode — one
encode at a pinned 32 kbps keeps the downstream duration heuristic (bytes / 4000) honest and
avoids re-encoding every segment. Turn length is capped by the transcript layer, so this
layer only guards it.
"""

import logging
import os
import shutil
import subprocess
import time

from google.cloud import texttospeech

from src.core import ledger as ledger_mod
from src.core.errors import ExternalError
from src.tasks.podcast.transcript import TURN_PATTERN

logger = logging.getLogger(__name__)

TTS_SOURCE = "cloud-tts"
VOICES: dict[str, str] = {
    "Person1": "en-US-Chirp3-HD-Iapetus",
    "Person2": "en-US-Chirp3-HD-Laomedeia",
}
PCM_RATE_HZ = 24_000
MP3_BITRATE = "32k"
MAX_TURN_BYTES = 4500
MONTH_CHAR_BUDGET = 1_000_000
# the Cloud TTS key keeps its historical secret name, so no CI secret has to be rotated
_TTS_KEY_LABEL = "GEMINI_API_KEY"
_LANGUAGE_CODE = "en-US"
_TTS_RETRIES = int(os.environ.get("TTS_RETRIES", "3"))
_BACKOFF_START_S = 2
_BACKOFF_CAP_S = 30
_TRANSIENT_MARKERS = ("429", "quota", "503", "deadline")
_WAV_RIFF_MARKER = b"RIFF"
_WAV_DATA_MARKER = b"data"
_WAV_CHUNK_HEADER_BYTES = 8
_STDERR_TAIL_CHARS = 400


# == Exceptions ===============================================================


class AudioError(ExternalError):
    """Episode audio could not be produced (synthesis or encoding failed)."""


# == Synthesis ================================================================


def synthesize(transcript: str, out_path: str, *, ledger: ledger_mod.Ledger) -> str:
    """Render the <Person1>/<Person2> transcript to an mp3 at out_path; returns out_path.

    One Cloud TTS request per turn (LINEAR16), raw-PCM concatenation, one ffmpeg encode.
    Raises AudioError on synthesis or encoding failure."""
    turns = _turns(transcript)
    if not turns:
        raise AudioError(TTS_SOURCE, detail="transcript has no Person1/Person2 turns")
    _meter(ledger, turns)

    logger.info("🚀 podcast audio: %d turns via cloud tts", len(turns))
    client = _client()
    pcm = b"".join(
        _strip_wav_header(_synthesize_turn(client, text, VOICES[speaker]))
        for speaker, text in turns
    )
    _encode_mp3(pcm, out_path)
    logger.info("✅ podcast audio: %s from %d pcm bytes", out_path, len(pcm))
    return out_path


def _synthesize_turn(client: texttospeech.TextToSpeechClient, text: str, voice: str) -> bytes:
    """One turn as a WAV-containered LINEAR16 payload.

    Owns the transport's retries: a transient refusal is retried with capped exponential
    backoff, anything else raises AudioError immediately."""
    attempts = max(_TTS_RETRIES, 1)
    backoff = _BACKOFF_START_S
    for attempt in range(attempts):
        try:
            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=texttospeech.VoiceSelectionParams(language_code=_LANGUAGE_CODE, name=voice),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=PCM_RATE_HZ,
                ),
            )
        except Exception as e:
            if attempt == attempts - 1 or not _is_transient(e):
                raise AudioError(TTS_SOURCE, cause=e) from e
            logger.warning("⚠️ podcast audio: turn refused (%s); backoff %ss", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_CAP_S)
            continue
        return response.audio_content
    raise AudioError(TTS_SOURCE, detail="turn retries exhausted")


def _client() -> texttospeech.TextToSpeechClient:
    """An API-key client, built once per episode; a missing key fails the episode, not the run."""
    key = os.environ.get(_TTS_KEY_LABEL)
    if not key:
        raise AudioError(TTS_SOURCE, detail=f"{_TTS_KEY_LABEL} unset")
    return texttospeech.TextToSpeechClient(client_options={"api_key": key})


# == Helper Functions =========================================================


def _turns(transcript: str) -> list[tuple[str, str]]:
    """(speaker, text) pairs in order, skipping empty turns.

    An oversized turn is a transcript-layer bug now that turn length is capped there, so it
    fails the episode rather than being silently truncated here."""
    turns: list[tuple[str, str]] = []
    for match in TURN_PATTERN.finditer(transcript):
        text = match.group(2).strip()
        if not text:
            continue
        size = len(text.encode())
        if size > MAX_TURN_BYTES:
            raise AudioError(TTS_SOURCE, detail=f"turn of {size} bytes exceeds {MAX_TURN_BYTES}")
        turns.append((match.group(1), text))
    return turns


def _meter(ledger: ledger_mod.Ledger, turns: list[tuple[str, str]]) -> None:
    """Charge the episode's characters to the month's Cloud TTS bucket.

    Overage warns and proceeds: past the free tier the episode costs cents, while skipping it
    costs the day's episode."""
    chars = sum(len(text) for _, text in turns)
    month_total = ledger_mod.consume_tts_chars(ledger, chars)
    if month_total > MONTH_CHAR_BUDGET:
        logger.warning(
            "⚠️ podcast audio: cloud tts free tier exceeded this month (%d chars)", month_total
        )


def _strip_wav_header(audio: bytes) -> bytes:
    """The PCM frames of a WAV-containered LINEAR16 response.

    The prelude is not reliably 44 bytes, so the data chunk is located rather than assumed.
    Only a RIFF payload is searched: those four bytes occur freely inside real samples, so
    scanning headerless audio would silently cut a turn short."""
    if not audio.startswith(_WAV_RIFF_MARKER):
        return audio
    marker = audio.find(_WAV_DATA_MARKER)
    if marker < 0:
        return audio
    return audio[marker + _WAV_CHUNK_HEADER_BYTES :]


def _encode_mp3(pcm: bytes, out_path: str) -> None:
    """Encode raw mono PCM to an mp3 at MP3_BITRATE; raises AudioError if ffmpeg can't."""
    if not shutil.which("ffmpeg"):
        raise AudioError(TTS_SOURCE, detail="ffmpeg not on PATH")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(PCM_RATE_HZ),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-b:a",
            MP3_BITRATE,
            out_path,
        ],
        input=pcm,
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode(errors="replace")[-_STDERR_TAIL_CHARS:]
        raise AudioError(TTS_SOURCE, detail=f"ffmpeg failed: {tail}")


def _is_transient(e: Exception) -> bool:
    message = str(e).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)
