# src/clients/cloud_tts.py
"""Google Cloud Text-to-Speech transport: one turn per request, returned as raw LINEAR16 PCM.

The API answers with a WAV container, so unwrapping it belongs here — callers get frames they
can concatenate. Its key is a GCP API key, distinct from the AI Studio one.
"""

import logging
import os
import time

from google.cloud import texttospeech

from src import config
from src.core.errors import ExternalError

logger = logging.getLogger(__name__)

SOURCE = "cloud-tts"

_TRANSIENT_MARKERS = ("429", "quota", "503", "deadline")
_WAV_RIFF_MARKER = b"RIFF"
_WAV_DATA_MARKER = b"data"
_WAV_CHUNK_HEADER_BYTES = 8


# == Exceptions ===============================================================


class AudioError(ExternalError):
    """Episode audio could not be produced (synthesis or encoding failed)."""


# == Primitive ================================================================

_CLIENT: texttospeech.TextToSpeechClient | None = None


def synthesize_turn(text: str, voice: str) -> bytes:
    """One turn spoken by voice, as mono LINEAR16 PCM frames at config.TTS_PCM_RATE_HZ.

    A transient refusal is retried with capped exponential backoff, anything else raises
    AudioError immediately; the raise is the terminal signal, so no failure is logged here."""
    client = _client()
    attempts = max(config.TTS_RETRIES, 1)
    backoff = config.BACKOFF_START_S
    for attempt in range(attempts):
        try:
            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=texttospeech.VoiceSelectionParams(
                    language_code=config.TTS_LANGUAGE_CODE, name=voice
                ),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=config.TTS_PCM_RATE_HZ,
                ),
            )
        except Exception as e:
            if attempt == attempts - 1 or not _is_transient(e):
                raise AudioError(SOURCE, cause=e) from e
            logger.warning("⚠️ cloud tts: turn refused (%s); backoff %ss", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, config.BACKOFF_CAP_S)
            continue
        return _strip_wav_header(response.audio_content)
    raise AudioError(SOURCE, detail="turn retries exhausted")


def _client() -> texttospeech.TextToSpeechClient:
    """The memoized API-key client; a missing key fails the episode, not the run."""
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get(config.TTS_KEY_LABEL)
        if not key:
            raise AudioError(SOURCE, detail=f"{config.TTS_KEY_LABEL} unset")
        _CLIENT = texttospeech.TextToSpeechClient(client_options={"api_key": key})
    return _CLIENT


# == Helper Functions =========================================================


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


def _is_transient(e: Exception) -> bool:
    message = str(e).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)
