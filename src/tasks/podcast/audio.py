# src/tasks/podcast/audio.py
"""Episode audio: the <Person1>/<Person2> transcript rendered by Google Cloud TTS.

One request per turn (LINEAR16) via src/clients/cloud_tts.py, raw-PCM concatenation, then a
single ffmpeg encode — one encode at a pinned 32 kbps keeps the downstream duration heuristic
(bytes / 4000) honest and avoids re-encoding every segment. Turn length is capped by the
transcript layer, so this layer only guards it.
"""

import logging
import shutil
import subprocess

from src.clients import cloud_tts
from src.clients.cloud_tts import AudioError
from src.core import ledger as ledger_mod
from src.tasks.podcast.transcript import TURN_PATTERN

logger = logging.getLogger(__name__)

VOICES: dict[str, str] = {
    "Person1": "en-US-Chirp3-HD-Iapetus",
    "Person2": "en-US-Chirp3-HD-Laomedeia",
}
MP3_BITRATE = "32k"
MAX_TURN_BYTES = 4500
MONTH_CHAR_BUDGET = 1_000_000
_STDERR_TAIL_CHARS = 400


# == Synthesis ================================================================


def synthesize(transcript: str, out_path: str, *, ledger: ledger_mod.Ledger) -> str:
    """Render the <Person1>/<Person2> transcript to an mp3 at out_path; returns out_path.

    One Cloud TTS request per turn, raw-PCM concatenation, one ffmpeg encode.
    Raises AudioError on synthesis or encoding failure."""
    turns = _turns(transcript)
    if not turns:
        raise AudioError(cloud_tts.SOURCE, detail="transcript has no Person1/Person2 turns")
    _meter(ledger, turns)

    logger.info("🚀 podcast audio: %d turns via cloud tts", len(turns))
    pcm = b"".join(cloud_tts.synthesize_turn(text, VOICES[speaker]) for speaker, text in turns)
    _encode_mp3(pcm, out_path)
    logger.info("✅ podcast audio: %s from %d pcm bytes", out_path, len(pcm))
    return out_path


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
            raise AudioError(
                cloud_tts.SOURCE, detail=f"turn of {size} bytes exceeds {MAX_TURN_BYTES}"
            )
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


def _encode_mp3(pcm: bytes, out_path: str) -> None:
    """Encode raw mono PCM to an mp3 at MP3_BITRATE; raises AudioError if ffmpeg can't."""
    if not shutil.which("ffmpeg"):
        raise AudioError(cloud_tts.SOURCE, detail="ffmpeg not on PATH")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(cloud_tts.PCM_RATE_HZ),
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
        raise AudioError(cloud_tts.SOURCE, detail=f"ffmpeg failed: {tail}")
