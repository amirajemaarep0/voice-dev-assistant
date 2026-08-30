"""Phase 2 - local speech-to-text with Whisper.

Uses faster-whisper (CTranslate2). Same Whisper weights as the reference
OpenAI implementation, roughly 3-4x faster on CPU and with a much smaller
memory footprint - which is what makes it viable next to a running LLM on
an 8 GB machine.

The model is loaded lazily and cached: the first transcription pays the
download/load cost, later ones do not.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import config


@dataclass
class Transcript:
    text: str
    language: str = ""
    duration: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@lru_cache(maxsize=2)
def load_model(
    size: str = config.WHISPER_MODEL_SIZE,
    device: str = config.WHISPER_DEVICE,
    compute_type: str = config.WHISPER_COMPUTE_TYPE,
):
    """Load (and cache) a Whisper model. First call downloads the weights."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - environment issue
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe_file(
    audio_path: Path | str,
    language: str | None = None,
    model=None,
) -> Transcript:
    """Transcribe a WAV/MP3/M4A file already on disk."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    model = model or load_model()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,          # greedy: noticeably faster, fine for dictation
        vad_filter=True,      # drop leading/trailing silence
    )
    text = "".join(segment.text for segment in segments).strip()
    return Transcript(
        text=text,
        language=getattr(info, "language", "") or "",
        duration=float(getattr(info, "duration", 0.0) or 0.0),
    )


def transcribe_bytes(
    audio_bytes: bytes,
    suffix: str = ".wav",
    language: str | None = None,
    model=None,
) -> Transcript:
    """Transcribe raw audio bytes (what Streamlit's mic widget hands us)."""
    if not audio_bytes:
        raise TranscriptionError("No audio was recorded.")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        return transcribe_file(tmp.name, language=language, model=model)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass
