"""Whisper wrapper, exercised with a fake model (no weights downloaded)."""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant import stt


class _Segment:
    def __init__(self, text):
        self.text = text


class _Info:
    language = "en"
    duration = 2.5


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel."""

    def __init__(self, segments=("Explain ", "the add function.")):
        self._segments = segments
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return (_Segment(s) for s in self._segments), _Info()


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    f = tmp_path / "clip.wav"
    f.write_bytes(b"RIFF....WAVEfmt ")
    return f


class TestTranscriptDataclass:
    def test_truthiness(self):
        assert stt.Transcript(text="hello")
        assert not stt.Transcript(text="   ")
        assert not stt.Transcript(text="")


class TestTranscribeFile:
    def test_joins_segments(self, wav):
        result = stt.transcribe_file(wav, model=FakeWhisper())
        assert result.text == "Explain the add function."
        assert result.language == "en"
        assert result.duration == 2.5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(stt.TranscriptionError):
            stt.transcribe_file(tmp_path / "nope.wav", model=FakeWhisper())

    def test_uses_vad_and_greedy_decoding(self, wav):
        model = FakeWhisper()
        stt.transcribe_file(wav, model=model)
        _, kwargs = model.calls[0]
        assert kwargs["vad_filter"] is True
        assert kwargs["beam_size"] == 1

    def test_language_is_forwarded(self, wav):
        model = FakeWhisper()
        stt.transcribe_file(wav, language="en", model=model)
        assert model.calls[0][1]["language"] == "en"

    def test_silence_yields_empty_transcript(self, wav):
        result = stt.transcribe_file(wav, model=FakeWhisper(segments=()))
        assert not result


class TestTranscribeBytes:
    def test_roundtrips_through_temp_file(self):
        result = stt.transcribe_bytes(b"RIFF....WAVE", model=FakeWhisper())
        assert result.text == "Explain the add function."

    def test_empty_audio_raises(self):
        with pytest.raises(stt.TranscriptionError, match="No audio"):
            stt.transcribe_bytes(b"", model=FakeWhisper())

    def test_temp_file_is_cleaned_up(self, monkeypatch):
        seen = {}

        def spy(path, language=None, model=None):
            seen["path"] = Path(path)
            return stt.Transcript(text="ok")

        monkeypatch.setattr(stt, "transcribe_file", spy)
        stt.transcribe_bytes(b"RIFF", model=FakeWhisper())
        assert not seen["path"].exists()
