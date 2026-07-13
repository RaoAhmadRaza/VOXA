"""Core tests. The real-model test loads "tiny" (mirrors tests/test_transcribe.py)
and downloads on first run; input-guard tests need no model."""

import numpy as np
import pytest

from voxa import Transcriber
from voxa.core import Info, Segment, TranscriptResult


def test_empty_ndarray_returns_empty_result_without_model():
    # Guard runs before any model call, so a bare (uninitialized) instance is fine.
    t = Transcriber.__new__(Transcriber)
    t.vad_filter = True
    t.vad_parameters = None
    result = t.transcribe(np.array([], dtype=np.float32))
    assert isinstance(result, TranscriptResult)
    assert result.segments == []
    assert result.info is None


def test_none_audio_raises():
    t = Transcriber.__new__(Transcriber)
    with pytest.raises(ValueError):
        t.transcribe(None)


def test_empty_path_raises():
    t = Transcriber.__new__(Transcriber)
    with pytest.raises(ValueError):
        t.transcribe("   ")


def test_transcribe_real_tiny_model(jfk_path):
    t = Transcriber("tiny", device="cpu", compute_type="int8")
    result = t.transcribe(jfk_path, word_timestamps=True, vad_filter=False)

    assert isinstance(result.info, Info)
    assert result.info.language == "en"
    assert result.info.language_probability > 0.9
    assert len(result.segments) >= 1

    seg = result.segments[0]
    assert isinstance(seg, Segment)
    assert seg.text.strip()
    assert seg.end > seg.start
    assert seg.words and seg.words[0].word
