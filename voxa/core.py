"""Voxa core: loads and reuses the transcription engine.

The one rule that matters here: load the model ONCE, reuse it. Rebuilding
WhisperModel per request is the number-one performance mistake.

The engine lives in voxa.engine (based on faster-whisper, MIT — see
voxa/engine/LICENSE). Voxa is self-contained: it depends on no external
faster-whisper package.
"""

from typing import BinaryIO, Optional, Union

import numpy as np

from voxa.engine import WhisperModel

from voxa.types import Info, Segment, TranscriptResult, Word


class Transcriber:
    """Loads a WhisperModel once and reuses it across transcribe() calls.

    Args:
      model_size: model size or HF id (e.g. "base", "large-v3", or a path).
      device: "cpu" or "cuda".
      compute_type: "int8" (CPU default), "float16" (GPU), "int8_float16", ...
      vad_filter: default VAD on/off, overridable per call.
      vad_parameters: default Silero VAD params (dict), overridable per call.
      **model_kwargs: forwarded to WhisperModel (e.g. cpu_threads, download_root).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        **model_kwargs,
    ):
        # Loaded ONCE. Reused for the lifetime of this Transcriber.
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            **model_kwargs,
        )
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        *,
        word_timestamps: bool = False,
        **kwargs,
    ) -> TranscriptResult:
        """Transcribe audio and return a clean structured result.

        Inference in faster-whisper is lazy: model.transcribe() returns a
        generator and nothing runs until it is iterated. We fully consume it
        here so callers get concrete data, not a generator.
        """
        if audio is None:
            raise ValueError("audio is required")
        if isinstance(audio, str) and not audio.strip():
            raise ValueError("audio path is empty")
        if isinstance(audio, np.ndarray) and audio.size == 0:
            # Empty buffer: nothing to transcribe. Return an empty result
            # rather than pushing a zero-length array through the model.
            return TranscriptResult(segments=[], info=None)

        kwargs.setdefault("vad_filter", self.vad_filter)
        if self.vad_parameters is not None:
            kwargs.setdefault("vad_parameters", self.vad_parameters)

        try:
            segments, info = self.model.transcribe(
                audio,
                word_timestamps=word_timestamps,
                **kwargs,
            )
            raw_segments = list(segments)  # <-- inference actually runs HERE
        except Exception as exc:
            raise RuntimeError(f"transcription failed: {exc}") from exc

        return TranscriptResult(
            segments=[_to_segment(s, word_timestamps) for s in raw_segments],
            info=Info(
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
            ),
        )


def _to_segment(seg, with_words: bool) -> Segment:
    words = None
    if with_words and seg.words:
        words = [Word(start=w.start, end=w.end, word=w.word) for w in seg.words]
    return Segment(start=seg.start, end=seg.end, text=seg.text, words=words)
