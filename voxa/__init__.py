from voxa import formats
from voxa.types import Info, Segment, TranscriptResult, Word

__all__ = [
    "Transcriber",
    "TranscriptResult",
    "Segment",
    "Word",
    "Info",
    "formats",
]


def __getattr__(name):
    # Transcriber pulls in voxa.engine (and ctranslate2 / av). Import it lazily
    # so `import voxa` for types/formats stays light and model-free.
    if name == "Transcriber":
        from voxa.core import Transcriber

        return Transcriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
