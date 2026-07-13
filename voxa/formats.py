"""Output formats for a Voxa TranscriptResult. Stdlib only, no subtitle library.

Works on any object exposing `.segments` (each with `.start`, `.end`, `.text`)
and an optional `.info`, so tests can feed hand-built fakes without a model.
"""

import json
from dataclasses import asdict, is_dataclass


def to_json(result, indent: int = 2) -> str:
    """Dump the full result to JSON. Uses asdict for dataclasses."""
    data = asdict(result) if is_dataclass(result) else result
    return json.dumps(data, ensure_ascii=False, indent=indent)


def to_txt(result) -> str:
    """Plain text: one segment per line, stripped."""
    return "\n".join(seg.text.strip() for seg in result.segments)


def _timestamp(seconds: float, sep: str) -> str:
    """HH:MM:SS<sep>mmm. sep is ',' for SRT, '.' for VTT."""
    ms = round(max(0.0, seconds) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(result) -> str:
    """SubRip. Indexed cues, comma-millisecond timestamps."""
    blocks = []
    for i, seg in enumerate(result.segments, start=1):
        blocks.append(
            f"{i}\n"
            f"{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def to_vtt(result) -> str:
    """WebVTT. Header, dot-millisecond timestamps."""
    blocks = ["WEBVTT\n"]
    for seg in result.segments:
        blocks.append(
            f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}\n"
            f"{seg.text.strip()}\n"
        )
    return "\n".join(blocks)
