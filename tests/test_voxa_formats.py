"""Format tests run WITHOUT loading a model: hand-built fake segments only."""

import json

from voxa import formats
from voxa.types import Info, Segment, TranscriptResult, Word


def _fake_result():
    return TranscriptResult(
        segments=[
            Segment(start=0.0, end=1.5, text=" Hello world.", words=None),
            Segment(
                start=1.5,
                end=3.25,
                text=" Second line",
                words=[Word(start=1.5, end=2.0, word=" Second")],
            ),
        ],
        info=Info(language="en", language_probability=0.99, duration=3.25),
    )


def test_to_txt_strips_and_joins():
    out = formats.to_txt(_fake_result())
    assert out == "Hello world.\nSecond line"


def test_to_json_roundtrips():
    out = formats.to_json(_fake_result())
    data = json.loads(out)
    assert data["info"]["language"] == "en"
    assert data["segments"][0]["text"] == " Hello world."
    assert data["segments"][1]["words"][0]["word"] == " Second"


def test_to_srt_format_and_timestamps():
    out = formats.to_srt(_fake_result())
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert lines[2] == "Hello world."
    # second cue index
    assert "2" in lines
    assert "00:00:01,500 --> 00:00:03,250" in out


def test_to_vtt_header_and_timestamps():
    out = formats.to_vtt(_fake_result())
    assert out.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.500" in out
    assert "00:00:01.500 --> 00:00:03.250" in out
    assert "Hello world." in out


def test_timestamp_hours_and_negative_clamp():
    assert formats._timestamp(3661.234, ",") == "01:01:01,234"
    assert formats._timestamp(-5.0, ".") == "00:00:00.000"


def test_empty_result_is_safe():
    empty = TranscriptResult(segments=[], info=None)
    assert formats.to_txt(empty) == ""
    assert formats.to_srt(empty) == ""
    assert formats.to_vtt(empty) == "WEBVTT\n"
