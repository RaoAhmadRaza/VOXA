"""API tests with a mocked Transcriber — no model load, CI stays CPU-fast."""

from fastapi.testclient import TestClient

from voxa.api import create_app
from voxa.types import Info, Segment, TranscriptResult


class FakeTranscriber:
    """Stand-in for voxa.core.Transcriber: returns a fixed result."""

    def transcribe(self, audio, *, word_timestamps=False, **kwargs):
        words = None
        return TranscriptResult(
            segments=[Segment(start=0.0, end=1.5, text=" Hi there.", words=words)],
            info=Info(language="en", language_probability=0.99, duration=1.5),
        )


def _client():
    app = create_app(FakeTranscriber(), model_size="tiny", device="cpu")
    return TestClient(app)


def test_health_reports_model_and_device():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "model": "tiny", "device": "cpu"}


def test_transcribe_json_default():
    with _client() as client:
        r = client.post(
            "/transcribe",
            files={"file": ("a.wav", b"fakeaudio", "audio/wav")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["info"]["language"] == "en"
        assert body["segments"][0]["text"] == " Hi there."


def test_transcribe_srt_content_type_and_body():
    with _client() as client:
        r = client.post(
            "/transcribe?format=srt",
            files={"file": ("a.wav", b"fakeaudio", "audio/wav")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-subrip")
        assert "00:00:00,000 --> 00:00:01,500" in r.text
        assert "Hi there." in r.text


def test_missing_file_is_400():
    with _client() as client:
        r = client.post("/transcribe")
        assert r.status_code == 400


def test_bad_format_is_422():
    with _client() as client:
        r = client.post(
            "/transcribe?format=bogus",
            files={"file": ("a.wav", b"fakeaudio", "audio/wav")},
        )
        assert r.status_code == 422
