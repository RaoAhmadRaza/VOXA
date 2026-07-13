"""Voxa FastAPI app. One Transcriber, loaded once at startup, reused forever.

The real Transcriber import lives inside the lifespan so this module (and its
tests) can run without the model stack when a Transcriber is injected.
"""

import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from voxa import formats


class OutFormat(str, Enum):
    json = "json"
    txt = "txt"
    srt = "srt"
    vtt = "vtt"


# format -> (formats function, media type)
_DISPATCH = {
    OutFormat.json: (formats.to_json, "application/json; charset=utf-8"),
    OutFormat.txt: (formats.to_txt, "text/plain; charset=utf-8"),
    OutFormat.srt: (formats.to_srt, "application/x-subrip; charset=utf-8"),
    OutFormat.vtt: (formats.to_vtt, "text/vtt; charset=utf-8"),
}


def create_app(
    transcriber=None,
    *,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    web_dir: Optional[str] = None,
) -> FastAPI:
    """Build the app. If `transcriber` is given, it is reused and no model is
    loaded (used by tests). Otherwise one is built from args/env at startup.
    """
    model_size = model_size or os.getenv("VOXA_MODEL", "base")
    device = device or os.getenv("VOXA_DEVICE", "cpu")
    compute_type = compute_type or os.getenv("VOXA_COMPUTE_TYPE", "int8")
    if web_dir is None:
        web_dir = os.path.join(os.path.dirname(__file__), "web")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if transcriber is None:
            from voxa.core import Transcriber  # heavy import, only when needed

            app.state.transcriber = Transcriber(
                model_size, device=device, compute_type=compute_type
            )
        else:
            app.state.transcriber = transcriber
        app.state.model_size = model_size
        app.state.device = device
        yield

    app = FastAPI(title="Voxa", lifespan=lifespan)

    @app.get("/health")
    def health():
        # Cheap: reports config only, runs no inference.
        return {
            "status": "ok",
            "model": app.state.model_size,
            "device": app.state.device,
        }

    @app.post("/transcribe")
    def transcribe(
        file: Optional[UploadFile] = File(None),
        language: Optional[str] = Query(None),
        format: OutFormat = Query(OutFormat.json),
        word_timestamps: bool = Query(False),
        vad: bool = Query(True),
    ):
        if file is None or not file.filename:
            raise HTTPException(status_code=400, detail="no audio file provided")

        kwargs = {"vad_filter": vad}
        if language:
            kwargs["language"] = language

        try:
            result = app.state.transcriber.transcribe(
                file.file, word_timestamps=word_timestamps, **kwargs
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(
                status_code=400, detail=f"could not process audio: {exc}"
            )

        render, media_type = _DISPATCH[format]
        return Response(content=render(result), media_type=media_type)

    # Serve the UI from the same server. Dir may be empty until the UI phase;
    # mount only if it exists so startup never fails on a missing folder.
    if os.path.isdir(web_dir):
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app
