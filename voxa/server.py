"""uvicorn entrypoint. `python -m voxa.server` boots the API.

Host/port configurable via VOXA_HOST / VOXA_PORT, default 127.0.0.1:8000.
"""

import os

import uvicorn

from voxa.api import create_app

app = create_app()


def main():
    uvicorn.run(
        app,
        host=os.getenv("VOXA_HOST", "127.0.0.1"),
        port=int(os.getenv("VOXA_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
