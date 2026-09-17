"""Serve the built frontend (frontend/dist) from the API process: one port, no CORS, no separate web server.

Only active when the build exists (production/`scripts/start.ps1`, Docker). In development Vite serves the UI.
API routes keep precedence; unknown /api paths still return the JSON 404. Any other path falls back to index.html
so client-side navigation keeps working.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

logger = logging.getLogger("voicelab.web")


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):  # noqa: ANN001
        key = path.replace("\\", "/")  # Starlette normalises with os.sep (backslashes on Windows)
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or key.startswith("api/") or key == "api":
                raise
            response = None
        if response is None or response.status_code == 404:
            return FileResponse(Path(str(self.directory)) / "index.html", headers={"Cache-Control": "no-cache"})
        if key.startswith("assets/"):  # content-hashed by Vite
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def mount_frontend(app: FastAPI, dist: Path | None) -> bool:
    if dist is None or not (dist / "index.html").is_file():
        return False
    app.mount("/", SPAStaticFiles(directory=dist, html=True), name="frontend")
    logger.info("frontend_mounted", extra={"dist": str(dist)})
    return True
