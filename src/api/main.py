from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.logging import logger
from src.api.routes import health, query, documents


def create_app() -> FastAPI:
    app = FastAPI(
        title="InsightDocs RAG API",
        version="1.0.0",
        description="Production-grade retrieval-augmented generation over your documents.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    @app.on_event("startup")
    def _startup():
        logger.info("InsightDocs API starting up")

    return app


app = create_app()


# --- serve the web UI (optional, enabled when frontend/ exists) ---
import os as _os
from fastapi.staticfiles import StaticFiles as _StaticFiles

_frontend_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "frontend")
if _os.path.isdir(_frontend_dir):
    app.mount("/app", _StaticFiles(directory=_frontend_dir, html=True), name="frontend")
