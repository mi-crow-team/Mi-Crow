from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.dependencies import get_settings
from server.routers import health, inference, layers, models
from server.routers import sae as sae_router
from server.routers import store as store_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Amber Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(layers.router)
    app.include_router(inference.router)
    app.include_router(sae_router.router)
    app.include_router(store_router.router)

    return app


app = create_app()
