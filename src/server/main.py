from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.dependencies import get_hook_factory, get_settings, get_job_manager
from server.routers import inference, layers, models
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

    app.include_router(models.router)
    app.include_router(layers.router)
    app.include_router(inference.router)
    app.include_router(sae_router.router)
    app.include_router(store_router.router)

    @app.get("/hooks", tags=["hooks"])
    def list_hooks() -> dict:
        hook_factory = get_hook_factory()
        return {"available": hook_factory.available_hooks()}

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/health/metrics", tags=["health"])
    def health_metrics() -> dict:
        jm = get_job_manager()
        counts = {"total": 0, "pending": 0, "running": 0, "completed": 0, "failed": 0, "timed_out": 0}
        try:
            for job in jm._jobs.values():  # type: ignore[attr-defined]
                counts["total"] += 1
                status = job.get("status", "unknown")
                if status in counts:
                    counts[status] += 1
        except Exception:
            pass
        return {"jobs": counts}

    return app


app = create_app()
