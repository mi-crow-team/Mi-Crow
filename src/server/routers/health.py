from __future__ import annotations

from fastapi import APIRouter, Depends

from server.dependencies import get_hook_factory, get_job_manager

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}


@router.get("/health/metrics")
def health_metrics(job_manager=Depends(get_job_manager)) -> dict[str, dict[str, int]]:
    """Health check with job metrics."""
    counts = {"total": 0, "pending": 0, "running": 0, "completed": 0, "failed": 0, "timed_out": 0}
    try:
        # Access jobs through public interface if available, otherwise use internal
        jobs = getattr(job_manager, "_jobs", {})
        for job in jobs.values():
            counts["total"] += 1
            status = job.get("status", "unknown")
            if status in counts:
                counts[status] += 1
    except Exception:
        pass
    return {"jobs": counts}


@router.get("/hooks")
def list_hooks(hook_factory=Depends(get_hook_factory)) -> dict[str, list[str]]:
    """List available hooks."""
    return {"available": hook_factory.available_hooks()}

