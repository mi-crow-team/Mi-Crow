from __future__ import annotations

from fastapi import APIRouter, Depends

from server.dependencies import get_hook_factory, get_job_manager

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/metrics")
def health_metrics(job_manager=Depends(get_job_manager)) -> dict[str, dict[str, int]]:
    counts = job_manager.get_job_counts()
    response_counts = {
        "total": counts.get("total", 0),
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "timed_out": counts.get("timed_out", 0),
    }
    return {"jobs": response_counts}


@router.get("/health/debug")
def health_debug(job_manager=Depends(get_job_manager)) -> dict:
    with job_manager._lock:
        jobs_data = {
            job_id: {
                "type": job.get("type"),
                "status": job.get("status"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
            }
            for job_id, job in job_manager._jobs.items()
        }
    counts = job_manager.get_job_counts()
    return {"job_counts": counts, "jobs": jobs_data, "total_jobs_in_dict": len(jobs_data)}


@router.get("/hooks")
def list_hooks(hook_factory=Depends(get_hook_factory)) -> dict[str, list[str]]:
    return {"available": hook_factory.available_hooks()}

