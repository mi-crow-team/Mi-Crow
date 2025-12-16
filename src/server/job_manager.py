from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional


class JobManager:
    """Lightweight in-memory job manager with basic timeouts and idempotency."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._idempotency: Dict[str, str] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        job_type: str,
        func: Callable[[], Any],
        *,
        idempotency_key: str | None = None,
        timeout_sec: Optional[float] = None,
    ) -> str:
        with self._lock:
            if idempotency_key and idempotency_key in self._idempotency:
                return self._idempotency[idempotency_key]

            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "type": job_type,
                "status": "pending",
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
                "timeout": timeout_sec,
                "idempotency_key": idempotency_key,
                "progress": None,
                "logs": [],
                "cancel_requested": False,
            }
            if idempotency_key:
                self._idempotency[idempotency_key] = job_id

        def _run():
            start = time.time()
            self._update(job_id, status="running", started_at=start)
            try:
                result = func()
                self._update(job_id, status="completed", result=result, finished_at=time.time(), progress=1.0)
            except Exception as exc:  # pragma: no cover - defensive
                self._update(job_id, status="failed", error=str(exc), finished_at=time.time())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return job_id

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(kwargs)

    def status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"job_id '{job_id}' not found")
            job = dict(self._jobs[job_id])

        if job["status"] == "running" and job.get("timeout") and job.get("started_at"):
            elapsed = time.time() - job["started_at"]
            if elapsed > job["timeout"] and job["status"] != "timed_out":
                self._update(job_id, status="timed_out", finished_at=time.time())
                job["status"] = "timed_out"
        return job

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            logs = self._jobs[job_id].setdefault("logs", [])
            logs.append(message)

    def set_progress(self, job_id: str, progress: float | None) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]["progress"] = progress

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"job_id '{job_id}' not found")
            job = self._jobs[job_id]
            if job.get("status") in {"completed", "failed", "timed_out"}:
                return dict(job)
            job["cancel_requested"] = True
            if job["status"] == "pending":
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
            else:
                job["status"] = "cancel_requested"
        return dict(job)
