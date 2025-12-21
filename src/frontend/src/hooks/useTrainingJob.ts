import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import type { TrainJobStatus } from "@/lib/types";

export function useTrainingJob() {
  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<TrainJobStatus | null>(null);
  const [polling, setPolling] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (polling) clearInterval(polling);
    };
  }, [polling]);

  const checkStatus = useCallback(async () => {
    if (!jobId) return;
    const res = await api.trainStatus(jobId);
    setStatus(res);
    if (["completed", "failed", "timed_out", "cancelled"].includes(res.status)) {
      if (polling) clearInterval(polling);
      setPolling(null);
      return true; // Job finished
    }
    return false; // Job still running
  }, [jobId, polling]);

  const startPolling = useCallback(() => {
    if (polling) clearInterval(polling);
    const t = setInterval(checkStatus, 2500);
    setPolling(t);
  }, [polling, checkStatus]);

  const cancel = useCallback(async () => {
    if (!jobId) return;
    const res = await api.cancelTrain(jobId);
    setStatus(res);
    if (polling) clearInterval(polling);
    setPolling(null);
  }, [jobId, polling]);

  const reset = useCallback(() => {
    setJobId("");
    setStatus(null);
    if (polling) clearInterval(polling);
    setPolling(null);
  }, [polling]);

  return {
    jobId,
    status,
    setJobId,
    setStatus,
    checkStatus,
    startPolling,
    cancel,
    reset,
  };
}

