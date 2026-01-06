"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { ActivationRunInfo, SaeRunInfo } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Card, Input, Label, Row, SectionTitle } from "@/components/ui";

export default function OpsPage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel } = useModelLoader();
  const { data: runs, mutate: refreshRuns } = useSWR<{ runs: ActivationRunInfo[] }>(
    modelId && modelLoaded ? `/sae/activations?model_id=${modelId}` : null,
    () => api.listActivations(modelId)
  );
  const { data: saes, mutate: refreshSaes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId && modelLoaded ? `/sae/saes?model_id=${modelId}` : null,
    () => api.listSaes(modelId)
  );
  const { data: health, mutate: refreshHealth } = useSWR<{ jobs: Record<string, number> }>("/health/metrics", api.health);
  const [apiKey, setApiKey] = useState("");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("apiKey") : "";
    if (saved) setApiKey(saved);
  }, []);

  useEffect(() => {
    if (models && models.length && !modelId) {
      setModelId(models[0].id);
      setModelLoaded(false);
    }
  }, [models, modelId]);

  const storeKey = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("apiKey", apiKey);
      setMsg("API key saved to localStorage (used for delete).");
    }
  };

  const delRun = async (run_id: string) => {
    try {
      await api.deleteActivation(modelId, run_id);
      setMsg(`Deleted activation run ${run_id}`);
      refreshRuns();
    } catch (e: any) {
      setMsg(`Error: ${e.message}`);
    }
  };

  const delSae = async (sae_id: string) => {
    try {
      await api.deleteSae(modelId, sae_id);
      setMsg(`Deleted SAE ${sae_id}`);
      refreshSaes();
    } catch (e: any) {
      setMsg(`Error: ${e.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <SectionTitle>Ops & Cleanup</SectionTitle>
      <Card className="space-y-3">
        <Row>
          <div className="space-y-1">
            <Label>Model</Label>
            <select
              className="input"
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value);
                setModelLoaded(false);
              }}
            >
              {models?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label>&nbsp;</Label>
            <Button
              variant="ghost"
              onClick={async () => {
                if (!modelId) return;
                setMsg("Loading model...");
                try {
                  await loadModel();
                  setMsg("Model loaded");
                } catch (e: any) {
                  setMsg(`Error loading model: ${e.message}`);
                }
              }}
              disabled={!modelId || modelLoaded}
            >
              {modelLoaded ? "Model loaded" : "Load model"}
            </Button>
          </div>
          <div className="space-y-1">
            <Label>API key (optional for delete)</Label>
            <Input value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <Button variant="ghost" onClick={storeKey}>
              Save key locally
            </Button>
          </div>
        </Row>
        {msg && <p className="text-sm text-slate-300">{msg}</p>}
      </Card>

      <SectionTitle>Health</SectionTitle>
      <Card className="space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-300">Server health metrics and status</p>
          <Button variant="ghost" onClick={() => refreshHealth()}>
            Refresh
          </Button>
        </div>
        {health ? (
          <div className="space-y-4">
            {health.jobs && (
              <div>
                <h3 className="text-sm font-semibold text-slate-200 mb-3">Job Status</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
                    <div className="text-xs text-slate-400 mb-1">Total Jobs</div>
                    <div className="text-lg font-semibold text-slate-100">
                      {health.jobs.total || 0}
                    </div>
                  </div>
                  <div className="bg-mi_crow-900/30 rounded-lg p-3 border border-mi_crow-700">
                    <div className="text-xs text-mi_crow-300 mb-1">Pending</div>
                    <div className="text-lg font-semibold text-mi_crow-200">
                      {health.jobs.pending || 0}
                    </div>
                  </div>
                  <div className="bg-blue-900/30 rounded-lg p-3 border border-blue-700">
                    <div className="text-xs text-blue-300 mb-1">Running</div>
                    <div className="text-lg font-semibold text-blue-200">
                      {health.jobs.running || 0}
                    </div>
                  </div>
                  <div className="bg-green-900/30 rounded-lg p-3 border border-green-700">
                    <div className="text-xs text-green-300 mb-1">Completed</div>
                    <div className="text-lg font-semibold text-green-200">
                      {health.jobs.completed || 0}
                    </div>
                  </div>
                  <div className="bg-red-900/30 rounded-lg p-3 border border-red-700">
                    <div className="text-xs text-red-300 mb-1">Failed</div>
                    <div className="text-lg font-semibold text-red-200">
                      {health.jobs.failed || 0}
                    </div>
                  </div>
                  <div className="bg-orange-900/30 rounded-lg p-3 border border-orange-700">
                    <div className="text-xs text-orange-300 mb-1">Timed Out</div>
                    <div className="text-lg font-semibold text-orange-200">
                      {health.jobs.timed_out || 0}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-sm text-slate-400">Loading health metrics...</p>
        )}
      </Card>

      <SectionTitle>Activation runs</SectionTitle>
      <div className={!modelLoaded ? "opacity-50 pointer-events-none" : ""}>
      <Card className="space-y-2 text-sm text-slate-200">
        {runs?.runs?.length ? (
          runs.runs.map((r) => (
            <div key={r.run_id} className="flex items-center justify-between border border-slate-800 rounded-md p-2">
              <div>
                <div className="font-semibold">{r.run_id}</div>
                <div className="text-slate-500 text-xs">layers: {r.layers.join(", ")}</div>
              </div>
              <Button variant="ghost" onClick={() => delRun(r.run_id)}>
                Delete
              </Button>
            </div>
          ))
        ) : (
          <div className="text-slate-500">None</div>
        )}
      </Card>
      </div>

      <SectionTitle>SAEs</SectionTitle>
      <div className={!modelLoaded ? "opacity-50 pointer-events-none" : ""}>
      <Card className="space-y-2 text-sm text-slate-200">
        {saes?.saes?.length ? (
          saes.saes.map((s) => (
            <div key={s.sae_id} className="flex items-center justify-between border border-slate-800 rounded-md p-2">
              <div>
                <div className="font-semibold">{s.sae_id}</div>
                <div className="text-slate-500 text-xs">layer: {s.layer ?? "-"}</div>
              </div>
              <Button variant="ghost" onClick={() => delSae(s.sae_id)}>
                Delete
              </Button>
            </div>
          ))
        ) : (
          <div className="text-slate-500">None</div>
        )}
      </Card>
      </div>
    </div>
  );
}

