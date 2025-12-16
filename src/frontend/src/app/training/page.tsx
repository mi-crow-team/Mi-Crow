"use client";

import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { ActivationRunInfo, SaeRunInfo, TrainJobStatus } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Card, Input, Label, Row, SectionTitle } from "@/components/ui";
import { StepCard, StepLayout } from "@/components/StepLayout";
import { RunHistorySidebar } from "@/components/RunHistorySidebar";

export default function TrainingPage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel } = useModelLoader();
  const { data: runs } = useSWR<{ runs: ActivationRunInfo[] }>(
    modelId && modelLoaded ? `/sae/activations?model_id=${modelId}` : null,
    () => api.listActivations(modelId)
  );
  const { data: saes, mutate: refreshSaes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId && modelLoaded ? `/sae/saes?model_id=${modelId}` : null,
    () => api.listSaes(modelId)
  );
  const { data: saeClasses } = useSWR<Record<string, string>>("/sae/classes", api.saeClasses);

  const [activationRun, setActivationRun] = useState("");
  const [layer, setLayer] = useState("");
  const [saeClass, setSaeClass] = useState("TopKSae");
  const [nLatents, setNLatents] = useState<number | undefined>();
  const [epochs, setEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(256);
  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<TrainJobStatus | null>(null);
  const [polling, setPolling] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (runs?.runs?.length && !activationRun) {
      const first = runs.runs[0];
      setActivationRun(first.run_id);
      setLayer(first.layers[0] ?? "");
    }
  }, [runs, activationRun]);

  useEffect(() => {
    return () => {
      if (polling) clearInterval(polling);
    };
  }, [polling]);

  const selectedRun = useMemo(
    () => runs?.runs?.find((r) => r.run_id === activationRun),
    [runs, activationRun]
  );

  const launch = async () => {
    if (!modelId || !activationRun || !layer || !modelLoaded) return;
    try {
      const payload = {
        model_id: modelId,
        activations_path: selectedRun?.manifest_path,
        layer,
        sae_class: saeClass,
        n_latents: nLatents,
        hyperparams: { epochs, batch_size: batchSize },
        run_id: activationRun,
      };
      const res = await api.train(payload);
      setJobId(res.job_id);
      setStatus({ job_id: res.job_id, status: res.status } as TrainJobStatus);
      if (polling) clearInterval(polling);
      const t = setInterval(checkStatus, 2500);
      setPolling(t);
    } catch (e: any) {
      setStatus({ job_id: "", status: "error", error: e.message } as TrainJobStatus);
    }
  };

  const checkStatus = async () => {
    if (!jobId) return;
    const res = await api.trainStatus(jobId);
    setStatus(res);
    if (["completed", "failed", "timed_out", "cancelled"].includes(res.status)) {
      if (polling) clearInterval(polling);
      refreshSaes();
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    const res = await api.cancelTrain(jobId);
    setStatus(res);
  };

  const steps = (
    <>
      <StepCard
        step={1}
        title="Load model"
        description="Choose which model to use for training SAEs. This must be loaded before selecting runs."
      >
        <Row>
          <div className="space-y-1">
            <Label>Model</Label>
            <select
              className="input"
              value={modelId}
              onChange={(e) => {
                setModelId(e.target.value);
                setModelLoaded(false);
                setActivationRun("");
              }}
            >
              {models?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.id}]
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label>&nbsp;</Label>
            <Button
              onClick={async () => {
                if (!modelId) return;
                await loadModel();
              }}
              disabled={!modelId || modelLoaded}
            >
              {modelLoaded ? "Model loaded" : "Load model"}
            </Button>
          </div>
        </Row>
      </StepCard>

      <StepCard
        step={2}
        title="Configure training and launch"
        description="Pick an activation run, layer and SAE class, set hyperparameters, and start the training job."
      >
        <div className={!modelLoaded ? "opacity-50 pointer-events-none space-y-4" : "space-y-4"}>
          <Row>
            <div className="space-y-1">
              <Label>Activation run</Label>
              <select
                className="input"
                value={activationRun}
                onChange={(e) => setActivationRun(e.target.value)}
              >
                {runs?.runs?.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} ({r.samples ?? "-"} samples)
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <Label>Layer</Label>
              <Input
                value={layer}
                onChange={(e) => setLayer(e.target.value)}
                placeholder={selectedRun?.layers?.[0] ?? ""}
              />
            </div>
          </Row>

          <Row>
            <div className="space-y-1">
              <Label>SAE class</Label>
              <select className="input" value={saeClass} onChange={(e) => setSaeClass(e.target.value)}>
                {Object.keys(saeClasses || { TopKSae: "" }).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <Label>n_latents (optional)</Label>
              <Input
                type="number"
                value={nLatents ?? ""}
                onChange={(e) => setNLatents(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>
          </Row>

          <Row>
            <div className="space-y-1">
              <Label>Epochs</Label>
              <Input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} min={1} />
            </div>
            <div className="space-y-1">
              <Label>Batch size</Label>
              <Input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} min={1} />
            </div>
          </Row>

          <div className="flex gap-2">
            <Button onClick={launch}>
              Launch training
            </Button>
            {["running", "pending"].includes(status?.status ?? "") && (
              <Button variant="ghost" onClick={cancel}>
                Cancel
              </Button>
            )}
          </div>
          {status && (
            <div className="text-sm text-slate-200 space-y-1">
              <div>Status: {status.status}</div>
              {status.progress !== undefined && status.progress !== null && (
                <div>Progress: {(status.progress * 100).toFixed(0)}%</div>
              )}
              {status.logs?.length ? (
                <div className="text-xs text-slate-400 space-y-1">
                  <div>Logs (last 5):</div>
                  <ul className="list-disc pl-4">
                    {status.logs.slice(-5).map((l, i) => (
                      <li key={i}>{l}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {status.sae_id && <div>SAE ID: {status.sae_id}</div>}
              {status.sae_path && <div>Path: {status.sae_path}</div>}
              {status.error && <div className="text-rose-400">Error: {status.error}</div>}
            </div>
          )}
        </div>
      </StepCard>
    </>
  );

  const sidebar = (
    <RunHistorySidebar
      title="Trained SAEs"
      items={saes?.saes ?? []}
      emptyMessage="No SAEs trained yet for this model."
      renderItem={(s: SaeRunInfo) => (
        <div className="space-y-1 text-xs">
          <div className="font-semibold text-slate-100 truncate">{s.sae_id}</div>
          <div className="text-slate-400">Layer: {s.layer ?? "-"}</div>
          <div className="text-slate-400">Class: {s.sae_class ?? "SAE"}</div>
          {s.sae_path && <div className="text-slate-500 truncate">path: {s.sae_path}</div>}
        </div>
      )}
    />
  );

  return (
    <StepLayout
      title="Train SAE"
      description="Use saved activations to train sparse autoencoders on selected layers."
      steps={steps}
      sidebar={sidebar}
    />
  );
}

