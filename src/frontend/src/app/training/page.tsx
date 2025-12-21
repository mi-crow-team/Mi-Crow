"use client";

import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { ActivationRunInfo, SaeRunInfo, TrainJobStatus } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Card, Input, Label, Row, SectionTitle, Spinner } from "@/components/ui";
import { StepCard, StepLayout } from "@/components/StepLayout";
import { RunHistorySidebar } from "@/components/RunHistorySidebar";
import { TrainingModal } from "@/components/TrainingModal";

type LatentMode = "n_latents" | "expansion_factor";

export default function TrainingPage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel, isLoading: isLoadingModel } = useModelLoader();
  const { data: runs } = useSWR<{ runs: ActivationRunInfo[] }>(
    modelId && modelLoaded ? `/sae/activations?model_id=${modelId}` : null,
    () => api.listActivations(modelId)
  );
  const { data: saes, mutate: refreshSaes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId ? `/sae/saes?model_id=${modelId}` : null,
    () => api.listSaes(modelId)
  );
  const { data: saeClasses } = useSWR<Record<string, string>>("/sae/classes", api.saeClasses);

  const [activationRun, setActivationRun] = useState("");
  const [layer, setLayer] = useState("");
  const [saeClass, setSaeClass] = useState("TopKSae");
  const [latentMode, setLatentMode] = useState<LatentMode>("n_latents");
  const [nLatents, setNLatents] = useState<number | undefined>();
  const [expansionFactor, setExpansionFactor] = useState<number | undefined>(1.0);
  const [hiddenDim, setHiddenDim] = useState<number | undefined>();
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Basic training config
  const [epochs, setEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(256);

  // Advanced training config (SaeTrainingConfig)
  const [lr, setLr] = useState(1e-3);
  const [l1Lambda, setL1Lambda] = useState(0.0);
  const [maxBatchesPerEpoch, setMaxBatchesPerEpoch] = useState<number | undefined>();
  const [verbose, setVerbose] = useState(false);
  const [useAmp, setUseAmp] = useState(true);
  const [gradAccumSteps, setGradAccumSteps] = useState(1);
  const [clipGrad, setClipGrad] = useState(1.0);
  const [monitoring, setMonitoring] = useState(1);
  const [memoryEfficient, setMemoryEfficient] = useState(false);

  // SAE kwargs
  const [saeK, setSaeK] = useState<number | undefined>();

  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<TrainJobStatus | null>(null);
  const [polling, setPolling] = useState<NodeJS.Timeout | null>(null);
  const [selectedSae, setSelectedSae] = useState<SaeRunInfo | null>(null);
  const [isLoadingLayerSize, setIsLoadingLayerSize] = useState(false);

  useEffect(() => {
    if (runs?.runs?.length && !activationRun) {
      const first = runs.runs[0];
      setActivationRun(first.run_id);
    }
  }, [runs, activationRun]);

  // Auto-set layer from selected activation run
  useEffect(() => {
    if (selectedRun?.layers?.length) {
      setLayer(selectedRun.layers[0]);
    } else {
      setLayer("");
    }
  }, [selectedRun]);

  useEffect(() => {
    return () => {
      if (polling) clearInterval(polling);
    };
  }, [polling]);

  const selectedRun = useMemo(
    () => runs?.runs?.find((r) => r.run_id === activationRun),
    [runs, activationRun]
  );

  // Fetch layer size when activation run and layer are selected
  useEffect(() => {
    if (selectedRun?.manifest_path && layer && modelLoaded) {
      setIsLoadingLayerSize(true);
      api
        .getLayerSize(selectedRun.manifest_path, layer)
        .then((info) => {
          setHiddenDim(info.hidden_dim);
          // Auto-calculate n_latents from expansion factor if in expansion mode
          if (latentMode === "expansion_factor" && expansionFactor && !nLatents) {
            setNLatents(Math.round(info.hidden_dim * expansionFactor));
          }
        })
        .catch((e) => {
          console.error("Failed to load layer size:", e);
          setHiddenDim(undefined);
        })
        .finally(() => {
          setIsLoadingLayerSize(false);
        });
    } else {
      setHiddenDim(undefined);
    }
  }, [selectedRun?.manifest_path, layer, modelLoaded, latentMode, expansionFactor]);

  // Update n_latents when expansion factor changes
  useEffect(() => {
    if (latentMode === "expansion_factor" && expansionFactor && hiddenDim) {
      setNLatents(Math.round(hiddenDim * expansionFactor));
    }
  }, [expansionFactor, hiddenDim, latentMode]);

  // Update expansion factor when n_latents changes (if in n_latents mode)
  useEffect(() => {
    if (latentMode === "n_latents" && nLatents && hiddenDim) {
      setExpansionFactor(nLatents / hiddenDim);
    }
  }, [nLatents, hiddenDim, latentMode]);

  const launch = async () => {
    if (!modelId || !activationRun || !layer || !modelLoaded) return;
    try {
      const hyperparams: Record<string, any> = {
        epochs,
        batch_size: batchSize,
        lr,
        l1_lambda: l1Lambda,
        verbose,
        use_amp: useAmp,
        grad_accum_steps: gradAccumSteps,
        clip_grad: clipGrad,
        monitoring,
        memory_efficient: memoryEfficient,
      };
      if (maxBatchesPerEpoch !== undefined) {
        hyperparams.max_batches_per_epoch = maxBatchesPerEpoch;
      }

      const sae_kwargs: Record<string, any> = {};
      if (saeK !== undefined) {
        sae_kwargs.k = saeK;
      }

      const payload = {
        model_id: modelId,
        activations_path: selectedRun?.manifest_path,
        layer,
        sae_class: saeClass,
        n_latents: nLatents,
        hyperparams,
        training_config: {},
        sae_kwargs,
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
                setHiddenDim(undefined);
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
              disabled={!modelId || modelLoaded || isLoadingModel}
            >
              {isLoadingModel ? (
                <span className="flex items-center gap-2">
                  <Spinner /> Loading...
                </span>
              ) : modelLoaded ? (
                "Model loaded"
              ) : (
                "Load model"
              )}
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
                onChange={(e) => {
                  setActivationRun(e.target.value);
                  setHiddenDim(undefined);
                }}
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
                disabled
                readOnly
                className="bg-slate-50 cursor-not-allowed"
              />
              {selectedRun?.layers && selectedRun.layers.length > 1 && (
                <p className="text-xs text-slate-500">
                  Note: This activation run has {selectedRun.layers.length} layers. Using first layer: {selectedRun.layers[0]}
                </p>
              )}
            </div>
          </Row>

          {/* Layer size display */}
          {layer && selectedRun?.manifest_path && (
            <Card className="bg-amber-50/50 border-amber-200 p-3">
              <div className="flex items-center gap-2 text-sm">
                {isLoadingLayerSize ? (
                  <>
                    <Spinner className="text-amber-600" />
                    <span className="text-slate-600">Loading layer size...</span>
                  </>
                ) : hiddenDim ? (
                  <>
                    <span className="text-slate-600">Layer size (hidden_dim):</span>
                    <span className="font-mono font-semibold text-slate-900">{hiddenDim.toLocaleString()}</span>
                  </>
                ) : (
                  <span className="text-slate-600">Select layer to see size</span>
                )}
              </div>
            </Card>
          )}

          <Row>
            <div className="space-y-1">
              <Label>SAE class</Label>
              <select
                className="input"
                value={saeClass}
                onChange={(e) => {
                  const newClass = e.target.value;
                  setSaeClass(newClass);
                  // Clear top-k if switching away from TopKSae
                  if (newClass !== "TopKSae") {
                    setSaeK(undefined);
                  }
                }}
              >
                {Object.keys(saeClasses || { TopKSae: "" }).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <Label>Latent size mode</Label>
              <select
                className="input"
                value={latentMode}
                onChange={(e) => setLatentMode(e.target.value as LatentMode)}
              >
                <option value="n_latents">n_latents (absolute)</option>
                <option value="expansion_factor">Expansion factor (relative)</option>
              </select>
            </div>
          </Row>

          {latentMode === "n_latents" ? (
            <Row>
              <div className="space-y-1">
                <Label>n_latents</Label>
                <Input
                  type="number"
                  value={nLatents ?? ""}
                  onChange={(e) => setNLatents(e.target.value ? Number(e.target.value) : undefined)}
                  placeholder={hiddenDim ? hiddenDim.toString() : ""}
                />
                {hiddenDim && nLatents && (
                  <p className="text-xs text-slate-500">
                    Expansion factor: {(nLatents / hiddenDim).toFixed(2)}x
                  </p>
                )}
              </div>
            </Row>
          ) : (
            <Row>
              <div className="space-y-1">
                <Label>Expansion factor</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={expansionFactor ?? ""}
                  onChange={(e) => setExpansionFactor(e.target.value ? Number(e.target.value) : undefined)}
                  placeholder="1.0"
                />
                {hiddenDim && expansionFactor && (
                  <p className="text-xs text-slate-500">
                    n_latents: {Math.round(hiddenDim * expansionFactor).toLocaleString()}
                  </p>
                )}
              </div>
            </Row>
          )}

          {/* SAE-specific kwargs */}
          {saeClass === "TopKSae" && (
            <Row>
              <div className="space-y-1">
                <Label>Top-K (k) *</Label>
                <Input
                  type="number"
                  value={saeK ?? ""}
                  onChange={(e) => setSaeK(e.target.value ? Number(e.target.value) : undefined)}
                  required
                  min={1}
                />
                <p className="text-xs text-slate-500">Number of top activations to keep (required for TopKSae)</p>
              </div>
            </Row>
          )}

          {/* Basic training config */}
          <div className="space-y-2">
            <SectionTitle>Basic Training Configuration</SectionTitle>
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
          </div>

          {/* Advanced configuration toggle */}
          <div className="space-y-2">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900"
            >
              <span>{showAdvanced ? "▼" : "▶"}</span>
              <span>Advanced Configuration</span>
            </button>

            {showAdvanced && (
              <Card className="bg-slate-50 border-slate-200 p-4 space-y-4">
                <Row>
                  <div className="space-y-1">
                    <Label>Learning rate (lr)</Label>
                    <Input
                      type="number"
                      step="1e-5"
                      value={lr}
                      onChange={(e) => setLr(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>L1 lambda (sparsity penalty)</Label>
                    <Input
                      type="number"
                      step="1e-5"
                      value={l1Lambda}
                      onChange={(e) => setL1Lambda(Number(e.target.value))}
                    />
                  </div>
                </Row>

                <Row>
                  <div className="space-y-1">
                    <Label>Max batches per epoch (optional)</Label>
                    <Input
                      type="number"
                      value={maxBatchesPerEpoch ?? ""}
                      onChange={(e) => setMaxBatchesPerEpoch(e.target.value ? Number(e.target.value) : undefined)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Gradient accumulation steps</Label>
                    <Input
                      type="number"
                      value={gradAccumSteps}
                      onChange={(e) => setGradAccumSteps(Number(e.target.value))}
                      min={1}
                    />
                  </div>
                </Row>

                <Row>
                  <div className="space-y-1">
                    <Label>Gradient clipping</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={clipGrad}
                      onChange={(e) => setClipGrad(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Monitoring level</Label>
                    <select
                      className="input"
                      value={monitoring}
                      onChange={(e) => setMonitoring(Number(e.target.value))}
                    >
                      <option value={0}>0 - Silent</option>
                      <option value={1}>1 - Basic</option>
                      <option value={2}>2 - Detailed</option>
                    </select>
                  </div>
                </Row>

                <div className="space-y-2">
                  <Label>Options</Label>
                  <div className="flex flex-col gap-2 text-sm">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={verbose}
                        onChange={(e) => setVerbose(e.target.checked)}
                        className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                      />
                      Verbose logging
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={useAmp}
                        onChange={(e) => setUseAmp(e.target.checked)}
                        className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                      />
                      Use AMP (Automatic Mixed Precision)
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={memoryEfficient}
                        onChange={(e) => setMemoryEfficient(e.target.checked)}
                        className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                      />
                      Memory efficient mode
                    </label>
                  </div>
                </div>
              </Card>
            )}
          </div>

          <div className="flex gap-2">
            <Button
              onClick={launch}
              disabled={!nLatents || (saeClass === "TopKSae" && !saeK)}
            >
              Launch training
            </Button>
            {["running", "pending"].includes(status?.status ?? "") && (
              <Button variant="ghost" onClick={cancel}>
                Cancel
              </Button>
            )}
          </div>
          {status && (
            <div className="text-sm space-y-2 p-3 bg-slate-50 rounded-md border border-slate-200">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-700">Status:</span>
                <span
                  className={`font-medium ${
                    status.status === "completed"
                      ? "text-green-600"
                      : status.status === "failed" || status.status === "error"
                      ? "text-red-600"
                      : status.status === "running" || status.status === "pending"
                      ? "text-amber-600"
                      : "text-slate-700"
                  }`}
                >
                  {status.status}
                </span>
                {["running", "pending"].includes(status.status ?? "") && <Spinner className="text-amber-600" />}
              </div>
              {status.progress !== undefined && status.progress !== null && (
                <div className="text-slate-700">
                  <span className="font-semibold">Progress:</span> {(status.progress * 100).toFixed(0)}%
                </div>
              )}
              {status.logs?.length ? (
                <div className="text-xs text-slate-600 space-y-1">
                  <div className="font-semibold">Logs (last 5):</div>
                  <ul className="list-disc pl-4 space-y-1">
                    {status.logs.slice(-5).map((l, i) => (
                      <li key={i} className="text-slate-700">
                        {l}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {status.sae_id && (
                <div className="text-slate-700">
                  <span className="font-semibold">SAE ID:</span> {status.sae_id}
                </div>
              )}
              {status.sae_path && (
                <div className="text-slate-700">
                  <span className="font-semibold">Path:</span> {status.sae_path}
                </div>
              )}
              {status.error && (
                <div className="text-red-600 font-medium bg-red-50 p-2 rounded border border-red-200">
                  Error: {status.error}
                </div>
              )}
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
        <button
          type="button"
          onClick={() => setSelectedSae(s)}
          className="w-full text-left hover:opacity-80 transition"
        >
          <div className="space-y-1 text-xs">
            <div className="flex items-center justify-between mb-1">
              <div className="font-semibold text-slate-900 truncate">{s.sae_id}</div>
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700">
                ✓ Trained
              </span>
            </div>
            <div className="text-slate-700">
              <span className="text-slate-600">Layer:</span> <span className="text-slate-900">{s.layer ?? "-"}</span>
            </div>
            <div className="text-slate-700">
              <span className="text-slate-600">Class:</span> <span className="text-slate-900">{s.sae_class ?? "SAE"}</span>
            </div>
            {s.sae_path && (
              <div className="text-slate-700 truncate">
                <span className="text-slate-600">Path:</span>{" "}
                <span className="text-slate-900 font-mono text-xs">{s.sae_path}</span>
              </div>
            )}
          </div>
        </button>
      )}
    />
  );

  return (
    <>
      <StepLayout
        title="Train SAE"
        description="Use saved activations to train sparse autoencoders on selected layers."
        steps={steps}
        sidebar={sidebar}
      />
      {selectedSae && <TrainingModal sae={selectedSae} onClose={() => setSelectedSae(null)} />}
    </>
  );
}
