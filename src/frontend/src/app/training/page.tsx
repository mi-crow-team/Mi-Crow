"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { ActivationRunInfo, SaeRunInfo, StoreInfo, TrainJobStatus } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { useTrainingState, type LatentMode } from "@/hooks/useTrainingState";
import { useTrainingJob } from "@/hooks/useTrainingJob";
import { Button, Card, Input, Label, Row, SectionTitle, Spinner } from "@/components/ui";
import { StepCard, StepLayout } from "@/components/StepLayout";
import { RunHistorySidebar } from "@/components/RunHistorySidebar";
import { TrainingModal } from "@/components/TrainingModal";

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
  const { data: storeInfo } = useSWR<StoreInfo>("/store/info", api.storeInfo);

  const trainingState = useTrainingState(runs, modelLoaded);
  const trainingJob = useTrainingJob();
  const [selectedSae, setSelectedSae] = useState<SaeRunInfo | null>(null);
  const [isLoadingLayerSize, setIsLoadingLayerSize] = useState(false);
  const [pendingTrainingJobs, setPendingTrainingJobs] = useState<Array<{ job_id: string; status: string; created_at: string }>>([]);

  // Destructure for easier access
  const {
    activationRun, setActivationRun, layer, setLayer, saeClass, setSaeClass,
    latentMode, setLatentMode, nLatents, expansionFactor,
    hiddenDim, showAdvanced, setShowAdvanced, epochs, setEpochs, batchSize, setBatchSize,
    lr, setLr, l1Lambda, setL1Lambda, maxBatchesPerEpoch, setMaxBatchesPerEpoch,
    verbose, setVerbose, useAmp, setUseAmp, gradAccumSteps, setGradAccumSteps,
    clipGrad, setClipGrad, monitoring, setMonitoring, memoryEfficient, setMemoryEfficient,
    saeK, setSaeK, useWandb, setUseWandb, wandbProject, setWandbProject,
    wandbEntity, setWandbEntity, wandbName, setWandbName,
    setNLatents, setExpansionFactor, setHiddenDim
  } = trainingState;

  const { jobId, status, setJobId, setStatus, checkStatus, startPolling, cancel } = trainingJob;

  const selectedRun = useMemo(
    () => runs?.runs?.find((r) => r.run_id === activationRun),
    [runs, activationRun]
  );

  useEffect(() => {
    if (activationRun && !wandbName) {
      setWandbName(activationRun);
    }
  }, [activationRun, wandbName, setWandbName]);

  useEffect(() => {
    if (storeInfo?.wandb_project) {
      if (wandbProject === "" || wandbProject === storeInfo.wandb_project) {
        setWandbProject(storeInfo.wandb_project);
      }
    }
  }, [storeInfo?.wandb_project]);

  useEffect(() => {
    if (storeInfo?.wandb_entity) {
      if (wandbEntity === "" || wandbEntity === storeInfo.wandb_entity) {
        setWandbEntity(storeInfo.wandb_entity);
      }
    }
  }, [storeInfo?.wandb_entity]);

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
    // Validate required fields and provide user feedback
    if (!modelId) {
      setStatus({ job_id: "", status: "error", error: "Please select a model" } as TrainJobStatus);
      return;
    }
    if (!modelLoaded) {
      setStatus({ job_id: "", status: "error", error: "Please load the model first" } as TrainJobStatus);
      return;
    }
    if (!activationRun) {
      setStatus({ job_id: "", status: "error", error: "Please select an activation run" } as TrainJobStatus);
      return;
    }
    if (!layer) {
      setStatus({ job_id: "", status: "error", error: "Please select a layer" } as TrainJobStatus);
      return;
    }
    if (!selectedRun?.manifest_path) {
      setStatus({ job_id: "", status: "error", error: "Activation run manifest path not found" } as TrainJobStatus);
      return;
    }
    if (!nLatents) {
      setStatus({ job_id: "", status: "error", error: "Please set n_latents or expansion factor" } as TrainJobStatus);
      return;
    }
    if (saeClass === "TopKSae" && !saeK) {
      setStatus({ job_id: "", status: "error", error: "TopKSae requires k parameter" } as TrainJobStatus);
      return;
    }

    try {
      setStatus({ job_id: "", status: "pending", error: undefined } as TrainJobStatus);
      const hyperparams: Record<string, unknown> = {
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
        use_wandb: useWandb,
      };
      if (maxBatchesPerEpoch !== undefined) {
        hyperparams.max_batches_per_epoch = maxBatchesPerEpoch;
      }
      if (useWandb) {
        if (wandbProject) {
          hyperparams.wandb_project = wandbProject;
        }
        if (wandbEntity) {
          hyperparams.wandb_entity = wandbEntity;
        }
        if (wandbName) {
          hyperparams.wandb_name = wandbName;
        }
      }

      const sae_kwargs: Record<string, unknown> = {};
      if (saeK !== undefined) {
        sae_kwargs.k = saeK;
      }

      const payload = {
        model_id: modelId,
        activations_path: selectedRun.manifest_path,
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
      const initialStatus = { job_id: res.job_id, status: res.status } as typeof status;
      setStatus(initialStatus);
      // Add to pending jobs immediately
      setPendingTrainingJobs((prev) => [
        { job_id: res.job_id, status: res.status, created_at: new Date().toISOString() },
        ...prev,
      ]);
      startPolling();
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : String(e);
      console.error("Training launch error:", e);
      setStatus({ job_id: "", status: "error", error } as TrainJobStatus);
    }
  };

  // Update pending jobs when status changes
  useEffect(() => {
    if (jobId && status) {
      setPendingTrainingJobs((prev) =>
        prev.map((j) => (j.job_id === jobId ? { ...j, status: status.status } : j))
      );
      // Remove from pending jobs when completed
      if (["completed", "failed", "timed_out", "cancelled"].includes(status.status)) {
        refreshSaes();
        setPendingTrainingJobs((prev) => prev.filter((j) => j.job_id !== jobId));
      }
    }
  }, [jobId, status, refreshSaes]);

  const handleDeleteSae = async (sae: SaeRunInfo) => {
    if (!modelId || !sae.sae_id) return;
    if (!confirm(`Are you sure you want to delete SAE ${sae.sae_id}?`)) return;
    try {
      await api.deleteSae(modelId, sae.sae_id);
      refreshSaes();
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : String(e);
      alert(`Failed to delete SAE: ${error}`);
    }
  };

  const steps = (
    <>
      <StepCard
        step={0}
        title="Configure Weights & Biases (wandb)"
        description="Optionally enable wandb logging to track your training experiments. Configure project, entity, and run name."
      >
        <div className="space-y-4">
          <Card className="bg-slate-50 border-slate-200 p-4 space-y-4">
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useWandb}
                  onChange={(e) => setUseWandb(e.target.checked)}
                  className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
                />
                <span className="text-sm font-medium text-slate-700">Enable wandb logging</span>
              </label>
            </div>
            {useWandb && (
              <div className="space-y-3 pl-6 border-l-2 border-mi_crow-200">
                <Row>
                  <div className="space-y-1">
                    <Label>Project name</Label>
                    <Input
                      value={wandbProject}
                      onChange={(e) => setWandbProject(e.target.value)}
                      placeholder={storeInfo?.wandb_project || "(uses env var or default: sae-training)"}
                    />
                    {storeInfo?.wandb_project && (
                      <p className="text-xs text-slate-500">
                        {wandbProject === storeInfo.wandb_project ? (
                          <>
                            <span className="text-mi_crow-600 font-medium">✓ Using value from environment:</span>{" "}
                            <span className="font-mono">{storeInfo.wandb_project}</span>
                          </>
                        ) : (
                          <>
                            Default from env: <span className="font-mono">{storeInfo.wandb_project}</span>
                            {wandbProject && " (overridden)"}
                          </>
                        )}
                      </p>
                    )}
                  </div>
                  <div className="space-y-1">
                    <Label>Entity (team/username)</Label>
                    <Input
                      value={wandbEntity || storeInfo?.wandb_entity || ""}
                      onChange={(e) => setWandbEntity(e.target.value)}
                      placeholder={storeInfo?.wandb_entity ? `(env: ${storeInfo.wandb_entity})` : "your-wandb-entity"}
                    />
                    {storeInfo?.wandb_entity && (
                      <p className="text-xs text-slate-500">
                        {wandbEntity === storeInfo.wandb_entity ? (
                          <>
                            <span className="text-mi_crow-600 font-medium">✓ Using value from environment:</span>{" "}
                            <span className="font-mono">{storeInfo.wandb_entity}</span>
                          </>
                        ) : (
                          <>
                            Default from env: <span className="font-mono">{storeInfo.wandb_entity}</span>
                            {wandbEntity && " (overridden)"}
                          </>
                        )}
                      </p>
                    )}
                  </div>
                </Row>
                <Row>
                  <div className="space-y-1">
                    <Label>Run name</Label>
                    <Input
                      value={wandbName}
                      onChange={(e) => setWandbName(e.target.value)}
                      placeholder={activationRun || "run-name"}
                    />
                    <p className="text-xs text-slate-500">Defaults to activation run ID if empty</p>
                  </div>
                </Row>
              </div>
            )}
          </Card>
        </div>
      </StepCard>

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
            <Card className="bg-mi_crow-50/50 border-mi_crow-200 p-3">
              <div className="flex items-center gap-2 text-sm">
                {isLoadingLayerSize ? (
                  <>
                    <Spinner className="text-mi_crow-600" />
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
                        className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
                      />
                      Verbose logging
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={useAmp}
                        onChange={(e) => setUseAmp(e.target.checked)}
                        className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
                      />
                      Use AMP (Automatic Mixed Precision)
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={memoryEfficient}
                        onChange={(e) => setMemoryEfficient(e.target.checked)}
                        className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
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
              disabled={
                !modelId ||
                !modelLoaded ||
                !activationRun ||
                !layer ||
                !selectedRun?.manifest_path ||
                !nLatents ||
                (saeClass === "TopKSae" && !saeK)
              }
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
                      ? "text-mi_crow-600"
                      : "text-slate-700"
                  }`}
                >
                  {status.status}
                </span>
                {["running", "pending"].includes(status.status ?? "") && <Spinner className="text-mi_crow-600" />}
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

  // Combine pending jobs and completed SAEs for display
  const allTrainingItems = useMemo(() => {
    const pending = pendingTrainingJobs.map((job) => ({
      type: "pending" as const,
      job_id: job.job_id,
      status: job.status,
      created_at: job.created_at,
    }));
    const completed = (saes?.saes ?? []).map((sae) => ({
      type: "completed" as const,
      sae,
    }));
    return [...pending, ...completed];
  }, [pendingTrainingJobs, saes]);

  const sidebar = (
    <RunHistorySidebar
      title="Training History"
      items={allTrainingItems}
      emptyMessage="No training runs yet for this model."
      getItemKey={(item, idx) => item.type === "pending" ? item.job_id : item.sae.sae_id}
      onDelete={(item) => {
        if (item.type === "completed") {
          handleDeleteSae(item.sae);
        } else {
          // For pending jobs, just remove from list (they'll be cleaned up when status is checked)
          setPendingTrainingJobs((prev) => prev.filter((j) => j.job_id !== item.job_id));
        }
      }}
      renderItem={(item, idx) => {
        if (item.type === "pending") {
          return (
            <button
              type="button"
              onClick={() => {
                setJobId(item.job_id);
                setStatus({ job_id: item.job_id, status: item.status } as typeof status);
                startPolling();
              }}
              className="w-full text-left hover:opacity-80 transition"
            >
              <div className="space-y-1 text-xs">
                <div className="font-semibold text-slate-900 truncate mb-1">
                  {item.job_id}
                </div>
                <div className="text-slate-700">
                  <span className="text-slate-600">Started:</span>{" "}
                  <span className="text-slate-900">
                    {new Date(item.created_at).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-end mt-4">
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-mi_crow-100 text-mi_crow-700">
                    ⏳ {item.status === "pending" ? "Pending" : "Running"}
                  </span>
                </div>
              </div>
            </button>
          );
        } else {
          const s = item.sae;
          return (
            <button
              type="button"
              onClick={() => setSelectedSae(s)}
              className="w-full text-left hover:opacity-80 transition relative"
            >
              <div className="space-y-1 text-xs">
                <div className="font-semibold text-slate-900 truncate mb-1">
                  {s.sae_id}
                </div>
                <div className="text-slate-700">
                  <span className="text-slate-600">Layer:</span>{" "}
                  <span className="text-slate-900 truncate block" title={s.layer ?? "-"}>
                    {s.layer ?? "-"}
                  </span>
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
                <div className="flex justify-end mt-4">
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700">
                    ✓ Trained
                  </span>
                </div>
              </div>
            </button>
          );
        }
      }}
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
