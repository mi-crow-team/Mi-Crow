"use client";

import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { ActivationRunInfo, LayerInfo, StoreInfo } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Input, Label, Modal, Row } from "@/components/ui";
import { StepCard, StepLayout } from "@/components/StepLayout";
import { RunHistorySidebar } from "@/components/RunHistorySidebar";

type DatasetMode = "hf" | "local";

export default function ActivationsPage() {
  const { models, modelId: selectedModel, setModelId: setSelectedModel, modelLoaded, setModelLoaded, loadModel } =
    useModelLoader();
  const { data: layers } = useSWR<LayerInfo[]>(
    selectedModel && modelLoaded ? `/models/${selectedModel}/layers` : null,
    () => api.layers(selectedModel)
  );
  const { data: runs, mutate: refreshRuns } = useSWR<{ runs: ActivationRunInfo[] }>(
    selectedModel && modelLoaded ? `/sae/activations?model_id=${selectedModel}` : null,
    () => api.listActivations(selectedModel)
  );
  const { data: storeInfo } = useSWR<StoreInfo>("/store/info", api.storeInfo);

  const [datasetMode, setDatasetMode] = useState<DatasetMode>("local");
  const [hfName, setHfName] = useState("ag_news");
  const [hfSplit, setHfSplit] = useState("train");
  const [hfField, setHfField] = useState("text");
  const hfPresets = [
    { id: "ag_news:text", name: "ag_news (text, train)", value: { name: "ag_news", split: "train", field: "text" } },
    {
      id: "roneneldan/TinyStories:text",
      name: "TinyStories (text, train)",
      value: { name: "roneneldan/TinyStories", split: "train", field: "text" },
    },
    {
      id: "imdb:text",
      name: "imdb (text, train)",
      value: { name: "imdb", split: "train", field: "text" },
    },
    {
      id: "yelp_review_full:text",
      name: "yelp_review_full (text, train)",
      value: { name: "yelp_review_full", split: "train", field: "text" },
    },
    {
      id: "c4:train",
      name: "c4 (text, en, train)",
      value: { name: "c4", split: "train", field: "text" },
    },
  ];
  const [paths, setPaths] = useState("/path/to/file.txt");
  const [layerSel, setLayerSel] = useState<string[]>([]);
  const [batchSize, setBatchSize] = useState(4);
  const [shardSize, setShardSize] = useState(64);
  const [sampleLimit, setSampleLimit] = useState<number | undefined>();
  const [status, setStatus] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [polling, setPolling] = useState<NodeJS.Timeout | null>(null);
  const [selectedRun, setSelectedRun] = useState<ActivationRunInfo | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    if (layers && layers.length && !layerSel.length) {
      setLayerSel([layers[0].layer_id]);
    }
  }, [layers, layerSel]);

  const submit = async () => {
    if (!selectedModel || !modelLoaded) return;
    if (isSubmitting) return;
    setIsSubmitting(true);
    setStatus("Submitting activation run...");
    try {
      const runId =
        typeof window !== "undefined"
          ? `${selectedModel}-${Math.random().toString(36).slice(2, 10)}`
          : `${selectedModel}-pending`;
      const payload =
        datasetMode === "hf"
          ? {
            model_id: selectedModel,
            layers: layerSel,
            dataset: { type: "hf", name: hfName, split: hfSplit, text_field: hfField },
            batch_size: batchSize,
            shard_size: shardSize,
            sample_limit: sampleLimit,
            run_id: runId,
          }
          : {
            model_id: selectedModel,
            layers: layerSel,
            dataset: { type: "local", paths: paths.split(",").map((p) => p.trim()) },
            batch_size: batchSize,
            shard_size: shardSize,
            sample_limit: sampleLimit,
            run_id: runId,
          };
      // Optimistically add an in-progress run to the sidebar so the user sees it immediately.
      const optimisticRun: ActivationRunInfo = {
        model_id: selectedModel,
        run_id: runId,
        layers: layerSel,
        dataset:
          datasetMode === "hf"
            ? { type: "hf", name: hfName, split: hfSplit, text_field: hfField }
            : { type: "local", paths: paths.split(",").map((p) => p.trim()) },
        samples: undefined,
        tokens: undefined,
        status: "running",
      };
      refreshRuns(
        (current) => {
          const existingRuns = current?.runs ?? [];
          if (existingRuns.find((r) => r.run_id === runId)) return current;
          return {
            ...(current || { model_id: selectedModel, runs: [] }),
            runs: [optimisticRun, ...existingRuns],
          };
        },
        false
      );
      const res: any = await api.saveActivations(payload);
      setStatus(`Saved. run_id=${res.run_id}, samples=${res.samples}, tokens=${res.tokens}`);
      // Update the optimistic run with final stats and mark as done.
      refreshRuns(
        (current) => {
          if (!current) return current;
          return {
            ...current,
            runs: (current.runs ?? []).map((r) =>
              r.run_id === runId
                ? {
                  ...r,
                  samples: res.samples ?? r.samples,
                  tokens: res.tokens ?? r.tokens,
                  manifest_path: res.manifest_path ?? r.manifest_path,
                  status: res.status ?? "done",
                  created_at: res.created_at ?? r.created_at,
                  layers: Array.isArray(res.layers) && res.layers.length ? res.layers : r.layers,
                  dataset: res.dataset ?? r.dataset,
                }
                : r
            ),
          };
        },
        false
      );
      // Finally, revalidate from the server to ensure everything is in sync.
      refreshRuns();
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  let activationRunsForSidebar: ActivationRunInfo[] = [];
  if (runs && Array.isArray(runs.runs)) {
    activationRunsForSidebar = runs.runs;
  } else if (
    selectedModel &&
    storeInfo &&
    storeInfo.activation_datasets &&
    Array.isArray(storeInfo.activation_datasets[selectedModel])
  ) {
    activationRunsForSidebar = storeInfo.activation_datasets[selectedModel];
  }

  // Sort runs from latest to oldest by created_at (fallback to run_id).
  activationRunsForSidebar = [...activationRunsForSidebar].sort((a, b) => {
    const aTime = a.created_at && !Number.isNaN(Date.parse(a.created_at)) ? Date.parse(a.created_at) : 0;
    const bTime = b.created_at && !Number.isNaN(Date.parse(b.created_at)) ? Date.parse(b.created_at) : 0;
    if (aTime !== bTime) return bTime - aTime;
    return (b.run_id || "").localeCompare(a.run_id || "");
  });

  const hasRuns = useMemo(() => activationRunsForSidebar.length > 0, [activationRunsForSidebar]);

  // Periodically refresh activation runs while a model is loaded and there are runs for it.
  useEffect(() => {
    if (!selectedModel || !modelLoaded || !hasRuns) {
      if (polling) {
        clearInterval(polling);
        setPolling(null);
      }
      return;
    }
    if (polling) return;
    const id = setInterval(() => {
      refreshRuns();
    }, 5000);
    setPolling(id);
    return () => {
      clearInterval(id);
    };
  }, [selectedModel, modelLoaded, hasRuns, refreshRuns, polling]);

  const steps = [
    (
      <StepCard
        step={1}
        title="Load model"
        description="Pick a language model and load it into memory. Other steps will unlock afterwards."
      >
        <div className="space-y-3">
          <div className="space-y-1">
            <Label>Model</Label>
            <select
              className="input"
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setModelLoaded(false);
                setLayerSel([]);
              }}
            >
              {models?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} [{m.id}] ({m.status})
                </option>
              ))}
            </select>
          </div>
          <div>
            <Button
              onClick={async () => {
                if (!selectedModel) return;
                setStatus("Loading model...");
                try {
                  await loadModel();
                  setStatus("Model loaded");
                } catch (e: any) {
                  setStatus(`Error loading model: ${e.message}`);
                }
              }}
              disabled={!selectedModel || modelLoaded}
            >
              {modelLoaded ? "Model loaded" : "Load model"}
            </Button>
          </div>
        </div>
      </StepCard>
    ),
    (
      <StepCard
        step={2}
        title="Choose layers, dataset and save activations"
        description="Select which layers to capture, choose a dataset, and launch an activation capture run."
      >
        <div className={`space-y-5 ${!modelLoaded ? "opacity-50 pointer-events-none" : ""}`}>
          <Row>
            <div className="space-y-1">
              <Label>Layers</Label>
              <select
                multiple
                className="input h-28"
                value={layerSel}
                onChange={(e) => setLayerSel(Array.from(e.target.selectedOptions).map((o) => o.value))}
              >
                {layers?.map((l) => (
                  <option key={l.layer_id} value={l.layer_id}>
                    {l.name}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500">Cmd/Ctrl+click to multi-select.</p>
            </div>
          </Row>

          <div className="space-y-4 text-sm">
            <div className="flex gap-3">
              <button
                className={`px-3 py-1 rounded-md border ${datasetMode === "hf" ? "border-sky-500 text-sky-300" : "border-slate-700"}`}
                onClick={() => setDatasetMode("hf")}
              >
                HF dataset
              </button>
              <button
                className={`px-3 py-1 rounded-md border ${datasetMode === "local" ? "border-sky-500 text-sky-300" : "border-slate-700"}`}
                onClick={() => setDatasetMode("local")}
              >
                Local files
              </button>
            </div>

            {storeInfo && selectedModel && storeInfo.activation_datasets?.[selectedModel]?.length ? (
              <div className="space-y-1">
                <Label>Previously used datasets for this model</Label>
                <select
                  className="input"
                  onChange={(e) => {
                    const runId = e.target.value;
                    const run = storeInfo.activation_datasets[selectedModel].find((r) => r.run_id === runId);
                    if (!run) return;
                    const ds = run.dataset || {};
                    const type = ds.type === "hf" ? "hf" : "local";
                    setDatasetMode(type);
                    if (type === "hf") {
                      setHfName(ds.name || "");
                      setHfSplit(ds.split || "train");
                      setHfField(ds.text_field || "text");
                    } else if (type === "local") {
                      const pathsVal = Array.isArray(ds.paths) ? ds.paths.join(", ") : "";
                      setPaths(pathsVal || "");
                    }
                  }}
                  defaultValue=""
                >
                  <option value="" disabled>
                    Select a previous run to reuse its dataset
                  </option>
                  {storeInfo.activation_datasets[selectedModel].map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.run_id} ({r.dataset?.type === "hf" ? r.dataset.name : "local"})
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-500">
                  Selecting a run will pre-fill the dataset fields below based on its manifest.
                </p>
              </div>
            ) : null}

            {datasetMode === "hf" ? (
              <div className="space-y-3">
                <div className="space-y-1">
                  <Label>Preset HF datasets</Label>
                  <select
                    className="input"
                    defaultValue=""
                    onChange={(e) => {
                      const preset = hfPresets.find((p) => p.id === e.target.value);
                      if (!preset) return;
                      setHfName(preset.value.name);
                      setHfSplit(preset.value.split);
                      setHfField(preset.value.field);
                    }}
                  >
                    <option value="">Custom / manual</option>
                    {hfPresets.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-500">
                    Choose a preset or leave on &quot;Custom&quot; to fill in fields manually.
                  </p>
                </div>
                <Row>
                  <div className="space-y-1">
                    <Label>HF name</Label>
                    <Input value={hfName} onChange={(e) => setHfName(e.target.value)} />
                  </div>
                  <div className="space-y-1">
                    <Label>Split</Label>
                    <Input value={hfSplit} onChange={(e) => setHfSplit(e.target.value)} />
                  </div>
                  <div className="space-y-1">
                    <Label>Text field</Label>
                    <Input value={hfField} onChange={(e) => setHfField(e.target.value)} />
                  </div>
                </Row>
              </div>
            ) : (
              <div className="space-y-1">
                <Label>Local paths (comma-separated)</Label>
                <Input value={paths} onChange={(e) => setPaths(e.target.value)} />
              </div>
            )}
          </div>

          <Row>
            <div className="space-y-1">
              <Label>Batch size</Label>
              <Input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                min={1}
              />
            </div>
            <div className="space-y-1">
              <Label>Shard size</Label>
              <Input type="number" value={shardSize} onChange={(e) => setShardSize(Number(e.target.value))} min={1} />
            </div>
            <div className="space-y-1">
              <Label>Sample limit (optional)</Label>
              <Input
                type="number"
                value={sampleLimit ?? ""}
                onChange={(e) => setSampleLimit(e.target.value ? Number(e.target.value) : undefined)}
                min={1}
              />
            </div>
          </Row>

          <div className="flex items-center justify-between">
          <Button onClick={submit} disabled={!modelLoaded || isSubmitting}>
              Save activations
            </Button>
            {status && <p className="text-sm text-slate-300">{status}</p>}
          </div>
        </div>
      </StepCard>
    ),
  ];

  const sidebar = (
    <RunHistorySidebar
      title="Activation runs"
      items={activationRunsForSidebar}
      emptyMessage="No activation runs yet for this model."
      renderItem={(r) => {
        const isInProgress = !r.samples && !r.tokens;
        const startedAt =
          r.created_at && !Number.isNaN(Date.parse(r.created_at))
            ? new Date(r.created_at).toLocaleString()
            : "-";
        return (
          <button
            type="button"
            onClick={() => {
              setSelectedRun(r);
              setIsModalOpen(true);
            }}
            className="w-full text-left"
          >
            <div className="space-y-1 text-xs">
              <div className="font-semibold text-amber-600 truncate" title={r.run_id}>
                {r.run_id}
              </div>
              <div className="text-slate-400">started: {startedAt}</div>
              <div className="text-slate-400">
                status: {isInProgress ? "in progress" : r.status ?? "done"}
              </div>
              <div className="text-slate-400">
                layers: {r.layers?.length ? r.layers.join(", ") : "-"}
              </div>
              <div className="text-slate-400">
                samples: {r.samples ?? "-"} | tokens: {r.tokens ?? "-"}
              </div>
              {r.dataset && (
                <div className="text-slate-500">
                  dataset:{" "}
                  {r.dataset.type === "hf"
                    ? r.dataset.name ?? "hf"
                    : r.dataset.type === "local"
                    ? "local files"
                    : r.dataset.type ?? "unknown"}
                </div>
              )}
            </div>
          </button>
        );
      }}
    />
  );

  return (
    <>
      <StepLayout
        title="Capture Activations"
        description="Save model activations to disk for later SAE training."
        steps={steps}
        sidebar={sidebar}
      />
      {isModalOpen && selectedRun && (
        <Modal
          title={`Activation run ${selectedRun.run_id}`}
          onClose={() => {
            setIsModalOpen(false);
            setSelectedRun(null);
          }}
        >
          <div className="space-y-1">
            <div>
              <span className="font-semibold text-slate-300">Model:</span>{" "}
              <span className="text-slate-200">{selectedRun.model_id}</span>
            </div>
            <div>
              <span className="font-semibold text-slate-300">Status:</span>{" "}
              <span className="text-slate-200">{selectedRun.status ?? (selectedRun.samples ? "done" : "in progress")}</span>
            </div>
            <div>
              <span className="font-semibold text-slate-300">Layers:</span>{" "}
              <span className="text-slate-200">{selectedRun.layers?.length ? selectedRun.layers.join(", ") : "-"}</span>
            </div>
            <div>
              <span className="font-semibold text-slate-300">Samples / tokens:</span>{" "}
              <span className="text-slate-200">
                {selectedRun.samples ?? "-"} samples, {selectedRun.tokens ?? "-"} tokens
              </span>
            </div>
            <div>
              <span className="font-semibold text-slate-300">Dataset:</span>{" "}
              <span className="text-slate-200">
                {selectedRun.dataset?.type === "hf"
                  ? `${selectedRun.dataset.name ?? "hf"} (${selectedRun.dataset.split ?? "train"}, field=${
                      selectedRun.dataset.text_field ?? "text"
                    })`
                  : selectedRun.dataset?.type === "local"
                  ? Array.isArray(selectedRun.dataset.paths)
                    ? selectedRun.dataset.paths.join(", ")
                    : "local files"
                  : "-"}
              </span>
            </div>
            {selectedRun.manifest_path && (
              <div className="text-slate-400">
                <span className="font-semibold text-slate-300">Manifest path:</span>{" "}
                <span className="break-all text-slate-400">{selectedRun.manifest_path}</span>
              </div>
            )}
            {selectedRun.created_at && (
              <div>
                <span className="font-semibold text-slate-300">Created at:</span>{" "}
                <span className="text-slate-200">{selectedRun.created_at}</span>
              </div>
            )}
          </div>
        </Modal>
      )}
    </>
  );
}

