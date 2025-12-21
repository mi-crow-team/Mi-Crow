"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import {
  InferenceOutput,
  SaeRunInfo,
  TopNeurons,
  TokenLatents,
  InferenceHistoryEntry,
  ConceptConfigInfo,
} from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Card, Input, Label, Row, SectionTitle, TextArea, Spinner } from "@/components/ui";
import { StepCard, StepLayout } from "@/components/StepLayout";
import { InferenceSidebar } from "@/components/InferenceSidebar";
import { InferenceRunCard } from "@/components/InferenceRunCard";
import { ConceptManipulator } from "@/components/ConceptManipulator";

type Prompt = { prompt: string };

type InferenceRun = {
  id: string;
  timestamp: string;
  config: {
    model_id: string;
    sae_id: string;
    layer: string;
    prompts: string[];
    topK: number;
    saveTopTexts: boolean;
    trackTexts: boolean;
    returnTokenLatents: boolean;
    conceptConfigPath?: string;
  };
  outputs: InferenceOutput[];
  top_neurons: TopNeurons[];
  token_latents: TokenLatents[];
  top_texts_path?: string;
};

const HISTORY_STORAGE_KEY = "inference_history";
const MAX_HISTORY_ENTRIES = 100;

export default function InferencePage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel, isLoading: isLoadingModel } = useModelLoader();
  const { data: saes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId ? `/sae/saes?model_id=${modelId}` : null,
    () => api.listSaes(modelId)
  );
  const [saeId, setSaeId] = useState("");
  const [layer, setLayer] = useState("");
  const [prompts, setPrompts] = useState<Prompt[]>([{ prompt: "Hello world" }]);
  const [topK, setTopK] = useState(5);
  const [saveTopTexts, setSaveTopTexts] = useState(false);
  const [trackTexts, setTrackTexts] = useState(false);
  const [returnTokenLatents, setReturnTokenLatents] = useState(false);
  const [conceptConfigPath, setConceptConfigPath] = useState("");
  const [status, setStatus] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  // Feature toggles
  const [loadConcepts, setLoadConcepts] = useState(false);
  const [conceptEdits, setConceptEdits] = useState<Record<string, number>>({});
  const [conceptBias, setConceptBias] = useState<Record<string, number>>({});
  const [conceptPreview, setConceptPreview] = useState<string>("");
  const [conceptSaveResult, setConceptSaveResult] = useState<string>("");

  // Multiple runs
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [runCounter, setRunCounter] = useState(1);

  // History
  const [history, setHistory] = useState<InferenceHistoryEntry[]>([]);
  const [selectedHistoryEntry, setSelectedHistoryEntry] = useState<InferenceHistoryEntry | null>(null);

  // Concept configs
  const { data: configs, mutate: refreshConfigs } = useSWR<{ configs: ConceptConfigInfo[] }>(
    modelId && modelLoaded && saeId ? `/sae/concepts/configs?model_id=${modelId}&sae_id=${saeId}` : null,
    () => api.listConceptConfigs(modelId, saeId)
  );

  // Load history from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setHistory(Array.isArray(parsed) ? parsed : []);
      }
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, []);

  useEffect(() => {
    if (saes?.saes?.length && !saeId) {
      const first = saes.saes[0];
      setSaeId(first.sae_id);
    }
  }, [saes, saeId]);

  // Auto-set layer from selected SAE
  const currentSae = useMemo(() => {
    return saes?.saes?.find((s) => s.sae_id === saeId);
  }, [saes, saeId]);

  useEffect(() => {
    if (currentSae?.layer) {
      setLayer(currentSae.layer);
    } else {
      setLayer("");
    }
  }, [currentSae]);

  const loadSelectedModel = async () => {
    if (!modelId) return;
    setStatus("Loading model...");
    try {
      await loadModel();
      setStatus("Model loaded");
    } catch (e: any) {
      setStatus(`Error loading model: ${(e as any).message}`);
      setModelLoaded(false);
    }
  };

  const saveToHistory = (entry: InferenceHistoryEntry) => {
    try {
      const newHistory = [entry, ...history].slice(0, MAX_HISTORY_ENTRIES);
      setHistory(newHistory);
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(newHistory));
    } catch (e) {
      console.error("Failed to save history:", e);
    }
  };

  const run = async () => {
    if (!modelId || !saeId || !modelLoaded) return;
    setIsRunning(true);
    setStatus("Running inference...");
    try {
      // Find the selected SAE to get its sae_path
      const selectedSae = saes?.saes?.find((s) => s.sae_id === saeId);
      if (!selectedSae) {
        throw new Error(`SAE ${saeId} not found in available SAEs`);
      }
      if (!selectedSae.sae_path) {
        throw new Error(`SAE ${saeId} is missing sae_path. The SAE file may not exist.`);
      }
      
      const payload = {
        model_id: modelId,
        sae_id: saeId,
        sae_path: selectedSae.sae_path,
        layer,
        inputs: prompts.map((p) => ({ prompt: p.prompt })),
        top_k_neurons: topK,
        save_top_texts: saveTopTexts,
        track_texts: trackTexts,
        return_token_latents: returnTokenLatents,
        concept_config_path: conceptConfigPath || undefined,
      };
      const res = await api.infer(payload);
      const runId = `run-${Date.now()}-${runCounter}`;
      const timestamp = new Date().toISOString();

      const newRun: InferenceRun = {
        id: runId,
        timestamp,
        config: {
          model_id: modelId,
          sae_id: saeId,
          layer,
          prompts: prompts.map((p) => p.prompt),
          topK,
          saveTopTexts,
          trackTexts,
          returnTokenLatents,
          conceptConfigPath: conceptConfigPath || undefined,
        },
        outputs: res.outputs || [],
        top_neurons: res.top_neurons || [],
        token_latents: res.token_latents || [],
        top_texts_path: res.top_texts_path,
      };

      setRuns([newRun, ...runs]);
      setRunCounter(runCounter + 1);

      // Save to history
      const historyEntry: InferenceHistoryEntry = {
        id: runId,
        timestamp,
        model_id: modelId,
        sae_id: saeId,
        layer,
        prompts: prompts.map((p) => p.prompt),
        outputs: res.outputs || [],
        top_neurons: res.top_neurons || [],
        token_latents: res.token_latents || [],
        top_texts_path: res.top_texts_path,
        status: "completed",
        settings: {
          saveTopTexts,
          trackTexts,
          returnTokenLatents,
          conceptConfigPath: conceptConfigPath || undefined,
          topK,
        },
      };
      saveToHistory(historyEntry);

      setStatus("Done");
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
      // Save failed run to history
      const failedEntry: InferenceHistoryEntry = {
        id: `run-${Date.now()}-${runCounter}`,
        timestamp: new Date().toISOString(),
        model_id: modelId,
        sae_id: saeId,
        layer,
        prompts: prompts.map((p) => p.prompt),
        outputs: [],
        top_neurons: [],
        token_latents: [],
        status: "failed",
        error: e.message,
        settings: {
          saveTopTexts,
          trackTexts,
          returnTokenLatents,
          conceptConfigPath: conceptConfigPath || undefined,
          topK,
        },
      };
      saveToHistory(failedEntry);
      setRunCounter(runCounter + 1);
    } finally {
      setIsRunning(false);
    }
  };

  const handleConceptPreview = async () => {
    if (!modelId || !saeId || !modelLoaded) return;
    try {
      const res = await api.previewConcept({
        model_id: modelId,
        sae_id: saeId,
        edits: conceptEdits,
        bias: conceptBias,
      });
      setConceptPreview(JSON.stringify(res, null, 2));
    } catch (e: any) {
      setConceptPreview(`Error: ${e.message}`);
    }
  };

  const handleConceptSaveConfig = async () => {
    if (!modelId || !saeId || !modelLoaded) return;
    try {
      const res: any = await api.manipulateConcept({
        model_id: modelId,
        sae_id: saeId,
        edits: conceptEdits,
        bias: conceptBias,
      });
      setConceptSaveResult(`Saved config at ${res.concept_config_path}`);
      setConceptConfigPath(res.concept_config_path);
      refreshConfigs();
    } catch (e: any) {
      setConceptSaveResult(`Error: ${e.message}`);
    }
  };

  const sidebar = (
    <InferenceSidebar
      history={history}
      onSelectHistory={setSelectedHistoryEntry}
      selectedHistoryEntry={selectedHistoryEntry}
      onCloseHistoryModal={() => setSelectedHistoryEntry(null)}
      settings={{
        loadConcepts,
        saveTopTexts,
        trackTexts,
      }}
      onSettingsChange={(newSettings) => {
        setLoadConcepts(newSettings.loadConcepts);
        setSaveTopTexts(newSettings.saveTopTexts);
        setTrackTexts(newSettings.trackTexts);
      }}
    />
  );

  const steps = (
    <>
      <StepCard
        step={1}
        title="Load model"
        description="Choose which model to use for inference. This must be loaded before running inference."
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
                setSaeId("");
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
            <Button onClick={loadSelectedModel} disabled={!modelId || modelLoaded || isLoadingModel}>
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
        {status && <p className="text-sm text-slate-600">{status}</p>}
      </StepCard>

      <StepCard
        step={2}
        title="Configure inference"
        description="Select SAE, layer, and set up prompts and options for inference."
      >
        <div className={!modelLoaded ? "opacity-50 pointer-events-none space-y-4" : "space-y-4"}>
          <Row>
            <div className="space-y-1">
              <Label>SAE</Label>
              <select
                className="input"
                value={saeId}
                onChange={(e) => setSaeId(e.target.value)}
                disabled={!modelLoaded}
              >
                {saes?.saes?.map((s) => (
                  <option key={s.sae_id} value={s.sae_id}>
                    {s.sae_id}
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
              {!currentSae?.layer && (
                <p className="text-xs text-slate-500">Layer will be set from selected SAE</p>
              )}
            </div>
          </Row>

          <Row>
            <div className="space-y-1">
              <Label>Top-K neurons</Label>
              <Input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} min={1} />
            </div>
            {loadConcepts && (
              <div className="space-y-1">
                <Label>Concept config path (optional)</Label>
                <Input value={conceptConfigPath} onChange={(e) => setConceptConfigPath(e.target.value)} />
              </div>
            )}
          </Row>

          <div className="space-y-2">
            <Label>Options</Label>
            <div className="flex gap-4 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={saveTopTexts}
                  onChange={(e) => setSaveTopTexts(e.target.checked)}
                  className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                />
                Save top texts
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={trackTexts}
                  onChange={(e) => setTrackTexts(e.target.checked)}
                  className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                />
                Track texts
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={returnTokenLatents}
                  onChange={(e) => setReturnTokenLatents(e.target.checked)}
                  className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                />
                Return token latents
              </label>
            </div>
          </div>
        </div>
      </StepCard>

      {loadConcepts && modelId && saeId && modelLoaded && (
        <StepCard
          step={3}
          title="Concept manipulation"
          description="Select and adjust concepts to manipulate during inference."
        >
          <div className="space-y-4">
            <ConceptManipulator
              modelId={modelId}
              saeId={saeId}
              onEditsChange={setConceptEdits}
              onBiasChange={setConceptBias}
              onPreview={handleConceptPreview}
              onSaveConfig={handleConceptSaveConfig}
            />
            {conceptPreview && (
              <pre className="text-xs bg-slate-900 border border-slate-800 rounded p-2 whitespace-pre-wrap text-slate-100">
                {conceptPreview}
              </pre>
            )}
            {conceptSaveResult && <p className="text-sm text-slate-600">{conceptSaveResult}</p>}
            {configs?.configs && configs.configs.length > 0 && (
              <div className="space-y-2">
                <Label>Concept Configs</Label>
                <div className="space-y-2 text-sm">
                  {configs.configs.map((c) => (
                    <div key={c.path} className="border border-slate-200 rounded-md p-2">
                      <div className="font-semibold text-slate-900">{c.name}</div>
                      <div className="text-slate-500 text-xs">path: {c.path}</div>
                      <Button
                        variant="ghost"
                        className="mt-2 text-xs"
                        onClick={() => setConceptConfigPath(c.path)}
                      >
                        Use this config
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </StepCard>
      )}

      <StepCard
        step={loadConcepts && modelId && saeId && modelLoaded ? 4 : 3}
        title="Set prompts and run"
        description="Enter prompts to run inference on and execute the inference."
      >
        <div className={!modelLoaded ? "opacity-50 pointer-events-none space-y-4" : "space-y-4"}>
          <div className="space-y-2">
            <Label>Prompts</Label>
            {prompts.map((p, idx) => (
              <div key={idx} className="flex gap-2">
                <TextArea
                  value={p.prompt}
                  onChange={(e) => {
                    const next = [...prompts];
                    next[idx] = { prompt: e.target.value };
                    setPrompts(next);
                  }}
                />
                <Button
                  variant="ghost"
                  onClick={() => setPrompts(prompts.filter((_, i) => i !== idx))}
                  disabled={prompts.length === 1}
                >
                  Remove
                </Button>
              </div>
            ))}
            <Button variant="ghost" onClick={() => setPrompts([...prompts, { prompt: "" }])}>
              Add prompt
            </Button>
          </div>

          <Button onClick={run} disabled={!modelLoaded || isRunning}>
            {isRunning ? (
              <span className="flex items-center gap-2">
                <Spinner /> Running inference...
              </span>
            ) : (
              "Run inference"
            )}
          </Button>
        </div>
      </StepCard>
    </>
  );

  return (
    <div className="space-y-6">
      <StepLayout
        title="Inference"
        description="Run inference with SAEs, manipulate concepts, and view results."
        steps={steps}
        sidebar={sidebar}
      />

      {/* Multiple Runs Display */}
      {runs.length > 0 && (
        <div className="space-y-4">
          <SectionTitle>Inference Runs</SectionTitle>
          {runs.map((run, idx) => (
            <InferenceRunCard
              key={run.id}
              runNumber={runs.length - idx}
              timestamp={run.timestamp}
              config={run.config}
              outputs={run.outputs}
              top_neurons={run.top_neurons}
              token_latents={run.token_latents}
              top_texts_path={run.top_texts_path}
              onDelete={() => {
                setRuns(runs.filter((r) => r.id !== run.id));
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
