"use client";

import { useEffect, useState, useMemo } from "react";
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
import type { ConceptManipulationResponse, ConceptPreviewResponse } from "@/lib/api/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { useInferenceState } from "@/hooks/useInferenceState";
import { useInferenceHistory } from "@/hooks/useInferenceHistory";
import { useConceptManipulation } from "@/hooks/useConceptManipulation";
import { Button, Card, Input, Label, Row, SectionTitle, TextArea, Spinner } from "@/components/ui";
import { StepCard, StepLayout } from "@/components/StepLayout";
import { InferenceSidebar } from "@/components/InferenceSidebar";
import { InferenceRunCard } from "@/components/InferenceRunCard";
import { ConceptManipulator } from "@/components/ConceptManipulator";
import { InferenceForm } from "@/components/InferenceForm";
import { InferenceConfig } from "@/components/InferenceConfig";

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


export default function InferencePage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel, isLoading: isLoadingModel } = useModelLoader();
  const { data: saes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId ? `/sae/saes?model_id=${modelId}` : null,
    () => api.listSaes(modelId)
  );
  const [status, setStatus] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [runs, setRuns] = useState<InferenceRun[]>([]);
  const [runCounter, setRunCounter] = useState(1);

  // Use custom hooks for state management
  const inferenceState = useInferenceState(saes);
  const { history, selectedHistoryEntry, setSelectedHistoryEntry, saveToHistory, clearHistory } = useInferenceHistory();
  const conceptManipulation = useConceptManipulation();
  
  // Destructure inference state for easier access
  const {
    saeId, setSaeId, layer, prompts, setPrompts, topK, setTopK,
    saveTopTexts, setSaveTopTexts, trackTexts, setTrackTexts,
    returnTokenLatents, setReturnTokenLatents, conceptConfigPath, setConceptConfigPath,
    loadConcepts, setLoadConcepts
  } = inferenceState;
  
  // Destructure concept manipulation
  const {
    conceptEdits, setConceptEdits, conceptBias, setConceptBias,
    conceptPreview, conceptSaveResult, previewConcept, saveConceptConfig, clearConceptManipulation
  } = conceptManipulation;

  // Concept configs
  const { data: configs, mutate: refreshConfigs } = useSWR<{ configs: ConceptConfigInfo[] }>(
    modelId && modelLoaded && saeId ? `/sae/concepts/configs?model_id=${modelId}&sae_id=${saeId}` : null,
    () => api.listConceptConfigs(modelId, saeId)
  );

  // Auto-set layer from selected SAE
  const currentSae = useMemo(() => {
    return saes?.saes?.find((s) => s.sae_id === saeId);
  }, [saes, saeId]);

  const loadSelectedModel = async () => {
    if (!modelId) return;
    setStatus("Loading model...");
    try {
      await loadModel();
      setStatus("Model loaded");
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : String(e);
      setStatus(`Error loading model: ${error}`);
      setModelLoaded(false);
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
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : String(e);
      setStatus(`Error: ${error}`);
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
        error: e instanceof Error ? e.message : String(e),
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
    await previewConcept(modelId, saeId);
  };

  const handleConceptSaveConfig = async () => {
    if (!modelId || !saeId || !modelLoaded) return;
    await saveConceptConfig(modelId, saeId, (path) => {
      setConceptConfigPath(path);
      refreshConfigs();
    });
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
        <InferenceConfig
          {...inferenceState}
          saes={saes}
          currentSae={currentSae}
          disabled={!modelLoaded}
        />
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
        <InferenceForm
          {...inferenceState}
          disabled={!modelLoaded}
          onRun={run}
          isRunning={isRunning}
        />
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
