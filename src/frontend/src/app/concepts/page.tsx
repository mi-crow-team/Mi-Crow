"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { ConceptConfigInfo, SaeRunInfo } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Card, Input, Label, Row, SectionTitle, TextArea } from "@/components/ui";

export default function ConceptsPage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel } = useModelLoader();
  const { data: saes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId && modelLoaded ? `/sae/saes?model_id=${modelId}` : null,
    () => api.listSaes(modelId)
  );
  const [saeId, setSaeId] = useState("");
  const { data: concepts } = useSWR<{ concepts: string[] }>(
    modelId && modelLoaded ? `/sae/concepts?model_id=${modelId}&sae_id=${saeId}` : null,
    () => api.listConcepts(modelId, saeId)
  );
  const { data: configs, mutate: refreshConfigs } = useSWR<{ configs: ConceptConfigInfo[] }>(
    modelId && modelLoaded ? `/sae/concepts/configs?model_id=${modelId}&sae_id=${saeId}` : null,
    () => api.listConceptConfigs(modelId, saeId)
  );

  const [sourcePath, setSourcePath] = useState("");
  const [edits, setEdits] = useState('{"0":1.2}');
  const [bias, setBias] = useState('{"0":0.0}');
  const [preview, setPreview] = useState<string>("");
  const [saveResult, setSaveResult] = useState<string>("");
  const [selectedConcept, setSelectedConcept] = useState<string | null>(null);
  const [conceptStrength, setConceptStrength] = useState<number>(1.0);

  useEffect(() => {
    if (saes?.saes?.length && !saeId) setSaeId(saes.saes[0].sae_id);
  }, [saes, saeId]);

  const loadSelectedModel = async () => {
    if (!modelId) return;
    setSaveResult("Loading model...");
    try {
      await loadModel();
      setSaveResult("Model loaded");
    } catch (e: any) {
      setSaveResult(`Error loading model: ${(e as any).message}`);
      setModelLoaded(false);
    }
  };

  const doLoad = async () => {
    if (!modelId || !saeId || !sourcePath || !modelLoaded) return;
    try {
      await api.loadConcept({ model_id: modelId, sae_id: saeId, source_path: sourcePath });
      setSaveResult("Loaded concepts");
    } catch (e: any) {
      setSaveResult(`Error: ${e.message}`);
    }
  };

  const doPreview = async () => {
    try {
      const res = await api.previewConcept({
        model_id: modelId,
        sae_id: saeId,
        edits: JSON.parse(edits || "{}"),
        bias: JSON.parse(bias || "{}"),
      });
      setPreview(JSON.stringify(res, null, 2));
    } catch (e: any) {
      setPreview(`Error: ${e.message}`);
    }
  };

  const doSaveConfig = async () => {
    try {
      const res: any = await api.manipulateConcept({
        model_id: modelId,
        sae_id: saeId,
        edits: JSON.parse(edits || "{}"),
        bias: JSON.parse(bias || "{}"),
      });
      setSaveResult(`Saved config at ${res.concept_config_path}`);
      refreshConfigs();
    } catch (e: any) {
      setSaveResult(`Error: ${e.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <SectionTitle>Concepts</SectionTitle>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
        <Card className="space-y-4">
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
            <Button onClick={loadSelectedModel} disabled={!modelId || modelLoaded}>
              {modelLoaded ? "Model loaded" : "Load model"}
            </Button>
          </div>
          <div className="space-y-1">
            <Label>SAE</Label>
            <select
              className={`input ${!modelLoaded ? "opacity-50" : ""}`}
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
            <Label>Concept files (from store)</Label>
            {concepts?.concepts?.length ? (
              <div className="max-h-48 overflow-auto rounded-md border border-slate-800">
                {concepts.concepts.map((c) => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => setSelectedConcept(c)}
                    className={`flex w-full items-center justify-between px-3 py-1 text-left text-sm ${
                      selectedConcept === c ? "bg-slate-800 text-slate-100" : "text-slate-200 hover:bg-slate-900"
                    }`}
                  >
                    <span className="truncate">{c}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-xs text-slate-500">No concept files found. You can still run inference without them.</div>
            )}
          </div>

          <div className="space-y-1">
            <Label>Upload concepts file (path on server)</Label>
            <Input
              value={sourcePath}
              onChange={(e) => setSourcePath(e.target.value)}
              placeholder="/path/to/concepts.json"
              disabled={!modelLoaded}
            />
            <Button onClick={doLoad} disabled={!modelLoaded}>
              Load concepts
            </Button>
          </div>
        </Card>

        <Card className={`space-y-4 ${!modelLoaded ? "opacity-50 pointer-events-none" : ""}`}>
          <SectionTitle>Concept edits</SectionTitle>
          <div className="space-y-2">
            <Label>Edits (per-neuron weights JSON)</Label>
            <TextArea value={edits} onChange={(e) => setEdits(e.target.value)} />
            <Label>Bias (per-neuron bias JSON)</Label>
            <TextArea value={bias} onChange={(e) => setBias(e.target.value)} />
          </div>

          <div className="space-y-2">
            <Label>
              Slider for current concept (applies a simple scalar to all its neurons when you save a config)
            </Label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={conceptStrength}
              onChange={(e) => setConceptStrength(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-slate-400">
              Selected: {selectedConcept ?? "none"} | strength: {conceptStrength.toFixed(1)}
            </div>
          </div>

          <div className="flex gap-2">
            <Button variant="primary" onClick={doPreview}>
              Preview
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                // If a concept is selected and edits is empty, generate a simple uniform edit map.
                if (selectedConcept && (!edits || edits === "{}")) {
                  const uniform = { "0": conceptStrength };
                  setEdits(JSON.stringify(uniform, null, 2));
                }
                doSaveConfig();
              }}
            >
              Save config
            </Button>
          </div>

          {preview && (
            <pre className="text-xs bg-slate-900 border border-slate-800 rounded p-2 whitespace-pre-wrap text-slate-100">
              {preview}
            </pre>
          )}
          {saveResult && <p className="text-sm text-slate-300">{saveResult}</p>}

          <div className="space-y-2">
            <SectionTitle>Concept configs</SectionTitle>
            {configs?.configs?.length ? (
              <div className="space-y-2 text-sm text-slate-200">
                {configs.configs.map((c) => (
                  <div key={c.path} className="border border-slate-800 rounded-md p-2">
                    <div className="font-semibold text-slate-100">{c.name}</div>
                    <div className="text-slate-400 text-xs">path: {c.path}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-slate-500">No concept configs saved yet.</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

