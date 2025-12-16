"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { InferenceOutput, SaeRunInfo, TopNeurons, TokenLatents } from "@/lib/types";
import { useModelLoader } from "@/lib/useModelLoader";
import { Button, Card, Input, Label, Row, SectionTitle, TextArea } from "@/components/ui";

type Prompt = { prompt: string };

export default function InferencePage() {
  const { models, modelId, setModelId, modelLoaded, setModelLoaded, loadModel } = useModelLoader();
  const { data: saes } = useSWR<{ saes: SaeRunInfo[] }>(
    modelId && modelLoaded ? `/sae/saes?model_id=${modelId}` : null,
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
  const [result, setResult] = useState<{ outputs: InferenceOutput[]; top_neurons: TopNeurons[]; token_latents: TokenLatents[]; top_texts_path?: string }>({
    outputs: [],
    top_neurons: [],
    token_latents: [],
  });
  const [status, setStatus] = useState("");

  useEffect(() => {
    if (saes?.saes?.length && !saeId) {
      const first = saes.saes[0];
      setSaeId(first.sae_id);
      setLayer(first.layer ?? "");
    }
  }, [saes, saeId]);

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

  const run = async () => {
    if (!modelId || !saeId || !modelLoaded) return;
    setStatus("Running inference...");
    try {
      const payload = {
        model_id: modelId,
        sae_id: saeId,
        layer,
        inputs: prompts.map((p) => ({ prompt: p.prompt })),
        top_k_neurons: topK,
        save_top_texts: saveTopTexts,
        track_texts: trackTexts,
        return_token_latents: returnTokenLatents,
        concept_config_path: conceptConfigPath || undefined,
      };
      const res = await api.infer(payload);
      setResult(res as any);
      setStatus("Done");
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    }
  };

  return (
    <div className="space-y-6">
      <SectionTitle>Inference</SectionTitle>
      <Card className="space-y-4">
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
            <Label>Layer</Label>
            <Input value={layer} onChange={(e) => setLayer(e.target.value)} disabled={!modelLoaded} />
          </div>
        </Row>

        <Row className={!modelLoaded ? "opacity-50 pointer-events-none" : ""}>
          <div className="space-y-1">
            <Label>Top-K neurons</Label>
            <Input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} min={1} />
          </div>
          <div className="space-y-1">
            <Label>Concept config path (optional)</Label>
            <Input value={conceptConfigPath} onChange={(e) => setConceptConfigPath(e.target.value)} />
          </div>
        </Row>

        <div className={`flex gap-4 text-sm ${!modelLoaded ? "opacity-50 pointer-events-none" : ""}`}>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={saveTopTexts} onChange={(e) => setSaveTopTexts(e.target.checked)} /> Save
            top texts
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={trackTexts} onChange={(e) => setTrackTexts(e.target.checked)} /> Track
            texts
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={returnTokenLatents}
              onChange={(e) => setReturnTokenLatents(e.target.checked)}
            />{" "}
            Return token latents
          </label>
        </div>

        <div className={`space-y-2 ${!modelLoaded ? "opacity-50 pointer-events-none" : ""}`}>
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

        <Button onClick={run} disabled={!modelLoaded}>
          Run inference
        </Button>
        {status && <p className="text-sm text-slate-300">{status}</p>}
      </Card>

      {result.outputs?.length ? (
        <div className="space-y-4">
          <SectionTitle>Outputs</SectionTitle>
          {result.outputs.map((o, idx) => (
            <Card key={idx} className="space-y-2">
              <div className="text-xs text-slate-400">Prompt #{idx + 1}</div>
              <div className="text-slate-100 whitespace-pre-wrap">{o.text}</div>
              <div className="text-xs text-slate-500">Tokens: {o.tokens?.join(" ")}</div>
            </Card>
          ))}

          <SectionTitle>Top neurons</SectionTitle>
          <Card className="space-y-2 text-sm">
            {result.top_neurons?.map((t, idx) => (
              <div key={idx} className="border border-slate-800 rounded-md p-2">
                <div className="font-semibold text-slate-100">Prompt #{t.prompt_index + 1}</div>
                <div className="text-slate-300">Neurons: {t.neuron_ids.join(", ")}</div>
                <div className="text-slate-500 text-xs">Acts: {t.activations.map((a) => a.toFixed(3)).join(", ")}</div>
              </div>
            ))}
          </Card>

          {result.token_latents?.length ? (
            <>
              <SectionTitle>Token latents (top-k per token)</SectionTitle>
              <Card className="space-y-3 text-sm">
                {result.token_latents.map((p, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="font-semibold text-slate-100">Prompt #{p.prompt_index + 1}</div>
                    {p.tokens.map((tok, j) => (
                      <div key={j} className="text-slate-300">
                        token {tok.token_index}: {tok.neuron_ids.join(", ")}
                      </div>
                    ))}
                  </div>
                ))}
              </Card>
            </>
          ) : null}

          {result.top_texts_path && (
            <Card>
              <div className="text-sm text-slate-200">
                Top texts saved at: <a href={result.top_texts_path}>{result.top_texts_path}</a>
              </div>
            </Card>
          )}
        </div>
      ) : null}
    </div>
  );
}

