"use client";

import { InferenceOutput, TopNeurons, TokenLatents } from "@/lib/types";
import { Card, SectionTitle, Button } from "./ui";

type InferenceRunCardProps = {
  runNumber: number;
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
  onDelete?: () => void;
};

export function InferenceRunCard({
  runNumber,
  timestamp,
  config,
  outputs,
  top_neurons,
  token_latents,
  top_texts_path,
  onDelete,
}: InferenceRunCardProps) {
  const formatTimestamp = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleString();
  };

  return (
    <Card className="space-y-4 border-2 border-mi_crow-200 bg-mi_crow-50/50">
      <div className="flex items-center justify-between border-b border-mi_crow-200 pb-2">
        <div>
          <h3 className="font-semibold text-slate-900">Run #{runNumber}</h3>
          <div className="text-xs text-slate-600">{formatTimestamp(timestamp)}</div>
        </div>
        {onDelete && (
          <Button
            variant="ghost"
            onClick={onDelete}
            className="text-slate-500 hover:text-red-600 hover:bg-red-50"
            title="Delete this run"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
          </Button>
        )}
      </div>

      <div className="space-y-2 text-sm">
        <div>
          <span className="font-medium text-slate-700">Configuration:</span>
          <div className="mt-1 space-y-1 text-xs text-slate-600">
            <div>
              Model: <span className="font-mono">{config.model_id}</span>
            </div>
            <div>
              SAE: <span className="font-mono">{config.sae_id}</span>
            </div>
            <div>
              Layer: <span className="font-mono">{config.layer}</span>
            </div>
            <div>
              Top-K: <span className="font-mono">{config.topK}</span>
            </div>
            <div>
              Prompts: <span className="font-mono">{config.prompts.length}</span>
            </div>
            <div className="flex gap-4 mt-1">
              <span>Save top texts: {config.saveTopTexts ? "✓" : "✗"}</span>
              <span>Track texts: {config.trackTexts ? "✓" : "✗"}</span>
              <span>Token latents: {config.returnTokenLatents ? "✓" : "✗"}</span>
            </div>
            {config.conceptConfigPath && (
              <div>
                Concept config: <span className="font-mono text-xs">{config.conceptConfigPath}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {outputs.length > 0 && (
        <div className="space-y-2">
          <SectionTitle>Outputs</SectionTitle>
          {outputs.map((o, idx) => (
            <Card key={idx} className="space-y-2 bg-white">
              <div className="text-xs text-slate-600">Prompt #{idx + 1}</div>
              <div className="text-slate-900 whitespace-pre-wrap">{o.text}</div>
              {o.tokens && o.tokens.length > 0 && (
                <div className="text-xs text-slate-600">Tokens: {o.tokens.join(" ")}</div>
              )}
            </Card>
          ))}
        </div>
      )}

      {top_neurons.length > 0 && (
        <div className="space-y-2">
          <SectionTitle>Top Neurons</SectionTitle>
          <Card className="space-y-2 text-sm bg-white">
            {top_neurons.map((t, idx) => (
              <div key={idx} className="border border-slate-200 rounded-md p-2">
                <div className="font-semibold text-slate-900">Prompt #{t.prompt_index + 1}</div>
                <div className="text-slate-700">Neurons: {t.neuron_ids.join(", ")}</div>
                <div className="text-slate-600 text-xs">
                  Activations: {t.activations.map((a) => a.toFixed(3)).join(", ")}
                </div>
              </div>
            ))}
          </Card>
        </div>
      )}

      {token_latents.length > 0 && (
        <div className="space-y-2">
          <SectionTitle>Token Latents</SectionTitle>
          <Card className="space-y-3 text-sm bg-white">
            {token_latents.map((p, idx) => (
              <div key={idx} className="space-y-2">
                <div className="font-semibold text-slate-900">Prompt #{p.prompt_index + 1}</div>
                {p.tokens.map((tok, j) => (
                  <div key={j} className="text-slate-700 text-xs">
                    Token {tok.token_index}: {tok.neuron_ids.join(", ")}
                  </div>
                ))}
              </div>
            ))}
          </Card>
        </div>
      )}

      {top_texts_path && (
        <Card className="bg-white">
          <div className="text-sm text-slate-700">
            Top texts saved at:{" "}
            <a href={top_texts_path} className="text-mi_crow-600 hover:underline">
              {top_texts_path}
            </a>
          </div>
        </Card>
      )}
    </Card>
  );
}

