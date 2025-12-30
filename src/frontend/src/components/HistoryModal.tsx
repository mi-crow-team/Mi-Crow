"use client";

import { InferenceHistoryEntry } from "@/lib/types";
import { Modal } from "./ui";

type HistoryModalProps = {
  entry: InferenceHistoryEntry;
  onClose: () => void;
};

export function HistoryModal({ entry, onClose }: HistoryModalProps) {
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const status = entry.status || (entry.outputs.length > 0 ? "completed" : "failed");
  const isCompleted = status === "completed";

  return (
    <Modal title={`Run - ${formatTimestamp(entry.timestamp)}`} onClose={onClose}>
      <div className="space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-slate-200">
          <h3 className="font-semibold text-slate-900">Configuration</h3>
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              isCompleted
                ? "bg-green-100 text-green-700 border border-green-300"
                : "bg-red-100 text-red-700 border border-red-300"
            }`}
          >
            {isCompleted ? "✓ Completed" : "✗ Failed"}
          </span>
        </div>
        <div className="space-y-2">
          <div className="space-y-1 text-sm">
            <div>
              <span className="text-slate-600">Model:</span> <span className="text-slate-900 font-mono">{entry.model_id}</span>
            </div>
            <div>
              <span className="text-slate-600">SAE:</span> <span className="text-slate-900 font-mono">{entry.sae_id}</span>
            </div>
            <div>
              <span className="text-slate-600">Layer:</span> <span className="text-slate-900 font-mono">{entry.layer}</span>
            </div>
            <div>
              <span className="text-slate-600">Top-K:</span> <span className="text-slate-900">{entry.settings.topK}</span>
            </div>
            <div className="mt-2">
              <span className="text-slate-600">Prompts:</span>
              <div className="mt-1 space-y-1">
                {entry.prompts.map((p, idx) => (
                  <div key={idx} className="text-slate-900 bg-slate-50 border border-slate-200 p-2 rounded text-xs">
                    {p}
                  </div>
                ))}
              </div>
            </div>
            <div className="mt-2">
              <span className="text-slate-600">Settings:</span>
              <div className="mt-1 text-slate-700 text-xs">
                Save top texts: {entry.settings.saveTopTexts ? "✓" : "✗"} | Track texts:{" "}
                {entry.settings.trackTexts ? "✓" : "✗"} | Return token latents:{" "}
                {entry.settings.returnTokenLatents ? "✓" : "✗"}
              </div>
              {entry.settings.conceptConfigPath && (
                <div className="mt-1 text-slate-700 text-xs">
                  Concept config: <span className="font-mono">{entry.settings.conceptConfigPath}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {entry.outputs.length > 0 && (
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900">Outputs</h3>
            <div className="space-y-2">
              {entry.outputs.map((o, idx) => (
                <div key={idx} className="bg-slate-50 border border-slate-200 p-3 rounded text-xs">
                  <div className="text-slate-600 mb-1 font-medium">Prompt #{idx + 1}</div>
                  <div className="text-slate-900 whitespace-pre-wrap">{o.text}</div>
                  {o.tokens && o.tokens.length > 0 && (
                    <div className="text-slate-600 mt-1">Tokens: {o.tokens.join(" ")}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {entry.top_neurons.length > 0 && (
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900">Top Neurons</h3>
            <div className="space-y-2">
              {entry.top_neurons.map((t, idx) => (
                <div key={idx} className="bg-slate-50 border border-slate-200 p-3 rounded text-xs">
                  <div className="font-semibold text-slate-900">Prompt #{t.prompt_index + 1}</div>
                  <div className="text-slate-700">Neurons: {t.neuron_ids.join(", ")}</div>
                  <div className="text-slate-600 text-xs">
                    Activations: {t.activations.map((a) => a.toFixed(3)).join(", ")}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {entry.token_latents.length > 0 && (
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900">Token Latents</h3>
            <div className="space-y-2">
              {entry.token_latents.map((p, idx) => (
                <div key={idx} className="bg-slate-50 border border-slate-200 p-3 rounded text-xs">
                  <div className="font-semibold text-slate-900">Prompt #{p.prompt_index + 1}</div>
                  <div className="space-y-1 mt-1">
                    {p.tokens.map((tok, j) => (
                      <div key={j} className="text-slate-700">
                        Token {tok.token_index}: {tok.neuron_ids.join(", ")}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {entry.top_texts_path && (
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900">Top Texts</h3>
            <div className="text-slate-700 text-xs">
              Saved at: <span className="text-amber-600 font-mono">{entry.top_texts_path}</span>
            </div>
          </div>
        )}

        {entry.error && (
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900">Error</h3>
            <div className="bg-red-50 border border-red-200 rounded p-2 text-red-700 text-xs">
              {entry.error}
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}

