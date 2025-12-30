"use client";

import { Input, Label, Row } from "./ui";
import type { InferenceState, InferenceStateActions } from "@/hooks/useInferenceState";
import type { SaeRunInfo } from "@/lib/types";

type InferenceConfigProps = InferenceState &
  InferenceStateActions & {
    saes?: { saes: SaeRunInfo[] };
    currentSae?: SaeRunInfo;
    disabled?: boolean;
  };

export function InferenceConfig({
  saeId,
  setSaeId,
  layer,
  topK,
  setTopK,
  saveTopTexts,
  setSaveTopTexts,
  trackTexts,
  setTrackTexts,
  returnTokenLatents,
  setReturnTokenLatents,
  conceptConfigPath,
  setConceptConfigPath,
  loadConcepts,
  saes,
  currentSae,
  disabled,
}: InferenceConfigProps) {
  return (
    <div className={disabled ? "opacity-50 pointer-events-none space-y-4" : "space-y-4"}>
      <Row>
        <div className="space-y-1">
          <Label>SAE</Label>
          <select
            className="input"
            value={saeId}
            onChange={(e) => setSaeId(e.target.value)}
            disabled={disabled}
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
          <Input value={layer} disabled readOnly className="bg-slate-50 cursor-not-allowed" />
          {!currentSae?.layer && (
            <p className="text-xs text-slate-500">Layer will be set from selected SAE</p>
          )}
        </div>
      </Row>

      <Row>
        <div className="space-y-1">
          <Label>Top-K neurons</Label>
          <Input
            type="number"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            min={1}
          />
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
              className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
            />
            Save top texts
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={trackTexts}
              onChange={(e) => setTrackTexts(e.target.checked)}
              className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
            />
            Track texts
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={returnTokenLatents}
              onChange={(e) => setReturnTokenLatents(e.target.checked)}
              className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
            />
            Return token latents
          </label>
        </div>
      </div>
    </div>
  );
}

