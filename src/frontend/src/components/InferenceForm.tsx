"use client";

import { Button, Input, Label, TextArea } from "./ui";
import type { InferenceState, InferenceStateActions } from "@/hooks/useInferenceState";

type InferenceFormProps = InferenceState &
  InferenceStateActions & {
    disabled?: boolean;
    onRun: () => void;
    isRunning: boolean;
  };

export function InferenceForm({
  prompts,
  setPrompts,
  disabled,
  onRun,
  isRunning,
}: InferenceFormProps) {
  return (
    <div className={disabled ? "opacity-50 pointer-events-none space-y-4" : "space-y-4"}>
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

      <Button onClick={onRun} disabled={disabled || isRunning}>
        {isRunning ? "Running inference..." : "Run inference"}
      </Button>
    </div>
  );
}

