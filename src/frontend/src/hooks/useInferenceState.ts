import { useState, useMemo } from "react";
import type { SaeRunInfo } from "@/lib/types";

type Prompt = { prompt: string };

export interface InferenceState {
  saeId: string;
  layer: string;
  prompts: Prompt[];
  topK: number;
  saveTopTexts: boolean;
  trackTexts: boolean;
  returnTokenLatents: boolean;
  conceptConfigPath: string;
  loadConcepts: boolean;
}

export interface InferenceStateActions {
  setSaeId: (id: string) => void;
  setLayer: (layer: string) => void;
  setPrompts: (prompts: Prompt[]) => void;
  setTopK: (k: number) => void;
  setSaveTopTexts: (save: boolean) => void;
  setTrackTexts: (track: boolean) => void;
  setReturnTokenLatents: (returnLatents: boolean) => void;
  setConceptConfigPath: (path: string) => void;
  setLoadConcepts: (load: boolean) => void;
}

export function useInferenceState(
  saes: { saes: SaeRunInfo[] } | undefined,
  initialSaeId?: string
): InferenceState & InferenceStateActions {
  const [saeId, setSaeId] = useState(initialSaeId || "");
  const [layer, setLayer] = useState("");
  const [prompts, setPrompts] = useState<Prompt[]>([{ prompt: "Hello world" }]);
  const [topK, setTopK] = useState(5);
  const [saveTopTexts, setSaveTopTexts] = useState(false);
  const [trackTexts, setTrackTexts] = useState(false);
  const [returnTokenLatents, setReturnTokenLatents] = useState(false);
  const [conceptConfigPath, setConceptConfigPath] = useState("");
  const [loadConcepts, setLoadConcepts] = useState(false);

  // Auto-set first SAE if available
  useMemo(() => {
    if (saes?.saes?.length && !saeId) {
      const first = saes.saes[0];
      setSaeId(first.sae_id);
    }
  }, [saes, saeId]);

  // Auto-set layer from selected SAE
  const currentSae = useMemo(() => {
    return saes?.saes?.find((s) => s.sae_id === saeId);
  }, [saes, saeId]);

  useMemo(() => {
    if (currentSae?.layer) {
      setLayer(currentSae.layer);
    } else {
      setLayer("");
    }
  }, [currentSae]);

  return {
    saeId,
    layer,
    prompts,
    topK,
    saveTopTexts,
    trackTexts,
    returnTokenLatents,
    conceptConfigPath,
    loadConcepts,
    setSaeId,
    setLayer,
    setPrompts,
    setTopK,
    setSaveTopTexts,
    setTrackTexts,
    setReturnTokenLatents,
    setConceptConfigPath,
    setLoadConcepts,
  };
}

