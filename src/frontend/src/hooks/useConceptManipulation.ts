import { useState } from "react";
import { api } from "@/lib/api";
import type { ConceptPreviewResponse, ConceptManipulationResponse } from "@/lib/api/types";

export function useConceptManipulation() {
  const [conceptEdits, setConceptEdits] = useState<Record<string, number>>({});
  const [conceptBias, setConceptBias] = useState<Record<string, number>>({});
  const [conceptPreview, setConceptPreview] = useState<string>("");
  const [conceptSaveResult, setConceptSaveResult] = useState<string>("");

  const previewConcept = async (modelId: string, saeId: string) => {
    try {
      const res: ConceptPreviewResponse = await api.previewConcept({
        model_id: modelId,
        sae_id: saeId,
        edits: conceptEdits,
        bias: conceptBias,
      });
      setConceptPreview(JSON.stringify(res, null, 2));
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : "Unknown error";
      setConceptPreview(`Error: ${error}`);
    }
  };

  const saveConceptConfig = async (
    modelId: string,
    saeId: string,
    onSuccess?: (configPath: string) => void
  ) => {
    try {
      const res: ConceptManipulationResponse = await api.manipulateConcept({
        model_id: modelId,
        sae_id: saeId,
        edits: conceptEdits,
        bias: conceptBias,
      });
      setConceptSaveResult(`Saved config at ${res.concept_config_path}`);
      if (onSuccess) {
        onSuccess(res.concept_config_path);
      }
      return res.concept_config_path;
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : "Unknown error";
      setConceptSaveResult(`Error: ${error}`);
      return null;
    }
  };

  const clearConceptManipulation = () => {
    setConceptEdits({});
    setConceptBias({});
    setConceptPreview("");
    setConceptSaveResult("");
  };

  return {
    conceptEdits,
    conceptBias,
    conceptPreview,
    conceptSaveResult,
    setConceptEdits,
    setConceptBias,
    previewConcept,
    saveConceptConfig,
    clearConceptManipulation,
  };
}

