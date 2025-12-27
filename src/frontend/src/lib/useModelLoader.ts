import { useEffect, useState } from "react";
import useSWR from "swr";

import { api } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";

export function useModelLoader() {
  const { data: models } = useSWR<ModelInfo[]>("/models", api.models);
  const [modelId, setModelId] = useState<string>("");
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Prefer Bielik as default model if present, otherwise fall back to first model
  useEffect(() => {
    if (models && models.length && !modelId) {
      const preferredId = "speakleash/Bielik-1.5B-v3.0-Instruct";
      const preferred = models.find((m) => m.id === preferredId);
      setModelId((preferred ?? models[0]).id);
      setModelLoaded(false);
    }
  }, [models, modelId]);

  const loadModel = async () => {
    if (!modelId) return;
    setIsLoading(true);
    try {
      await api.loadModel(modelId);
      setModelLoaded(true);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    models,
    modelId,
    setModelId,
    modelLoaded,
    setModelLoaded,
    loadModel,
    isLoading,
  };
}


