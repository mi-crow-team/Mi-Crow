export type ModelInfo = { id: string; name: string; status: string };
export type LayerInfo = { layer_id: string; name: string; type: string; path: string[] };

export type ActivationRunInfo = {
  model_id: string;
  run_id: string;
  manifest_path?: string;
  samples?: number;
  tokens?: number;
  layers: string[];
  dataset: Record<string, any>;
  created_at?: string;
  status?: string;
};

export type SaeRunInfo = {
  model_id: string;
  sae_id: string;
  sae_path?: string;
  metadata_path?: string;
  sae_class?: string;
  layer?: string;
  created_at?: string;
};

export type StoreInfo = {
  artifact_base_path: string;
  activation_datasets: Record<string, ActivationRunInfo[]>;
};

export type TrainJobStatus = {
  job_id: string;
  status: string;
  sae_id?: string;
  sae_path?: string;
  metadata_path?: string;
  progress?: number | null;
  logs?: string[];
  error?: string | null;
};

export type InferenceOutput = {
  text: string;
  tokens: string[];
  logits?: number[] | null;
  probabilities?: number[] | null;
  hooks?: any[];
  timing_ms?: number | null;
};

export type TopNeurons = {
  prompt_index: number;
  prompt: string;
  neuron_ids: number[];
  activations: number[];
};

export type TokenLatents = {
  prompt_index: number;
  tokens: { token_index: number; neuron_ids: number[]; activations: number[] }[];
};

export type ConceptConfigInfo = {
  name: string;
  path: string;
  created_at?: string;
  sae_id?: string;
  layer?: string;
};

export type ConceptData = {
  neuron_index: number;
  name: string;
  score: number;
};

export type InferenceHistoryEntry = {
  id: string;
  timestamp: string;
  model_id: string;
  sae_id: string;
  layer: string;
  prompts: string[];
  outputs: InferenceOutput[];
  top_neurons: TopNeurons[];
  token_latents: TokenLatents[];
  top_texts_path?: string;
  status?: "completed" | "failed";
  error?: string;
  settings: {
    saveTopTexts: boolean;
    trackTexts: boolean;
    returnTokenLatents: boolean;
    conceptConfigPath?: string;
    topK: number;
  };
};

export type InferenceRunConfig = {
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
