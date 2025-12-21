const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const getHeaders = () => {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (typeof window !== "undefined") {
    const key = localStorage.getItem("apiKey");
    if (key) headers["X-API-Key"] = key;
  }
  return headers;
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...getHeaders(), ...(options.headers || {}) },
  });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = res.statusText;
    }
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  if (res.status === 204) return {} as T;
  return (await res.json()) as T;
}

export const api = {
  models: () => request("/models"),
  loadModel: (model_id: string) =>
    request("/models/load", { method: "POST", body: JSON.stringify({ model_id }) }),
  layers: (model_id: string) => request(`/models/${model_id}/layers`),
  saveActivations: (payload: any) =>
    request("/sae/activations/save", { method: "POST", body: JSON.stringify(payload) }),
  listActivations: (model_id: string) => request(`/sae/activations?model_id=${model_id}`),
  train: (payload: any) => request("/sae/train", { method: "POST", body: JSON.stringify(payload) }),
  trainStatus: (job_id: string) => request(`/sae/train/status/${job_id}`),
  cancelTrain: (job_id: string) => request(`/sae/train/cancel/${job_id}`, { method: "POST" }),
  listSaes: (model_id: string) => request(`/sae/saes?model_id=${model_id}`),
  getSaeMetadata: (model_id: string, sae_id: string) =>
    request(`/sae/saes/${sae_id}/metadata?model_id=${model_id}`),
  infer: (payload: any) => request("/sae/infer", { method: "POST", body: JSON.stringify(payload) }),
  listConcepts: (model_id: string, sae_id?: string) =>
    request(`/sae/concepts?model_id=${model_id}${sae_id ? `&sae_id=${sae_id}` : ""}`),
  getConceptDictionary: (model_id: string, sae_id: string) =>
    request(`/sae/concepts/dictionary?model_id=${model_id}${sae_id ? `&sae_id=${sae_id}` : ""}`),
  listConceptConfigs: (model_id: string, sae_id?: string) =>
    request(`/sae/concepts/configs?model_id=${model_id}${sae_id ? `&sae_id=${sae_id}` : ""}`),
  loadConcept: (payload: any) => request("/sae/concepts/load", { method: "POST", body: JSON.stringify(payload) }),
  previewConcept: (payload: any) => request("/sae/concepts/preview", { method: "POST", body: JSON.stringify(payload) }),
  manipulateConcept: (payload: any) =>
    request("/sae/concepts/manipulate", { method: "POST", body: JSON.stringify(payload) }),
  deleteActivation: (model_id: string, run_id: string) =>
    request(`/sae/activations/${run_id}?model_id=${model_id}`, { method: "DELETE" }),
  deleteSae: (model_id: string, sae_id: string) =>
    request(`/sae/saes/${sae_id}?model_id=${model_id}`, { method: "DELETE" }),
  health: () => request("/health/metrics"),
  storeInfo: () => request("/store/info"),
  setStorePath: (artifact_base_path: string) =>
    request("/store/path", { method: "POST", body: JSON.stringify({ artifact_base_path }) }),
  saeClasses: () => request<Record<string, string>>("/sae/classes"),
  getLayerSize: (activations_path: string, layer: string) =>
    request<{ layer: string; hidden_dim: number }>(`/sae/train/layer-size?activations_path=${encodeURIComponent(activations_path)}&layer=${encodeURIComponent(layer)}`),
};

export type ApiClient = typeof api;

