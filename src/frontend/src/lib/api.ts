import {
  ApiError,
  NetworkError,
  NotFoundError,
  ServerError,
  UnauthorizedError,
  ValidationError,
} from "./api/errors";
import type {
  ActivationRunListResponse,
  ConceptConfigListResponse,
  ConceptDictionaryResponse,
  ConceptLoadRequest,
  ConceptLoadResponse,
  ConceptListResponse,
  ConceptManipulationRequest,
  ConceptManipulationResponse,
  ConceptPreviewResponse,
  LayerInfo,
  LayerSizeInfo,
  ModelInfo,
  SAEInferenceRequest,
  SAEInferenceResponse,
  SaeRunListResponse,
  SaveActivationsRequest,
  SaveActivationsResponse,
  TrainSAERequest,
  TrainSAEResponse,
} from "./api/types";
import type { StoreInfo, TrainJobStatus } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const getHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (typeof window !== "undefined") {
    const key = localStorage.getItem("apiKey");
    if (key) headers["X-API-Key"] = key;
  }
  return headers;
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: { ...getHeaders(), ...(options.headers || {}) },
    });

    if (!res.ok) {
      let detail = "";
      let body: unknown = null;
      try {
        body = await res.json();
        if (typeof body === "object" && body !== null && "detail" in body) {
          detail = String(body.detail);
        } else {
          detail = JSON.stringify(body);
        }
      } catch {
        detail = res.statusText || `Request failed with status ${res.status}`;
      }

      // Create appropriate error based on status code
      switch (res.status) {
        case 400:
          throw new ValidationError(detail || "Validation error", body);
        case 401:
          throw new UnauthorizedError(detail || "Unauthorized", body);
        case 404:
          throw new NotFoundError(detail || "Resource not found", body);
        case 500:
        default:
          throw new ServerError(detail || `Server error: ${res.status}`, body);
      }
    }

    if (res.status === 204) return {} as T;
    return (await res.json()) as T;
  } catch (error) {
    // Re-throw ApiError instances
    if (error instanceof ApiError) {
      throw error;
    }
    // Handle network errors
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new NetworkError("Network request failed");
    }
    // Wrap other errors
    throw new ApiError(error instanceof Error ? error.message : "Unknown error occurred");
  }
}

export const api = {
  models: (): Promise<ModelInfo[]> => request<ModelInfo[]>("/models"),
  loadModel: (model_id: string): Promise<ModelInfo> =>
    request<ModelInfo>("/models/load", { method: "POST", body: JSON.stringify({ model_id }) }),
  layers: (model_id: string): Promise<LayerInfo[]> => request<LayerInfo[]>(`/models/${model_id}/layers`),
  saveActivations: (payload: SaveActivationsRequest): Promise<SaveActivationsResponse> =>
    request<SaveActivationsResponse>("/sae/activations/save", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listActivations: (model_id: string): Promise<ActivationRunListResponse> =>
    request<ActivationRunListResponse>(`/sae/activations?model_id=${model_id}`),
  train: (payload: TrainSAERequest): Promise<TrainSAEResponse> =>
    request<TrainSAEResponse>("/sae/train", { method: "POST", body: JSON.stringify(payload) }),
  trainStatus: (job_id: string): Promise<TrainJobStatus> =>
    request<TrainJobStatus>(`/sae/train/status/${job_id}`),
  cancelTrain: (job_id: string): Promise<TrainJobStatus> =>
    request<TrainJobStatus>(`/sae/train/cancel/${job_id}`, { method: "POST" }),
  listSaes: (model_id: string): Promise<SaeRunListResponse> =>
    request<SaeRunListResponse>(`/sae/saes?model_id=${model_id}`),
  getSaeMetadata: (model_id: string, sae_id: string): Promise<Record<string, unknown>> =>
    request<Record<string, unknown>>(`/sae/saes/${sae_id}/metadata?model_id=${model_id}`),
  infer: (payload: SAEInferenceRequest): Promise<SAEInferenceResponse> =>
    request<SAEInferenceResponse>("/sae/infer", { method: "POST", body: JSON.stringify(payload) }),
  listConcepts: (model_id: string, sae_id?: string): Promise<ConceptListResponse> =>
    request<ConceptListResponse>(
      `/sae/concepts?model_id=${model_id}${sae_id ? `&sae_id=${sae_id}` : ""}`
    ),
  getConceptDictionary: (model_id: string, sae_id: string): Promise<ConceptDictionaryResponse> =>
    request<ConceptDictionaryResponse>(
      `/sae/concepts/dictionary?model_id=${model_id}${sae_id ? `&sae_id=${sae_id}` : ""}`
    ),
  listConceptConfigs: (model_id: string, sae_id?: string): Promise<ConceptConfigListResponse> =>
    request<ConceptConfigListResponse>(
      `/sae/concepts/configs?model_id=${model_id}${sae_id ? `&sae_id=${sae_id}` : ""}`
    ),
  loadConcept: (payload: ConceptLoadRequest): Promise<ConceptLoadResponse> =>
    request<ConceptLoadResponse>("/sae/concepts/load", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  previewConcept: (payload: ConceptManipulationRequest): Promise<ConceptPreviewResponse> =>
    request<ConceptPreviewResponse>("/sae/concepts/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  manipulateConcept: (payload: ConceptManipulationRequest): Promise<ConceptManipulationResponse> =>
    request<ConceptManipulationResponse>("/sae/concepts/manipulate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  deleteActivation: (model_id: string, run_id: string): Promise<{ deleted: boolean; run_id: string }> =>
    request<{ deleted: boolean; run_id: string }>(`/sae/activations/${run_id}?model_id=${model_id}`, {
      method: "DELETE",
    }),
  deleteSae: (model_id: string, sae_id: string): Promise<{ deleted: boolean; sae_id: string }> =>
    request<{ deleted: boolean; sae_id: string }>(`/sae/saes/${sae_id}?model_id=${model_id}`, {
      method: "DELETE",
    }),
  health: (): Promise<{ jobs: Record<string, number> }> => request<{ jobs: Record<string, number> }>("/health/metrics"),
  storeInfo: (): Promise<StoreInfo> => request<StoreInfo>("/store/info"),
  setStorePath: (artifact_base_path: string): Promise<StoreInfo> =>
    request<StoreInfo>("/store/path", {
      method: "POST",
      body: JSON.stringify({ artifact_base_path }),
    }),
  saeClasses: (): Promise<Record<string, string>> => request<Record<string, string>>("/sae/classes"),
  getLayerSize: (activations_path: string, layer: string): Promise<LayerSizeInfo> =>
    request<LayerSizeInfo>(
      `/sae/train/layer-size?activations_path=${encodeURIComponent(activations_path)}&layer=${encodeURIComponent(layer)}`
    ),
};

export type ApiClient = typeof api;

