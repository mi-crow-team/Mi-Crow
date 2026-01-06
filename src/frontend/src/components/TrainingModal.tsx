"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";
import { SaeRunInfo } from "@/lib/types";
import { api } from "@/lib/api";
import { Modal, Spinner } from "./ui";

type TrainingModalProps = {
  sae: SaeRunInfo;
  onClose: () => void;
};

export function TrainingModal({ sae, onClose }: TrainingModalProps) {
  const { data: metadata, isLoading, error } = useSWR<any>(
    sae.model_id && sae.sae_id ? `/sae/saes/${sae.sae_id}/metadata?model_id=${sae.model_id}` : null,
    () => api.getSaeMetadata(sae.model_id, sae.sae_id)
  );

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return "-";
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <Modal title={`SAE Training Details - ${sae.sae_id}`} onClose={onClose}>
      <div className="space-y-4">
        <div className="space-y-2">
          <h3 className="font-semibold text-slate-900">Basic Information</h3>
          <div className="space-y-1 text-sm">
            <div>
              <span className="text-slate-600">SAE ID:</span> <span className="text-slate-900 font-mono">{sae.sae_id}</span>
            </div>
            <div>
              <span className="text-slate-600">Model ID:</span> <span className="text-slate-900">{sae.model_id}</span>
            </div>
            <div>
              <span className="text-slate-600">Layer:</span> <span className="text-slate-900">{sae.layer ?? "-"}</span>
            </div>
            <div>
              <span className="text-slate-600">SAE Class:</span> <span className="text-slate-900">{sae.sae_class ?? "SAE"}</span>
            </div>
            {sae.created_at && (
              <div>
                <span className="text-slate-600">Created:</span> <span className="text-slate-900">{formatTimestamp(sae.created_at)}</span>
              </div>
            )}
          </div>
        </div>

        {sae.sae_path && (
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900">Paths</h3>
            <div className="space-y-1 text-sm">
              <div>
                <span className="text-slate-600">SAE Path:</span>
                <div className="text-slate-900 font-mono text-xs mt-1 break-all bg-slate-50 border border-slate-200 p-2 rounded">
                  {sae.sae_path}
                </div>
              </div>
              {sae.metadata_path && (
                <div>
                  <span className="text-slate-600">Metadata Path:</span>
                  <div className="text-slate-900 font-mono text-xs mt-1 break-all bg-slate-50 border border-slate-200 p-2 rounded">
                    {sae.metadata_path}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="space-y-2">
          <h3 className="font-semibold text-slate-900">Status</h3>
          <div className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-700 border border-green-300 inline-block">
            ✓ Training Completed
          </div>
        </div>

        {isLoading && (
          <div className="flex items-center gap-2 text-slate-600">
            <Spinner className="text-slate-600" />
            <span>Loading training details...</span>
          </div>
        )}

        {error && (
          <div className="text-xs text-red-600 bg-red-50 border border-red-200 p-2 rounded">
            Could not load training metadata: {error.message || "Unknown error"}
          </div>
        )}

        {metadata && (
          <>
            {metadata.training && (
              <div className="space-y-2">
                <h3 className="font-semibold text-slate-900">Training Details</h3>
                <div className="space-y-1 text-sm">
                  {metadata.training.duration_sec && (
                    <div>
                      <span className="text-slate-600">Duration:</span>{" "}
                      <span className="text-slate-900">
                        {Math.round(metadata.training.duration_sec)} seconds (
                        {Math.round(metadata.training.duration_sec / 60)} minutes)
                      </span>
                    </div>
                  )}
                  {metadata.training.wandb_url && (
                    <div>
                      <span className="text-slate-600">Weights & Biases:</span>{" "}
                      <a
                        href={metadata.training.wandb_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 underline"
                      >
                        View run on wandb.ai
                      </a>
                    </div>
                  )}
                  {metadata.training.config && (
                    <div className="space-y-1">
                      <span className="text-slate-600">Training Config:</span>
                      <div className="bg-slate-50 border border-slate-200 p-2 rounded text-xs text-slate-900 font-mono overflow-x-auto">
                        <pre>{JSON.stringify(metadata.training.config, null, 2)}</pre>
                      </div>
                    </div>
                  )}
                  {metadata.training.result && (
                    <div className="space-y-1">
                      <span className="text-slate-600">Training Results:</span>
                      <div className="bg-slate-50 border border-slate-200 p-2 rounded text-xs text-slate-900 font-mono overflow-x-auto">
                        <pre>{JSON.stringify(metadata.training.result, null, 2)}</pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {metadata.activations_path && (
              <div className="space-y-2">
                <h3 className="font-semibold text-slate-900">Source</h3>
                <div className="text-sm">
                  <span className="text-slate-600">Activations Path:</span>
                  <div className="text-slate-900 font-mono text-xs mt-1 break-all bg-slate-50 border border-slate-200 p-2 rounded">
                    {metadata.activations_path}
                  </div>
                </div>
              </div>
            )}

            {metadata.manifest && (
              <div className="space-y-2">
                <h3 className="font-semibold text-slate-900">Manifest</h3>
                <div className="bg-slate-50 border border-slate-200 p-2 rounded text-xs text-slate-900 font-mono overflow-x-auto">
                  <pre>{JSON.stringify(metadata.manifest, null, 2)}</pre>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Modal>
  );
}

