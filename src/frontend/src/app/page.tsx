"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";

import { api } from "@/lib/api";
import type { StoreInfo } from "@/lib/types";
import { Button, Card, Input, Label, SectionTitle } from "@/components/ui";

export default function Page() {
  const { data, mutate, error, isLoading } = useSWR<StoreInfo>("/store/info", api.storeInfo);
  const [pendingPath, setPendingPath] = useState<string>("");
  const storePath = pendingPath || data?.artifact_base_path || "";
  const [wandbApiKey, setWandbApiKey] = useState<string>("");
  const [wandbApiKeySaved, setWandbApiKeySaved] = useState(false);

  useEffect(() => {
    // Load wandb API key from localStorage or .env
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("wandbApiKey");
      if (saved) {
        setWandbApiKey(saved);
        setWandbApiKeySaved(true);
      } else {
        // Try to get from environment variable (for Next.js)
        const envKey = process.env.NEXT_PUBLIC_WANDB_API_KEY;
        if (envKey) {
          setWandbApiKey(envKey);
          setWandbApiKeySaved(true);
        }
      }
    }
  }, []);

  const handleSavePath = async () => {
    if (!storePath) return;
    try {
      const updated = await api.setStorePath(storePath);
      setPendingPath("");
      mutate(updated, false);
    } catch (e: unknown) {
      console.error("Failed to update store path", e);
    }
  };

  const handleSaveWandbKey = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("wandbApiKey", wandbApiKey);
      setWandbApiKeySaved(true);
    }
  };

  return (
    <div className="space-y-6">
      <SectionTitle>mi_crow SAE Studio</SectionTitle>
      
      <Card className="space-y-3">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Navigation</h3>
        <div className="space-y-2 text-sm text-slate-600">
          <div>
            <span className="font-medium text-slate-900">Home</span> - Configure store paths and manage settings
          </div>
          <div>
            <span className="font-medium text-slate-900">Activations</span> - Save activations from datasets for selected model layers
          </div>
          <div>
            <span className="font-medium text-slate-900">Training</span> - Train sparse autoencoders (SAEs) using saved activations
          </div>
          <div>
            <span className="font-medium text-slate-900">Inference</span> - Run inference with SAEs, manipulate concepts, and explore neuron activations
          </div>
        </div>
      </Card>
      
      <Card className="space-y-3">
        <p className="text-sm text-slate-600">
          Configure where activations, SAEs and concepts are stored on disk. Existing runs will remain at their original
          paths; new runs will use the path below.
        </p>
        <div className="space-y-2">
          <Label>Artifact store path</Label>
          <div className="flex gap-2">
            <Input
              value={storePath}
              onChange={(e) => setPendingPath(e.target.value)}
              placeholder="/path/to/mi_crow_store"
            />
            <Button onClick={handleSavePath}>Save</Button>
          </div>
          {isLoading && <p className="text-xs text-slate-600">Loading current path…</p>}
          {error && <p className="text-xs text-rose-400">Failed to load store info.</p>}
        </div>
      </Card>

      <Card className="space-y-3">
        <p className="text-sm text-slate-600">
          Configure Weights & Biases (wandb) API key for training experiment tracking. The key can be loaded from
          <code className="text-xs bg-slate-100 px-1 rounded">NEXT_PUBLIC_WANDB_API_KEY</code> environment variable or set manually below.
        </p>
        <div className="space-y-2">
          <Label>Wandb API Key</Label>
          <div className="flex gap-2">
            <Input
              type="password"
              value={wandbApiKey}
              onChange={(e) => {
                setWandbApiKey(e.target.value);
                setWandbApiKeySaved(false);
              }}
              placeholder="wandb_api_key_here"
            />
            <Button onClick={handleSaveWandbKey} disabled={!wandbApiKey}>
              {wandbApiKeySaved ? "Saved" : "Save"}
            </Button>
          </div>
          {wandbApiKeySaved && (
            <p className="text-xs text-green-600">✓ Wandb API key saved to localStorage</p>
          )}
          <p className="text-xs text-slate-500">
            This key will be used when training SAEs with wandb enabled. You can get your API key from{" "}
            <a
              href="https://wandb.ai/settings"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              wandb.ai/settings
            </a>
          </p>
        </div>
      </Card>
    </div>
  );
}
