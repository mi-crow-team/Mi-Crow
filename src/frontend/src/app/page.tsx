"use client";

import { useState } from "react";
import useSWR from "swr";

import { api } from "@/lib/api";
import type { StoreInfo } from "@/lib/types";
import { Button, Card, Input, Label, SectionTitle } from "@/components/ui";

export default function Page() {
  const { data, mutate, error, isLoading } = useSWR<StoreInfo>("/store/info", api.storeInfo);
  const { data: health, mutate: refreshHealth } = useSWR<Record<string, unknown>>("/health/metrics", api.health);
  const [pendingPath, setPendingPath] = useState<string>("");
  const storePath = pendingPath || data?.artifact_base_path || "";

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

  return (
    <div className="space-y-6">
      <SectionTitle>mi_crow SAE Studio</SectionTitle>
      
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

      <SectionTitle>System Health</SectionTitle>
      <Card className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-600">Server health metrics and status</p>
          <Button variant="ghost" onClick={() => refreshHealth()}>
            Refresh
          </Button>
        </div>
        {health ? (
          <div className="bg-slate-50 rounded-md p-4 border border-slate-200">
            <pre className="text-xs text-slate-700 whitespace-pre-wrap overflow-x-auto">
              {JSON.stringify(health, null, 2)}
            </pre>
          </div>
        ) : (
          <p className="text-sm text-slate-600">Loading health metrics...</p>
        )}
      </Card>
    </div>
  );
}
