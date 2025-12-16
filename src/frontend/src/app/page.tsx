"use client";

import { useState } from "react";
import useSWR from "swr";

import { api } from "@/lib/api";
import type { ActivationRunInfo, StoreInfo } from "@/lib/types";
import { Button, Card, Input, Label, SectionTitle } from "@/components/ui";

export default function Page() {
  const { data, mutate, error, isLoading } = useSWR<StoreInfo>("/store/info", api.storeInfo);
  const [pendingPath, setPendingPath] = useState<string>("");
  const storePath = pendingPath || data?.artifact_base_path || "";

  const handleSavePath = async () => {
    if (!storePath) return;
    try {
      const updated = await api.setStorePath(storePath);
      setPendingPath("");
      mutate(updated, false);
    } catch (e: any) {
      console.error("Failed to update store path", e);
    }
  };

  const hasDatasets = !!data && Object.keys(data.activation_datasets || {}).length > 0;

  return (
    <div className="space-y-6">
      <SectionTitle>Amber SAE UI</SectionTitle>
      <Card className="space-y-3">
        <p className="text-sm text-slate-500">
          Configure where activations, SAEs and concepts are stored on disk. Existing runs will remain at their original
          paths; new runs will use the path below.
        </p>
        <div className="space-y-2">
          <Label>Artifact store path</Label>
          <div className="flex gap-2">
            <Input
              value={storePath}
              onChange={(e) => setPendingPath(e.target.value)}
              placeholder="/path/to/amber_store"
            />
            <Button onClick={handleSavePath}>Save</Button>
          </div>
          {isLoading && <p className="text-xs text-slate-500">Loading current path…</p>}
          {error && <p className="text-xs text-rose-400">Failed to load store info.</p>}
        </div>
      </Card>

      <Card className="space-y-3">
        <SectionTitle>Local datasets & activation runs</SectionTitle>
        {!hasDatasets && <p className="text-sm text-slate-400">No activation runs found yet for this store path.</p>}
        {hasDatasets && (
          <div className="space-y-4 text-sm">
            {Object.entries(data!.activation_datasets).map(([modelId, runs]) => (
              <div key={modelId} className="space-y-1">
                <div className="font-semibold text-slate-100">Model: {modelId}</div>
                <RunList runs={runs} />
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function RunList({ runs }: { runs: ActivationRunInfo[] }) {
  if (!runs.length) return null;
  return (
    <div className="space-y-2">
      {runs.map((r) => (
        <div key={r.run_id} className="border border-slate-800 rounded-md p-2">
          <div className="flex justify-between">
            <div className="font-medium text-slate-100">{r.run_id}</div>
            <div className="text-slate-400">
              samples: {r.samples ?? "-"} | tokens: {r.tokens ?? "-"}
            </div>
          </div>
          <div className="text-xs text-slate-500">
            dataset: {r.dataset?.type === "hf" ? r.dataset.name : "local"}
          </div>
        </div>
      ))}
    </div>
  );
}

