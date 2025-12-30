"use client";

import { useState } from "react";
import { InferenceHistoryEntry } from "@/lib/types";
import { HistoryModal } from "./HistoryModal";

type Tab = "history" | "settings";

type InferenceSidebarProps = {
  history: InferenceHistoryEntry[];
  onSelectHistory: (entry: InferenceHistoryEntry) => void;
  selectedHistoryEntry: InferenceHistoryEntry | null;
  onCloseHistoryModal: () => void;
  settings: {
    loadConcepts: boolean;
    saveTopTexts: boolean;
    trackTexts: boolean;
  };
  onSettingsChange: (settings: { loadConcepts: boolean; saveTopTexts: boolean; trackTexts: boolean }) => void;
};

export function InferenceSidebar({
  history,
  onSelectHistory,
  selectedHistoryEntry,
  onCloseHistoryModal,
  settings,
  onSettingsChange,
}: InferenceSidebarProps) {
  const [activeTab, setActiveTab] = useState<Tab>("history");

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <>
      <div className="space-y-4">
        <div className="flex gap-2 border-b border-slate-200 pb-2">
          <button
            onClick={() => setActiveTab("history")}
            className={`flex-1 px-3 py-1.5 text-sm rounded-md transition ${
              activeTab === "history"
                ? "bg-mi_crow-100 text-mi_crow-900 font-medium"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            History
          </button>
          <button
            onClick={() => setActiveTab("settings")}
            className={`flex-1 px-3 py-1.5 text-sm rounded-md transition ${
              activeTab === "settings"
                ? "bg-mi_crow-100 text-mi_crow-900 font-medium"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            Settings
          </button>
        </div>

        {activeTab === "history" && (
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-mi_crow-600 mb-2">
              Past Runs ({history.length})
            </div>
            {history.length === 0 ? (
              <p className="text-xs text-slate-600">No history yet. Run some inferences to see them here.</p>
            ) : (
              <div className="space-y-2 max-h-[60vh] overflow-y-auto">
                {history.map((entry) => {
                  const status = entry.status || (entry.outputs.length > 0 ? "completed" : "failed");
                  const isCompleted = status === "completed";
                  return (
                    <button
                      key={entry.id}
                      onClick={() => onSelectHistory(entry)}
                      className="w-full text-left p-2 rounded-md border border-slate-200 bg-white hover:bg-mi_crow-50 hover:border-mi_crow-300 transition text-xs"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="font-medium text-slate-900 truncate">
                          <span className="text-slate-600">Model:</span> <span className="text-slate-900">{entry.model_id}</span> / <span className="text-slate-600">SAE:</span> <span className="text-slate-900">{entry.sae_id}</span>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium ${
                            isCompleted
                              ? "bg-green-100 text-green-700"
                              : "bg-red-100 text-red-700"
                          }`}
                        >
                          {isCompleted ? "✓ Completed" : "✗ Failed"}
                        </span>
                      </div>
                      <div className="text-slate-900 text-xs mt-1 font-mono font-semibold">{entry.id}</div>
                      <div className="text-slate-700 text-xs mt-1">{formatTimestamp(entry.timestamp)}</div>
                      <div className="text-slate-700 text-xs mt-1">
                        {entry.prompts.length} prompt{entry.prompts.length !== 1 ? "s" : ""}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === "settings" && (
          <div className="space-y-4">
            <div className="text-xs font-semibold uppercase tracking-wide text-mi_crow-600 mb-2">Options</div>
            <div className="space-y-3">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.loadConcepts}
                  onChange={(e) => onSettingsChange({ ...settings, loadConcepts: e.target.checked })}
                  className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
                />
                <span className="text-slate-700">Load concepts</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.saveTopTexts}
                  onChange={(e) => onSettingsChange({ ...settings, saveTopTexts: e.target.checked })}
                  className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
                />
                <span className="text-slate-700">Save top texts</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.trackTexts}
                  onChange={(e) => onSettingsChange({ ...settings, trackTexts: e.target.checked })}
                  className="rounded border-slate-300 text-mi_crow-600 focus:ring-mi_crow-500"
                />
                <span className="text-slate-700">Track texts</span>
              </label>
            </div>
          </div>
        )}
      </div>

      {selectedHistoryEntry && (
        <HistoryModal entry={selectedHistoryEntry} onClose={onCloseHistoryModal} />
      )}
    </>
  );
}

