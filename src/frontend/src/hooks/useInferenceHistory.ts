import { useState, useEffect } from "react";
import type { InferenceHistoryEntry } from "@/lib/types";
import { getStorageItem, setStorageItem, removeStorageItem } from "@/lib/storage";

const HISTORY_STORAGE_KEY = "inference_history";
const MAX_HISTORY_ENTRIES = 100;

export function useInferenceHistory() {
  const [history, setHistory] = useState<InferenceHistoryEntry[]>([]);
  const [selectedHistoryEntry, setSelectedHistoryEntry] = useState<InferenceHistoryEntry | null>(null);

  // Load history from localStorage on mount
  useEffect(() => {
    const stored = getStorageItem<InferenceHistoryEntry[]>(HISTORY_STORAGE_KEY, []);
    setHistory(stored);
  }, []);

  const saveToHistory = (entry: InferenceHistoryEntry) => {
    const newHistory = [entry, ...history].slice(0, MAX_HISTORY_ENTRIES);
    setHistory(newHistory);
    setStorageItem(HISTORY_STORAGE_KEY, newHistory);
  };

  const clearHistory = () => {
    setHistory([]);
    removeStorageItem(HISTORY_STORAGE_KEY);
  };

  return {
    history,
    selectedHistoryEntry,
    setSelectedHistoryEntry,
    saveToHistory,
    clearHistory,
  };
}

