"use client";

import { useState, useEffect } from "react";
import { ConceptData } from "@/lib/types";
import { api } from "@/lib/api";
import { Button, Card, Label } from "./ui";

type ConceptManipulatorProps = {
  modelId: string;
  saeId: string;
  onEditsChange: (edits: Record<string, number>) => void;
  onBiasChange: (bias: Record<string, number>) => void;
  onPreview: () => void;
  onSaveConfig: () => void;
};

export function ConceptManipulator({
  modelId,
  saeId,
  onEditsChange,
  onBiasChange,
  onPreview,
  onSaveConfig,
}: ConceptManipulatorProps) {
  const [concepts, setConcepts] = useState<ConceptData[]>([]);
  const [selectedConcepts, setSelectedConcepts] = useState<Set<number>>(new Set());
  const [conceptWeights, setConceptWeights] = useState<Record<number, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (modelId && saeId) {
      loadConcepts();
    } else {
      setConcepts([]);
      setSelectedConcepts(new Set());
      setConceptWeights({});
    }
  }, [modelId, saeId]);

  const loadConcepts = async () => {
    if (!modelId || !saeId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getConceptDictionary(modelId, saeId);
      const conceptList: ConceptData[] = [];
      for (const [neuronIdxStr, concept] of Object.entries(data.concepts)) {
        conceptList.push({
          neuron_index: parseInt(neuronIdxStr),
          name: concept.name,
          score: concept.score,
        });
      }
      conceptList.sort((a, b) => a.neuron_index - b.neuron_index);
      setConcepts(conceptList);
    } catch (e: unknown) {
      const error = e instanceof Error ? e.message : String(e);
      setError(`Failed to load concepts: ${error}`);
      setConcepts([]);
    } finally {
      setLoading(false);
    }
  };

  const toggleConcept = (neuronIndex: number) => {
    const newSelected = new Set(selectedConcepts);
    if (newSelected.has(neuronIndex)) {
      newSelected.delete(neuronIndex);
      const newWeights = { ...conceptWeights };
      delete newWeights[neuronIndex];
      setConceptWeights(newWeights);
    } else {
      newSelected.add(neuronIndex);
      if (!conceptWeights[neuronIndex]) {
        setConceptWeights({ ...conceptWeights, [neuronIndex]: 1.0 });
      }
    }
    setSelectedConcepts(newSelected);
    updateEditsAndBias(newSelected, conceptWeights);
  };

  const updateWeight = (neuronIndex: number, weight: number) => {
    const newWeights = { ...conceptWeights, [neuronIndex]: weight };
    setConceptWeights(newWeights);
    updateEditsAndBias(selectedConcepts, newWeights);
  };

  const updateEditsAndBias = (selected: Set<number>, weights: Record<number, number>) => {
    const edits: Record<string, number> = {};
    const bias: Record<string, number> = {};
    selected.forEach((neuronIdx) => {
      const weight = weights[neuronIdx] ?? 1.0;
      edits[neuronIdx.toString()] = weight;
      bias[neuronIdx.toString()] = 0.0;
    });
    onEditsChange(edits);
    onBiasChange(bias);
  };

  if (!modelId || !saeId) {
    return (
      <Card className="p-4 bg-slate-50">
        <div className="text-sm text-slate-600">Select a model and SAE to load concepts.</div>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card className="p-4 bg-slate-50">
        <div className="text-sm text-slate-600">Loading concepts...</div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4 bg-red-50 border-red-200">
        <div className="text-sm text-red-600">{error}</div>
      </Card>
    );
  }

  return (
    <Card className="space-y-4">
      <div>
        <Label>Select Concepts to Manipulate</Label>
        <div className="mt-2 max-h-64 overflow-y-auto border border-slate-200 rounded-md p-2 space-y-1">
          {concepts.length === 0 ? (
            <div className="text-xs text-slate-600">No concepts found in dictionary.</div>
          ) : (
            concepts.map((concept) => (
              <button
                key={concept.neuron_index}
                type="button"
                onClick={() => toggleConcept(concept.neuron_index)}
                className={`w-full text-left px-3 py-2 rounded-md text-sm transition ${
                  selectedConcepts.has(concept.neuron_index)
                    ? "bg-mi_crow-100 border-2 border-mi_crow-400 text-mi_crow-900"
                    : "bg-white border border-slate-200 hover:bg-slate-50 text-slate-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{concept.name}</div>
                    <div className="text-xs text-slate-600">
                      Neuron {concept.neuron_index} (score: {concept.score.toFixed(3)})
                    </div>
                  </div>
                  {selectedConcepts.has(concept.neuron_index) && (
                    <div className="text-mi_crow-600">✓</div>
                  )}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {selectedConcepts.size > 0 && (
        <div className="space-y-3">
          <Label>Adjust Weights for Selected Concepts</Label>
          {Array.from(selectedConcepts)
            .sort()
            .map((neuronIdx) => {
              const concept = concepts.find((c) => c.neuron_index === neuronIdx);
              return (
                <div key={neuronIdx} className="space-y-2 p-3 bg-mi_crow-50 rounded-md border border-mi_crow-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-sm text-slate-900">
                        {concept?.name || `Neuron ${neuronIdx}`}
                      </div>
                      <div className="text-xs text-slate-600">Neuron {neuronIdx}</div>
                    </div>
                    <div className="text-sm font-mono text-slate-700">
                      {conceptWeights[neuronIdx]?.toFixed(2) ?? "1.00"}
                    </div>
                  </div>
                  <input
                    type="range"
                    min={-2}
                    max={2}
                    step={0.1}
                    value={conceptWeights[neuronIdx] ?? 1.0}
                    onChange={(e) => updateWeight(neuronIdx, Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              );
            })}
        </div>
      )}

      {selectedConcepts.size > 0 && (
        <div className="flex gap-2 pt-2">
          <Button variant="primary" onClick={onPreview}>
            Preview
          </Button>
          <Button variant="ghost" onClick={onSaveConfig}>
            Save Config
          </Button>
        </div>
      )}
    </Card>
  );
}

