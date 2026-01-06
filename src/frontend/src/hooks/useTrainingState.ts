import { useState, useEffect, useMemo } from "react";
import type { ActivationRunInfo } from "@/lib/types";

export type LatentMode = "n_latents" | "expansion_factor";

export interface TrainingState {
  activationRun: string;
  layer: string;
  saeClass: string;
  latentMode: LatentMode;
  nLatents: number | undefined;
  expansionFactor: number | undefined;
  hiddenDim: number | undefined;
  showAdvanced: boolean;
  epochs: number;
  batchSize: number;
  lr: number;
  l1Lambda: number;
  maxBatchesPerEpoch: number | undefined;
  verbose: boolean;
  useAmp: boolean;
  gradAccumSteps: number;
  clipGrad: number;
  monitoring: number;
  memoryEfficient: boolean;
  saeK: number | undefined;
  useWandb: boolean;
  wandbProject: string;
  wandbEntity: string;
  wandbName: string;
}

export interface TrainingStateActions {
  setActivationRun: (run: string) => void;
  setLayer: (layer: string) => void;
  setSaeClass: (cls: string) => void;
  setLatentMode: (mode: LatentMode) => void;
  setNLatents: (n: number | undefined) => void;
  setExpansionFactor: (factor: number | undefined) => void;
  setHiddenDim: (dim: number | undefined) => void;
  setShowAdvanced: (show: boolean) => void;
  setEpochs: (epochs: number) => void;
  setBatchSize: (size: number) => void;
  setLr: (lr: number) => void;
  setL1Lambda: (lambda: number) => void;
  setMaxBatchesPerEpoch: (batches: number | undefined) => void;
  setVerbose: (verbose: boolean) => void;
  setUseAmp: (use: boolean) => void;
  setGradAccumSteps: (steps: number) => void;
  setClipGrad: (clip: number) => void;
  setMonitoring: (monitoring: number) => void;
  setMemoryEfficient: (efficient: boolean) => void;
  setSaeK: (k: number | undefined) => void;
  setUseWandb: (use: boolean) => void;
  setWandbProject: (project: string) => void;
  setWandbEntity: (entity: string) => void;
  setWandbName: (name: string) => void;
}

export function useTrainingState(
  runs: { runs: ActivationRunInfo[] } | undefined,
  modelLoaded: boolean
): TrainingState & TrainingStateActions {
  const [activationRun, setActivationRun] = useState("");
  const [layer, setLayer] = useState("");
  const [saeClass, setSaeClass] = useState("TopKSae");
  const [latentMode, setLatentMode] = useState<LatentMode>("n_latents");
  const [nLatents, setNLatents] = useState<number | undefined>();
  const [expansionFactor, setExpansionFactor] = useState<number | undefined>(1.0);
  const [hiddenDim, setHiddenDim] = useState<number | undefined>();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [epochs, setEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(256);
  const [lr, setLr] = useState(1e-3);
  const [l1Lambda, setL1Lambda] = useState(0.0);
  const [maxBatchesPerEpoch, setMaxBatchesPerEpoch] = useState<number | undefined>();
  const [verbose, setVerbose] = useState(false);
  const [useAmp, setUseAmp] = useState(true);
  const [gradAccumSteps, setGradAccumSteps] = useState(1);
  const [clipGrad, setClipGrad] = useState(1.0);
  const [monitoring, setMonitoring] = useState(1);
  const [memoryEfficient, setMemoryEfficient] = useState(false);
  const [saeK, setSaeK] = useState<number | undefined>();
  const [useWandb, setUseWandb] = useState(false);
  const [wandbProject, setWandbProject] = useState("");
  const [wandbEntity, setWandbEntity] = useState("");
  const [wandbName, setWandbName] = useState("");

  const selectedRun = useMemo(
    () => runs?.runs?.find((r) => r.run_id === activationRun),
    [runs, activationRun]
  );

  // Auto-select first run
  useEffect(() => {
    if (runs?.runs?.length && !activationRun) {
      const first = runs.runs[0];
      setActivationRun(first.run_id);
    }
  }, [runs, activationRun]);

  // Auto-set layer from selected activation run
  useEffect(() => {
    if (selectedRun?.layers?.length) {
      setLayer(selectedRun.layers[0]);
    } else {
      setLayer("");
    }
  }, [selectedRun]);

  return {
    activationRun,
    layer,
    saeClass,
    latentMode,
    nLatents,
    expansionFactor,
    hiddenDim,
    showAdvanced,
    epochs,
    batchSize,
    lr,
    l1Lambda,
    maxBatchesPerEpoch,
    verbose,
    useAmp,
    gradAccumSteps,
    clipGrad,
    monitoring,
    memoryEfficient,
    saeK,
    useWandb,
    wandbProject,
    wandbEntity,
    wandbName,
    setActivationRun,
    setLayer,
    setSaeClass,
    setLatentMode,
    setNLatents,
    setExpansionFactor,
    setHiddenDim,
    setShowAdvanced,
    setEpochs,
    setBatchSize,
    setLr,
    setL1Lambda,
    setMaxBatchesPerEpoch,
    setVerbose,
    setUseAmp,
    setGradAccumSteps,
    setClipGrad,
    setMonitoring,
    setMemoryEfficient,
    setSaeK,
    setUseWandb,
    setWandbProject,
    setWandbEntity,
    setWandbName,
  };
}

