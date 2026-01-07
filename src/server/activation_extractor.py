from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional

import torch
from datasets import load_dataset

from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
from mi_crow.language_model.utils import move_tensors_to_device, get_device_from_model


class ActivationExtractor:
    def __init__(
        self,
        lm,
        layers: Sequence[str],
        batch_size: int = 4,
        shard_size: int = 64,
    ):
        self.lm = lm
        self.layers = list(layers)
        self.batch_size = batch_size
        self.shard_size = shard_size

    def _register_detectors(self) -> List[str]:
        handles: List[str] = []
        for layer in self.layers:
            detector = LayerActivationDetector(layer_signature=layer)
            handle = self.lm.layers.register_hook(layer_signature=layer, hook=detector)
            handles.append(handle)
        return handles

    def _clear_detectors(self, handles: Sequence[str]) -> None:
        for handle in handles:
            try:
                self.lm.layers.unregister_hook(handle)
            except Exception:
                continue

    def _collect_activation_snapshot(self, detectors) -> Dict[str, torch.Tensor]:
        snapshot: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            layer_detectors = self.lm.layers.get_hooks(layer_signature=layer, hook_type=None)
            if not layer_detectors:
                continue
            det = layer_detectors[-1]
            tensor = det.tensor_metadata.get("activations")
            if tensor is None:
                continue
            snapshot[layer] = tensor.detach().cpu()
            det.clear_captured()
        return snapshot

    def extract(
        self,
        texts: Iterable[str],
        out_dir: Path,
        limit: int | None = None,
        *,
        store=None,
        run_id: Optional[str] = None,
    ) -> Dict[str, any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.lm, "context") and getattr(self.lm.context, "device", None) is not None:
            device = torch.device(self.lm.context.device)
        else:
            device = get_device_from_model(self.lm.model)
        handles = self._register_detectors()
        total = 0
        total_tokens = 0
        shard_index = 0
        shard_paths: List[str] = []
        batches: List[Dict[str, any]] = []
        buffer: List[str] = []

        def batched(iterable, size):
            iterator = iter(iterable)
            while True:
                batch = list(itertools.islice(iterator, size))
                if not batch:
                    break
                yield batch

        def _token_counts(encodings) -> List[int]:
            ids = encodings.get("input_ids")
            mask = encodings.get("attention_mask")
            counts: List[int] = []
            if ids is None:
                return counts
            for i in range(ids.shape[0]):
                if mask is not None:
                    counts.append(int(mask[i].sum().item()))
                else:
                    counts.append(int(ids[i].numel()))
            return counts

        def flush_shard(data: List[str]) -> None:
            nonlocal shard_index, total, total_tokens
            if not data:
                return
            enc = self.lm.tokenizer(data, return_tensors="pt", padding=True, truncation=True)
            enc = move_tensors_to_device(enc, device)
            token_counts = _token_counts(enc)
            with torch.no_grad():
                _ = self.lm.model(**enc)
            snapshot = self._collect_activation_snapshot(handles)
            if not snapshot:
                return
            if store is not None and run_id is not None:
                tensor_metadata = {
                    layer: {"activations": tensor}
                    for layer, tensor in snapshot.items()
                }
                metadata = {
                    "texts": list(data),
                    "token_counts": token_counts,
                    "layers": self.layers,
                }
                store_key = store.put_detector_metadata(
                    run_id=run_id,
                    batch_index=shard_index,
                    metadata=metadata,
                    tensor_metadata=tensor_metadata,
                )
                batches.append(
                    {
                        "batch_index": shard_index,
                        "size": len(data),
                        "token_counts": token_counts,
                        "store_key": store_key,
                    }
                )
            else:
                shard_path = out_dir / f"shard_{shard_index}.pt"
                torch.save({"batch": list(data), "activations": snapshot}, shard_path)
                shard_paths.append(str(shard_path))
            shard_index += 1
            total += len(data)
            total_tokens += sum(token_counts) if token_counts else 0

        try:
            iterable = texts if limit is None else itertools.islice(texts, limit)
            for batch in batched(iterable, self.batch_size):
                buffer.extend(batch)
                if len(buffer) >= self.shard_size:
                    flush_shard(buffer[: self.shard_size])
                    buffer = buffer[self.shard_size :]
            if buffer:
                flush_shard(buffer)
        finally:
            self._clear_detectors(handles)

        manifest_path = out_dir / "manifest.json"
        return {
            "samples": total,
            "tokens": total_tokens,
            "shards": shard_paths,
            "batches": batches,
            "manifest_path": str(manifest_path),
        }


def iter_hf_dataset(name: str, split: str, text_field: str) -> Iterable[str]:
    ds = load_dataset(name, split=split)
    for row in ds:
        if text_field not in row:
            continue
        text = row[text_field]
        if isinstance(text, str):
            yield text


def iter_local_files(paths: Sequence[str]) -> Iterable[str]:
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.is_dir():
            for sub in p.glob("**/*"):
                if sub.is_file():
                    yield from _iter_file(sub)
        else:
            yield from _iter_file(p)


def _iter_file(path: Path) -> Iterable[str]:
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    except Exception:
        return
