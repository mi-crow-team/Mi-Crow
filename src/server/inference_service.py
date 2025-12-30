from __future__ import annotations

import time
from typing import Dict, List, Sequence

import torch
from mi_crow.hooks.hook import Hook
from mi_crow.language_model.utils import get_device_from_model, move_tensors_to_device

from server.hook_factory import HookFactory
from server.schemas import HookPayload, InferenceInput, InferenceOutput


class InferenceService:
    """Runs inference with per-call hooks on a LanguageModel."""

    def __init__(self, hook_factory: HookFactory):
        self._hook_factory = hook_factory

    def _register_hooks(self, lm, hooks_payload: Sequence[HookPayload]) -> List[str]:
        hook_ids: List[str] = []
        for payload in hooks_payload:
            hook = self._hook_factory.create(payload)
            hook_id = lm.layers.register_hook(payload.layer_id, hook)
            hook_ids.append(hook_id)
        return hook_ids

    def _cleanup_hooks(self, lm, hook_ids: Sequence[str]) -> None:
        for hook_id in hook_ids:
            try:
                lm.layers.unregister_hook(hook_id)
            except Exception:
                continue

    def _serialize_hook(self, hook: Hook, layer_id: str) -> Dict:
        metadata = getattr(hook, "metadata", {}) or {}
        tensor_metadata = getattr(hook, "tensor_metadata", {}) or {}

        serialized_tensors: Dict[str, List[int]] = {}
        for name, tensor in tensor_metadata.items():
            if tensor is None:
                continue
            try:
                serialized_tensors[name] = list(tensor.shape)
            except Exception:
                serialized_tensors[name] = []

        return {
            "hook_name": hook.__class__.__name__,
            "layer_id": layer_id,
            "metadata": dict(metadata),
            "tensors": serialized_tensors,
        }

    def _apply_generation(
        self, lm, prompt: str, generation, return_options, hooks_payload: Sequence[HookPayload]
    ) -> InferenceOutput:
        device = get_device_from_model(lm.model)
        enc = lm.tokenizer([prompt], return_tensors="pt")
        enc = move_tensors_to_device(enc, device)

        gen_kwargs: Dict = {
            "max_new_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "do_sample": generation.do_sample,
        }
        if generation.top_k is not None:
            gen_kwargs["top_k"] = generation.top_k
        if generation.top_p is not None:
            gen_kwargs["top_p"] = generation.top_p
        if generation.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = generation.repetition_penalty
        if generation.seed is not None:
            torch.manual_seed(generation.seed)

        start = time.perf_counter()
        outputs = lm.model.generate(**enc, **gen_kwargs)
        timing_ms = (time.perf_counter() - start) * 1000

        decoded = lm.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text = decoded[0] if decoded else ""

        if generation.stop:
            stops = [s for s in generation.stop if s]
            if stops:
                earliest = None
                for stop_seq in stops:
                    idx = text.find(stop_seq)
                    if idx != -1:
                        earliest = idx if earliest is None else min(earliest, idx)
                if earliest is not None:
                    text = text[:earliest]
        token_ids = outputs[0].tolist() if len(outputs) else []
        tokens = [lm.tokenizer.decode([tid]) for tid in token_ids]

        hook_results: List[Dict] = []
        for payload in hooks_payload:
            hook_instances = lm.layers.get_hooks(layer_signature=payload.layer_id, hook_type=None)
            hook_results.extend([self._serialize_hook(hook, payload.layer_id) for hook in hook_instances])

        logits: List[float] | None = None
        probs: List[float] | None = None
        if return_options.logits or return_options.probabilities:
            try:
                logits_tensor = lm.inference.extract_logits(outputs)
                if return_options.logits:
                    logits = logits_tensor[0, -1].detach().cpu().tolist()
                if return_options.probabilities:
                    probs_tensor = torch.softmax(logits_tensor[0, -1], dim=-1)
                    probs = probs_tensor.detach().cpu().tolist()
            except Exception:
                pass

        return InferenceOutput(
            text=text,
            tokens=tokens if return_options.tokens else [],
            logits=logits if return_options.logits else None,
            probabilities=probs if return_options.probabilities else None,
            hooks=hook_results,
            timing_ms=timing_ms,
        )

    def run(self, lm, input_payloads: Sequence[InferenceInput]) -> List[InferenceOutput]:
        results: List[InferenceOutput] = []
        for payload in input_payloads:
            hook_ids = self._register_hooks(lm, payload.hooks)
            try:
                result = self._apply_generation(
                    lm=lm,
                    prompt=payload.prompt,
                    generation=payload.generation,
                    return_options=payload.return_options,
                    hooks_payload=payload.hooks,
                )
                results.append(result)
            finally:
                self._cleanup_hooks(lm, hook_ids)
        return results
