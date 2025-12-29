from __future__ import annotations

from experiments.baselines.guard_adapters import LlamaGuardAdapter
from experiments.baselines.guard_model import GuardModel


def create_llama_guard(
    model_path: str = "meta-llama/Llama-Guard-3-1B",
    device: str = "cpu",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    threshold: float = 0.5,
) -> GuardModel:
    """Convenience factory for LlamaGuard as a binary guard baseline.

    Saves a binary `predicted_label` plus optional `threat_category` (S1-S14).
    """

    adapter = LlamaGuardAdapter(
        model_path=model_path,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        threshold=threshold,
    )
    return GuardModel(adapter=adapter)
