from __future__ import annotations

from experiments.baselines.guard_adapters import LlamaGuardAdapter
from experiments.baselines.guard_model import GuardModel


def create_llama_guard(
    model_path: str,
    device: str = "cpu",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> GuardModel:
    """Convenience factory for LlamaGuard as a binary guard baseline.

    Saves a binary `predicted_label` plus optional `threat_category` (S1-S14).
    """

    adapter = LlamaGuardAdapter(
        model_path=model_path,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return GuardModel(adapter=adapter)
