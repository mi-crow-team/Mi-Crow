from __future__ import annotations

from experiments.baselines.guard_adapters import BielikGuardAdapter
from experiments.baselines.guard_model import GuardModel


def create_bielik_guard(
    model_path: str = "speakleash/Bielik-Guard-0.1B-v1.0",
    threshold: float = 0.5,
    device: str = "cpu",
) -> GuardModel:
    """Convenience factory for BielikGuard as a binary guard baseline."""

    adapter = BielikGuardAdapter(model_path=model_path, threshold=threshold, device=device)
    return GuardModel(adapter=adapter)
