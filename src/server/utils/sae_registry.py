"""SAE class registry utilities."""

from __future__ import annotations

from typing import Dict, Type

from amber.mechanistic.sae.modules.l1_sae import L1Sae
from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.mechanistic.sae.sae import Sae


class SAERegistry:
    """Registry for SAE classes."""

    def __init__(self):
        self._sae_classes: Dict[str, Type[Sae]] = {
            "TopKSae": TopKSae,
            "L1Sae": L1Sae,
        }

    def get_class(self, name: str) -> Type[Sae]:
        """Get SAE class by name."""
        if name not in self._sae_classes:
            raise ValueError(f"Unsupported SAE class '{name}'. Available: {sorted(self._sae_classes.keys())}")
        return self._sae_classes[name]

    def register(self, name: str, sae_class: Type[Sae]) -> None:
        """Register a new SAE class."""
        self._sae_classes[name] = sae_class

    def list_classes(self) -> Dict[str, str]:
        """List all registered SAE classes with their module paths."""
        out: Dict[str, str] = {}
        for name, cls in self._sae_classes.items():
            out[name] = f"{cls.__module__}.{cls.__name__}"
        return out

    def get_all_classes(self) -> Dict[str, Type[Sae]]:
        """Get all registered SAE classes."""
        return self._sae_classes.copy()

