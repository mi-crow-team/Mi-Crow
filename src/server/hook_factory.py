from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Type

from mi_crow.hooks.hook import Hook

from server.schemas import HookPayload


class HookFactory:
    """Create hook instances from payload definitions."""

    def __init__(self, hook_classes: Mapping[str, Type[Hook]]):
        self._hook_classes = dict(hook_classes)

    @classmethod
    def from_modules(cls, classes: Iterable[Type[Hook]]) -> "HookFactory":
        registry: Dict[str, Type[Hook]] = {}
        for hook_cls in classes:
            if inspect.isabstract(hook_cls):
                continue
            registry[hook_cls.__name__] = hook_cls
        return cls(registry)

    def available_hooks(self) -> List[str]:
        return sorted(self._hook_classes.keys())

    def create(self, payload: HookPayload) -> Hook:
        if payload.hook_name not in self._hook_classes:
            raise ValueError(f"hook '{payload.hook_name}' is not available")

        hook_cls = self._hook_classes[payload.hook_name]
        sig = inspect.signature(hook_cls.__init__)
        kwargs: MutableMapping[str, Any] = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name == "layer_signature":
                kwargs[name] = payload.layer_id
                continue
            if name in payload.config:
                kwargs[name] = payload.config[name]
                continue
            if param.default is not inspect._empty:
                continue
            raise ValueError(f"Missing required parameter '{name}' for hook '{payload.hook_name}'")

        return hook_cls(**kwargs)
