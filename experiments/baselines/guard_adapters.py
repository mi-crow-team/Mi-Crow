from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from amber.utils import get_logger

logger = get_logger(__name__)


JsonDict = Dict[str, Any]


class GuardAdapter(ABC):
    """Adapter interface to normalize various guard model families.

    Each adapter must return a list of dicts with at least:
    - `predicted_label` (int): 0=safe, 1=unsafe

    Optional (recommended):
    - `threat_category` (str|None): e.g. "S1".."S14" (LlamaGuard)
    - `score_unsafe` / `score_safe` (float|None)
    - `raw_output` (str|None)
    - `extra_json` (str): JSON-encoded model-specific fields
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[JsonDict]:
        raise NotImplementedError


@dataclass
class BielikGuardAdapter(GuardAdapter):
    model_path: str = "speakleash/Bielik-Guard-0.5B-v1.0"
    threshold: float = 0.5
    device: str = "cpu"  # "cpu" | "cuda" | "mps"

    safe_labels: Optional[List[str]] = None
    unsafe_labels: Optional[List[str]] = None

    def __post_init__(self) -> None:
        from transformers import pipeline

        # transformers pipeline uses -1=CPU, >=0 CUDA device index. No native MPS index.
        if self.device == "cuda":
            pipeline_device = 0
        else:
            pipeline_device = -1

        self._pipe = pipeline(
            "text-classification",
            model=self.model_path,
            device=pipeline_device,
            return_all_scores=True,
        )

    @property
    def adapter_id(self) -> str:
        return f"bielik_guard:{self.model_path}"

    def predict_batch(self, texts: List[str]) -> List[JsonDict]:
        outputs = self._pipe(texts)

        predictions: List[JsonDict] = []
        safe_labels = [s.lower() for s in (self.safe_labels or ["safe", "benign"])]
        unsafe_labels = [s.lower() for s in (self.unsafe_labels or ["unsafe", "harmful", "toxic", "attack"])]

        for text_scores in outputs:
            scores_by_label = {item["label"]: float(item["score"]) for item in text_scores}

            # Try to find explicit unsafe vs safe labels; otherwise fallback to max score.
            score_safe = None
            score_unsafe = None
            for label, score in scores_by_label.items():
                label_lower = label.lower()
                if any(tok in label_lower for tok in safe_labels):
                    score_safe = max(score_safe or 0.0, score)
                if any(tok in label_lower for tok in unsafe_labels):
                    score_unsafe = max(score_unsafe or 0.0, score)

            if score_unsafe is None and score_safe is None:
                # Fallback: treat argmax label containing "safe" as safe, else unsafe.
                best_label = max(scores_by_label.items(), key=lambda kv: kv[1])[0]
                best_score = scores_by_label[best_label]
                if "safe" in best_label.lower():
                    score_safe = best_score
                    score_unsafe = 1.0 - best_score
                else:
                    score_unsafe = best_score
                    score_safe = 1.0 - best_score

            predicted_label = 1 if (score_unsafe or 0.0) >= self.threshold else 0

            predictions.append(
                {
                    "predicted_label": predicted_label,
                    "score_safe": float(score_safe) if score_safe is not None else None,
                    "score_unsafe": float(score_unsafe) if score_unsafe is not None else None,
                    "threat_category": None,
                    "raw_output": None,
                    "extra_json": json.dumps(
                        {
                            "scores_by_label": scores_by_label,
                            "threshold": self.threshold,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        return predictions


@dataclass
class LlamaGuardAdapter(GuardAdapter):
    """LlamaGuard-style adapter (generative moderation).

    This adapter assumes the model generates a textual moderation decision and (optionally)
    a threat category in the range S1..S14.

    Parsing is heuristic by design; store `raw_output` for later auditing.
    """

    model_path: str
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    max_new_tokens: int = 128
    temperature: float = 0.0

    decision_tokens: Optional[List[str]] = None  # override if needed

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Avoid device_map complexity; keep simple + explicit.
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.device in {"cuda", "mps"}:
            self._model = self._model.to(self.device)

        self._model.eval()

    @property
    def adapter_id(self) -> str:
        return f"llamaguard:{self.model_path}"

    def _build_prompt(self, text: str) -> str:
        # Minimal, stable prompt that encourages structured output.
        # If the model has a chat template, we use it; otherwise raw string.
        system = (
            "You are a safety classifier. Answer with: SAFE or UNSAFE. If UNSAFE, also output a threat category S1-S14."
        )

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
            try:
                return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        return f"System: {system}\nUser: {text}\nAssistant:"

    def _parse(self, generated_text: str) -> JsonDict:
        # Normalize whitespace for parsing
        norm = " ".join(generated_text.strip().split())
        upper = norm.upper()

        threat_match = re.search(r"\bS(1[0-4]|[1-9])\b", upper)
        threat_category = threat_match.group(0) if threat_match else None

        # Decision heuristic
        is_unsafe = None

        if "UNSAFE" in upper:
            is_unsafe = True
        elif "SAFE" in upper:
            is_unsafe = False

        if is_unsafe is None:
            # Fallback: if threat category present => unsafe
            is_unsafe = threat_category is not None

        predicted_label = 1 if is_unsafe else 0

        return {
            "predicted_label": predicted_label,
            "score_safe": None,
            "score_unsafe": None,
            "threat_category": threat_category,
            "raw_output": generated_text,
            "extra_json": json.dumps({"parsed_from": "generation"}, ensure_ascii=False),
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[JsonDict]:
        from transformers import GenerationConfig

        predictions: List[JsonDict] = []
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
        )

        for text in texts:
            prompt = self._build_prompt(text)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.device in {"cuda", "mps"}:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            output_ids = self._model.generate(**inputs, generation_config=gen_cfg)
            decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Try to isolate the assistant part if prompt is echoed.
            parsed = self._parse(decoded)
            predictions.append(parsed)

        return predictions
