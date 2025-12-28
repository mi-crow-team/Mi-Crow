from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

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

    categories = ["self-harm", "hate", "vulgar", "sex", "crime"]  # Categories used by Bielik-Guard-0.5B-v1.0

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

        for text_scores in outputs:
            scores_by_label = {item["label"]: float(item["score"]) for item in text_scores}

            # Identify ALL categories that exceed the threshold
            triggered_categories = [cat for cat in self.categories if scores_by_label.get(cat, 0.0) >= self.threshold]

            # The overall unsafe score is the highest risk found
            max_unsafe_score = max([scores_by_label.get(cat, 0.0) for cat in self.categories])

            # The prompt is "unsafe" if at least one category triggered
            is_unsafe = len(triggered_categories) > 0

            predictions.append(
                {
                    "predicted_label": 1 if is_unsafe else 0,
                    "score_safe": 1 - float(max_unsafe_score) if max_unsafe_score is not None else None,
                    "score_unsafe": float(max_unsafe_score) if max_unsafe_score is not None else None,
                    "threat_category": triggered_categories if is_unsafe else None,
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
    threshold: float = 0.5  # for score-based decision

    decision_tokens: Optional[List[str]] = None  # override if needed

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Avoid device_map complexity; keep simple + explicit.
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.device in {"cuda", "mps"}:
            self._model = self._model.to(self.device)

        self._model.eval()

        self.safe_token_id = self._tokenizer.convert_tokens_to_ids("safe")
        self.unsafe_token_id = self._tokenizer.convert_tokens_to_ids("unsafe")

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
                tmpl = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                #### DEBUG
                logger.info("Chat template for LlamaGuard (first 100 chars): %s", tmpl[:100])
                return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        return f"System: {system}\nUser: {text}\nAssistant:"

    def _build_official_prompt(self, text: str) -> str:
        # It is CRITICAL to use the model's own chat template.
        # Llama Guard 3 expects the <BEGIN UNSAFE CONTENT CATEGORIES> tags.
        messages = [{"role": "user", "content": text}]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # The trick: add the newline to bypass the model's tendency to start with whitespace
        return prompt + "\n"

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

    @torch.no_grad()
    def predict_batch_with_scores(self, texts: List[str]) -> List[Dict[str, Any]]:
        predictions = []

        for text in texts:
            full_prompt = self._build_official_prompt(text)
            inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self.device)

            # 1. Get Logits for the NEXT token only
            outputs = self._model(inputs.input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

            # 2. Extract specific probabilities
            # Probability = P(unsafe) / (P(safe) + P(unsafe))
            p_safe = probs[0, self.safe_token_id].item()
            p_unsafe = probs[0, self.unsafe_token_id].item()

            # Normalized unsafe score
            conf_score_unsafe = p_unsafe / (p_safe + p_unsafe + 1e-9)

            # 3. Generate the actual text to get the Threat Category (S1-S14)
            # We only generate if it's likely unsafe, or always if we need the S-code
            gen_out = self._model.generate(**inputs, max_new_tokens=10, pad_token_id=self._tokenizer.eos_token_id)
            generated_text = self._tokenizer.decode(
                gen_out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            # Logic: Decision based on text OR threshold
            is_flagged = "unsafe" in generated_text.lower() or conf_score_unsafe >= self.threshold

            # Parse S-codes (e.g., "unsafe\nS1, S2")
            categories = [cat.strip() for cat in generated_text.split("\n")[-1].split(",")] if is_flagged else []

            predictions.append(
                {
                    "predicted_label": 1 if is_flagged else 0,
                    "score_unsafe": conf_score_unsafe,
                    "score_safe": 1.0 - conf_score_unsafe,
                    "threat_category": categories if is_flagged else None,
                    "raw_output": generated_text,
                    "extra_json": json.dumps(
                        {"logit_p_unsafe": p_unsafe, "logit_p_safe": p_safe, "threshold": self.threshold}
                    ),
                }
            )

        return predictions
