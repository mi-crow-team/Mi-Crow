from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

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
        )

        ### DEBUG
        self.total_truncated = 0

    def _effective_max_length(self) -> int:
        tokenizer = getattr(self._pipe, "tokenizer", None)
        model = getattr(self._pipe, "model", None)
        config = getattr(model, "config", None)

        tok_max = getattr(tokenizer, "model_max_length", None)
        cfg_max = getattr(config, "max_position_embeddings", None)

        # Some tokenizers use a very large sentinel to mean "no limit".
        if isinstance(tok_max, int) and tok_max > 1_000_000:
            tok_max = None

        model_type = getattr(config, "model_type", "") or ""
        if isinstance(cfg_max, int) and cfg_max > 0 and model_type in {"roberta", "xlm-roberta", "camembert"}:
            # RoBERTa-family commonly has 514 position embeddings but supports 512-token inputs.
            cfg_max = max(1, cfg_max - 2)

        candidates = [x for x in (tok_max, cfg_max) if isinstance(x, int) and x > 0]
        return min(candidates) if candidates else 512

    @property
    def adapter_id(self) -> str:
        return f"bielik_guard:{self.model_path}"

    def predict_batch(self, texts: List[str]) -> List[JsonDict]:
        max_len = self._effective_max_length()
        ### DEBUG
        # Check lenght - number of tokens in each of texts, find maximum and log
        token_lengths = [len(self._pipe.tokenizer.tokenize(text)) for text in texts]
        # Calculate how many texts exceed max_len
        num_exceeding = sum(1 for length in token_lengths if length > max_len)
        if num_exceeding > 0:
            self.total_truncated += num_exceeding
            logger.warning(
                "BielikGuardAdapter: %d/%d texts exceed max token length of %d; they will be truncated.",
                num_exceeding,
                len(texts),
                max_len,
            )
        # Log example text that exceeds max_len
        for i, length in enumerate(token_lengths):
            if length > max_len:
                logger.warning(
                    "Example exceeding text (length %d tokens): %.100r",
                    length,
                    texts[i],
                )
                break
        ### END DEBUG

        outputs = self._pipe(
            texts,
            truncation=True,
            max_length=max_len,
            padding=True,  # Pad to longest in batch
            top_k=None,  # Return scores for all labels
        )
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

        # Required for true batching: tokenizers for some Llama checkpoints ship without a pad token.
        # When padding=True, Transformers will raise unless pad_token is set.
        if self._tokenizer.pad_token is None:
            # Use EOS as PAD (common practice for decoder-only LMs).
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.warning(
                "LlamaGuard tokenizer had no pad_token; setting pad_token=eos_token (%r).",
                self._tokenizer.eos_token,
            )

        # Batched generation with decoder-only LMs is most reliable with left padding.
        self._tokenizer.padding_side = "left"

        if getattr(self._model.config, "pad_token_id", None) is None and self._tokenizer.pad_token_id is not None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id

        self.safe_token_id = self._tokenizer.convert_tokens_to_ids("safe")
        self.unsafe_token_id = self._tokenizer.convert_tokens_to_ids("unsafe")

        # Debug logging controls for true batching.
        self._true_batch_call_idx = 0
        self._true_batch_log_first_n = 3
        self._true_batch_log_every_n = 200
        self._true_batch_log_decoded_chars = 240

    def _effective_max_input_length(self, reserved_new_tokens: int = 0) -> int:
        tok_max = getattr(self._tokenizer, "model_max_length", None)
        cfg_max = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)

        candidates = [x for x in (tok_max, cfg_max) if isinstance(x, int) and x > 0]
        # Default to something sane if config/tokenizer doesn't report a limit.
        max_ctx = min(candidates) if candidates else 2048

        # Ensure we leave room for generation (and at least 1 token of input).
        max_inp = max(1, max_ctx - int(reserved_new_tokens))
        return max_inp

    @property
    def adapter_id(self) -> str:
        return f"llamaguard:{self.model_path}"

    def _build_prompt(self, text: str) -> str:
        # Minimal, stable prompt that encourages structured output.
        # Important: some Llama chat templates reject the `system` role.
        # So we embed instructions in the user message and use a single user turn.

        # instruction = "Answer with: SAFE or UNSAFE. If UNSAFE, also output a threat category S1-S14."

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    # "content": f"{instruction}\n\n{text}",
                    "content": text,
                }
            ]
            try:
                tmpl = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Log the full template only once (for first call)
                ### DEBUG
                if self._true_batch_call_idx == 1:
                    logger.info("LlamaGuard full chat template for model %s:", self.model_path)
                    logger.info(tmpl)
                logger.info("Chat template for LlamaGuard (first 100 chars): %s", tmpl[:100])
                return tmpl + "\n"
            except Exception as e:
                logger.warning("LlamaGuard tokenizer chat template application failed.", exc_info=True)
                logger.warning("Error message: %s", str(e))

        logger.info("LlamaGuard (%s): using basic prompt (no chat template).", self.model_path)
        return f"User: {text}\nAssistant: \n"

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

    def _true_batch_decode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> List[str]:
        """Decode only the generated continuation per sample (not the echoed prompt).

        Works with padded batches by using per-sample prompt lengths from attention_mask.
        """

        prompt_lens = attention_mask.sum(dim=1).detach().cpu().tolist()
        input_seq_len = int(input_ids.shape[1])
        decoded: List[str] = []
        for row_idx, prompt_len in enumerate(prompt_lens):
            # `generate` appends tokens after the padded input length (same for every row),
            # regardless of padding side. This makes decoding robust.
            gen_part = output_ids[row_idx, input_seq_len:]
            gen_part_ids = gen_part.detach().cpu().tolist()
            decoded.append(self._tokenizer.decode(gen_part_ids, skip_special_tokens=True).strip())
        return decoded

    def _true_batch_parse(self, generated_texts: List[str]) -> List[JsonDict]:
        return [self._parse(t) for t in generated_texts]

    @torch.no_grad()
    def _true_predict_batch(self, texts: List[str]) -> List[JsonDict]:
        """Generate in a single batched `model.generate` call.

        This is substantially faster than looping per sample, especially on GPU.
        """

        if not texts:
            return []

        self._true_batch_call_idx += 1
        should_log = (
            self._true_batch_call_idx <= self._true_batch_log_first_n
            or self._true_batch_call_idx % self._true_batch_log_every_n == 0
        )

        prompts = [self._build_prompt(t) for t in texts]
        max_inp = self._effective_max_input_length(reserved_new_tokens=self.max_new_tokens)

        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_inp,
        )

        if self.device in {"cuda", "mps"}:
            enc = {k: v.to(self.device) for k, v in enc.items()}

        attention_mask = enc.get("attention_mask")
        used_fallback_attention_mask = attention_mask is None
        if attention_mask is None:
            # Should not happen with padding=True, but keep safe. This affects decode prompt lengths.
            logger.warning("LlamaGuard true-batch: tokenizer returned no attention_mask; using ones_like fallback.")
            attention_mask = torch.ones_like(enc["input_ids"])

        if should_log:
            prompt_lens = attention_mask.sum(dim=1).detach().cpu().tolist()
            logger.info(
                "LlamaGuard true-batch[%d]: n=%d max_inp=%d max_new_tokens=%d device=%s",
                self._true_batch_call_idx,
                len(texts),
                max_inp,
                self.max_new_tokens,
                self.device,
            )
            logger.info(
                "LlamaGuard true-batch[%d]: input_ids shape=%s attention_mask shape=%s fallback_mask=%s",
                self._true_batch_call_idx,
                tuple(enc["input_ids"].shape),
                tuple(attention_mask.shape),
                used_fallback_attention_mask,
            )
            logger.info(
                "LlamaGuard true-batch[%d]: prompt_len(min/median/max)=%d/%d/%d",
                self._true_batch_call_idx,
                int(min(prompt_lens)) if prompt_lens else -1,
                int(sorted(prompt_lens)[len(prompt_lens) // 2]) if prompt_lens else -1,
                int(max(prompt_lens)) if prompt_lens else -1,
            )

        # Ensure pad_token_id is set for models/tokenizers that don't define it.
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        if should_log:
            logger.info(
                "LlamaGuard true-batch[%d]: pad_token_id=%s eos_token_id=%s",
                self._true_batch_call_idx,
                pad_token_id,
                self._tokenizer.eos_token_id,
            )

        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
        )

        output_ids = self._model.generate(
            input_ids=enc["input_ids"],
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            pad_token_id=pad_token_id,
        )

        if should_log:
            logger.info(
                "LlamaGuard true-batch[%d]: output_ids shape=%s",
                self._true_batch_call_idx,
                tuple(output_ids.shape),
            )

        generated_texts = self._true_batch_decode(
            input_ids=enc["input_ids"],
            attention_mask=attention_mask,
            output_ids=output_ids,
        )

        if should_log and generated_texts:
            sample = generated_texts[0].replace("\n", " ").strip()
            sample = sample[: self._true_batch_log_decoded_chars]
            logger.info(
                "LlamaGuard true-batch[%d]: decoded sample[0] (%d chars): %r",
                self._true_batch_call_idx,
                len(sample),
                sample,
            )

        return self._true_batch_parse(generated_texts)

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[JsonDict]:
        # Temporary switch: use true batched generation for performance.
        return self._true_predict_batch(texts)

        # --- legacy per-item generation (kept for reference) ---
        # predictions: List[JsonDict] = []
        # gen_cfg = GenerationConfig(
        #     max_new_tokens=self.max_new_tokens,
        #     do_sample=self.temperature > 0,
        #     temperature=self.temperature if self.temperature > 0 else None,
        # )
        #
        # for text in texts:
        #     prompt = self._build_prompt(text)
        #     max_inp = self._effective_max_input_length(reserved_new_tokens=self.max_new_tokens)
        #     inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_inp)
        #     if self.device in {"cuda", "mps"}:
        #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #
        #     output_ids = self._model.generate(**inputs, generation_config=gen_cfg)
        #     decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        #
        #     parsed = self._parse(decoded)
        #     predictions.append(parsed)
        #
        # return predictions

    @torch.no_grad()
    def predict_batch_with_scores(self, texts: List[str]) -> List[Dict[str, Any]]:
        predictions = []

        for text in texts:
            full_prompt = self._build_official_prompt(text)
            # Reserve a small budget for the short generation we do below.
            reserved = 10
            max_inp = self._effective_max_input_length(reserved_new_tokens=reserved)
            inputs = self._tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_inp)
            if self.device in {"cuda", "mps"}:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
            gen_out = self._model.generate(
                **inputs,
                max_new_tokens=reserved,
                pad_token_id=self._tokenizer.eos_token_id,
            )
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
