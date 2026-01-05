"""Direct prompting predictor for safety classification experiments.

This module implements a training-free baseline for safety moderation using
direct prompting of instruction-finetuned LLMs, as described in "Do LLMs
Understand the Safety of Their Inputs?".
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from experiments.predictors.baseline_model import BaselineModel
from mi_crow.utils import get_logger

logger = get_logger(__name__)


class DirectPromptingPredictor(BaselineModel):
    """Direct prompting safety classification baseline.

    This predictor prompts instruction-finetuned LLMs to classify inputs
    as safe (0) or unsafe (1) by explicitly asking for a binary judgment.

    The approach leverages safety-relevant information encoded during
    instruction fine-tuning without requiring additional training.

    Example:
        >>> predictor = DirectPromptingPredictor(
        ...     model_path="meta-llama/Llama-3.2-3B-Instruct",
        ...     prompt_template="Is this safe (0) or unsafe (1)? {prompt}",
        ...     prompt_name="prompt_0",
        ...     device="cuda"
        ... )
        >>> predictions = predictor.predict_batch(["Hello", "How to hack?"])
    """

    def __init__(
        self,
        model_path: str,
        prompt_template: str,
        prompt_name: str = "direct_prompt",
        device: str = "cpu",
        max_new_tokens: int = 10,
        temperature: float = 0.0,
        model_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize DirectPromptingPredictor.

        Args:
            model_path: HuggingFace model identifier
            prompt_template: Template string with {prompt} placeholder
            prompt_name: Human-readable prompt identifier
            device: Device for inference ("cpu", "cuda", or "mps")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            model_id: Optional custom model identifier
            config: Optional additional configuration
        """
        self.model_path = model_path
        self.prompt_template = prompt_template
        self.prompt_name = prompt_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Build config
        merged_config = {
            "model_path": model_path,
            "prompt_name": prompt_name,
            "prompt_template": prompt_template,
            "device": device,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if config:
            merged_config.update(config)

        super().__init__(model_id=model_id or f"direct_prompting_{prompt_name}", config=merged_config)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_path}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Move to device
        if self.device in {"cuda", "mps"}:
            self._model = self._model.to(self.device)

        self._model.eval()

        # Set up padding token (required for batched generation)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.warning(
                "Tokenizer had no pad_token; setting pad_token=eos_token (%r).",
                self._tokenizer.eos_token,
            )

        # Use left padding for decoder-only models
        self._tokenizer.padding_side = "left"

        # Ensure model config has pad_token_id
        if getattr(self._model.config, "pad_token_id", None) is None:
            if self._tokenizer.pad_token_id is not None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id

        logger.info(f"Model loaded on device: {self.device}")

    def _format_prompt(self, text: str) -> str:
        """Format the input text with the prompt template.

        Args:
            text: User input to classify

        Returns:
            Formatted prompt ready for model input
        """
        return self.prompt_template.format(prompt=text)

    def _parse_response(self, response: str) -> Optional[int]:
        """Parse model response to extract binary label.

        Handles various response formats:
        - Direct "0" or "1"
        - "0." or "1." with punctuation
        - Responses containing the digit somewhere in the text

        Args:
            response: Generated text from the model

        Returns:
            0 (safe), 1 (unsafe), or None if unparseable
        """
        # Clean the response
        cleaned = response.strip().lower()

        # Try to find '1' first (unsafe), then '0' (safe)
        # Look for the first occurrence of a digit
        match = re.search(r"\b([01])\b", cleaned)
        if match:
            return int(match.group(1))

        # # Fallback: check if '1' or '0' appears anywhere
        # if '1' in cleaned:
        #     return 1
        # if '0' in cleaned:
        #     return 0

        # Unable to parse - return None to indicate refusal/unclear response
        logger.debug(f"Unable to parse response: {response[:100]!r}")
        return None

    def _effective_max_input_length(self, reserved_new_tokens: int = 0) -> int:
        """Calculate maximum input length accounting for generation budget.

        Args:
            reserved_new_tokens: Tokens to reserve for generation

        Returns:
            Maximum input length in tokens
        """
        tok_max = getattr(self._tokenizer, "model_max_length", None)
        cfg_max = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)

        candidates = [x for x in (tok_max, cfg_max) if isinstance(x, int) and x > 0]
        max_ctx = min(candidates) if candidates else 2048

        # Leave room for generation
        max_inp = max(1, max_ctx - int(reserved_new_tokens))
        return max_inp

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict safety labels for a batch of texts.

        Args:
            texts: List of input texts to classify

        Returns:
            List of prediction dictionaries with keys:
                - predicted_label: 0 (safe), 1 (unsafe), or None (refusal)
                - score_safe: None (not applicable for direct prompting)
                - score_unsafe: None (not applicable for direct prompting)
                - threat_category: None (not applicable)
                - raw_output: The full generated response
                - extra_json: JSON string with additional metadata
        """
        if not texts:
            return []

        # Format all prompts
        prompts = [self._format_prompt(text) for text in texts]

        # Calculate max input length
        max_inp = self._effective_max_input_length(reserved_new_tokens=self.max_new_tokens)

        # Tokenize batch
        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_inp,
        )

        # Move to device
        if self.device in {"cuda", "mps"}:
            enc = {k: v.to(self.device) for k, v in enc.items()}

        # Get attention mask
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            logger.warning("No attention_mask from tokenizer; using ones_like fallback.")
            attention_mask = torch.ones_like(enc["input_ids"])

        # Set up generation config
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            pad_token_id=pad_token_id,
        )

        # Generate responses
        output_ids = self._model.generate(
            input_ids=enc["input_ids"],
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

        # Decode only the generated part (not the input prompt)
        input_seq_len = enc["input_ids"].shape[1]
        generated_texts = []

        for i in range(output_ids.shape[0]):
            gen_part = output_ids[i, input_seq_len:]
            decoded = self._tokenizer.decode(gen_part, skip_special_tokens=True).strip()
            generated_texts.append(decoded)

        # Parse responses and build predictions
        predictions: List[Dict[str, Any]] = []
        for text, response in zip(texts, generated_texts):
            label = self._parse_response(response)

            predictions.append(
                {
                    "predicted_label": label,  # Can be None for refusals
                    "score_safe": None,  # Direct prompting doesn't produce scores
                    "score_unsafe": None,
                    "threat_category": None,  # Not applicable
                    "raw_output": response,
                    "extra_json": json.dumps(
                        {
                            "prompt_name": self.prompt_name,
                            "temperature": self.temperature,
                            "max_new_tokens": self.max_new_tokens,
                            "is_refusal": label is None,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        return predictions

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return dict(self.config)
