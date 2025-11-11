from pathlib import Path
from typing import Sequence, Any, Dict, TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from amber.core.language_model_layers import LanguageModelLayers
from amber.core.language_model_tokenizer import LanguageModelTokenizer
from amber.core.language_model_activations import LanguageModelActivations
from amber.core.language_model_context import LanguageModelContext
from amber.store.local_store import LocalStore
from amber.store.store import Store
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.sae.concepts.input_tracker import InputTracker

logger = get_logger(__name__)


class LanguageModel:
    """
    Fence-style language model wrapper.
    """

    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            store: Store,
            model_id: str | None = None,
    ):
        self.context = LanguageModelContext(self)
        self.context.model = model
        self.context.tokenizer = tokenizer

        if model_id is not None:
            self.context.model_id = model_id
        elif hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
            self.context.model_id = model.config.name_or_path.replace("/", "_")
        else:
            self.context.model_id = model.__class__.__name__

        self.layers = LanguageModelLayers(self.context)
        self.lm_tokenizer = LanguageModelTokenizer(self.context)
        self.activations = LanguageModelActivations(self.context)

        self.context.store = store
        self._input_tracker: "InputTracker | None" = None
        self._activation_text_trackers: list[Any] = []

    @property
    def model(self) -> nn.Module:
        return self.context.model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.context.tokenizer

    @property
    def model_id(self) -> str:
        return self.context.model_id

    @property
    def store(self) -> Store:
        return self.context.store

    def tokenize(self, texts: Sequence[str], **kwargs: Any):
        return self.lm_tokenizer.tokenize(texts, **kwargs)

    def _inference(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
    ):
        if tok_kwargs is None:
            tok_kwargs = {}

        tok_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            **tok_kwargs,
        }
        enc = self.tokenize(texts, **tok_kwargs)

        first_param = next(self.model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        device_type = str(getattr(device, 'type', 'cpu'))

        if device_type == "cuda":
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        else:
            enc = {k: v.to(device) for k, v in enc.items()}

        self.model.eval()

        if self._input_tracker is not None and self._input_tracker.enabled:
            self._input_tracker.set_current_texts(texts)
        # Also set on any other activation text trackers (for backward compatibility)
        for _tracker in self._activation_text_trackers:
            if hasattr(_tracker, "set_current_texts") and _tracker is not self._input_tracker:
                _tracker.set_current_texts(texts)

        controllers_to_restore = []
        if not with_controllers:
            controllers = self.layers.get_controllers()
            for controller in controllers:
                if controller.enabled:
                    controller.disable()
                    controllers_to_restore.append(controller)

        try:
            with torch.inference_mode():
                if autocast and device_type == "cuda":
                    amp_dtype = autocast_dtype or torch.float16
                    with torch.autocast(device_type, dtype=amp_dtype):
                        output = self.model(**enc)
                else:
                    output = self.model(**enc)

            return output, enc
        finally:
            for controller in controllers_to_restore:
                controller.enable()

    def forwards(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
    ):
        return self._inference(
            texts,
            tok_kwargs=tok_kwargs,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
            with_controllers=with_controllers
        )

    def generate(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
            skip_special_tokens: bool = True,
    ) -> Sequence[str]:
        """
        Run inference and automatically decode the output with the tokenizer.
        
        Args:
            texts: Input texts to process
            tok_kwargs: Optional tokenizer keyword arguments
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Optional dtype for autocast
            with_controllers: Whether to use controllers during inference
            skip_special_tokens: Whether to skip special tokens when decoding
            
        Returns:
            Sequence of decoded text strings
        """
        output, enc = self._inference(
            texts,
            tok_kwargs=tok_kwargs,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
            with_controllers=with_controllers
        )

        # Extract logits from output
        if hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, tuple) and len(output) > 0:
            logits = output[0]
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            raise ValueError(f"Unable to extract logits from output type: {type(output)}")

        # Get predicted token IDs (argmax on last dimension)
        # logits shape: [batch_size, sequence_length, vocab_size]
        predicted_token_ids = logits.argmax(dim=-1)

        # Decode each sequence in the batch
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for decoding but is None")

        decoded_texts = []
        for i in range(predicted_token_ids.shape[0]):
            token_ids = predicted_token_ids[i].cpu().tolist()
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            decoded_texts.append(decoded_text)

        return decoded_texts

    def register_activation_text_tracker(self, tracker: Any) -> None:
        if tracker not in self._activation_text_trackers:
            self._activation_text_trackers.append(tracker)

    def unregister_activation_text_tracker(self, tracker: Any) -> None:
        try:
            self._activation_text_trackers.remove(tracker)
        except ValueError:
            pass

    def get_input_tracker(self) -> "InputTracker | None":
        """Get the InputTracker singleton for this LanguageModel."""
        return self._input_tracker

    def _ensure_input_tracker(self) -> "InputTracker":
        """
        Ensure InputTracker singleton exists.
        
        Returns:
            The InputTracker instance
        """
        if self._input_tracker is not None:
            return self._input_tracker

        from amber.mechanistic.sae.concepts.input_tracker import InputTracker

        self._input_tracker = InputTracker(language_model=self)

        logger.debug(f"Created InputTracker singleton for {self.context.model_id}")

        return self._input_tracker

    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
            store: Store,
            tokenizer_params: dict = None,
            model_params: dict = None,
    ) -> "LanguageModel":
        if tokenizer_params is None:
            tokenizer_params = {}
        if model_params is None:
            model_params = {}

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_params)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)

        lm = cls(model, tokenizer, store)
        return lm

    @classmethod
    def from_local(cls, model_path: str, tokenizer_path: str):
        pass
