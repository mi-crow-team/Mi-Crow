from pathlib import Path
from typing import Callable, Sequence, Any, Dict, Union, TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from amber.core.language_model_layers import LanguageModelLayers
from amber.core.language_model_tokenizer import LanguageModelTokenizer
from amber.core.language_model_activations import LanguageModelActivations
from amber.core.language_model_context import LanguageModelContext
from amber.store import Store
from amber.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LanguageModel:
    """
    Fence-style language model wrapper.
    """

    def __init__(
            self,
            model: nn.Module,
            tokenizer: AutoTokenizer,
            store: Store | None = None
    ):
        # Validate context
        self.context = LanguageModelContext(self)
        self.context.model = model
        self.context.tokenizer = tokenizer
        # Set model ID - handle both HuggingFace models and custom models
        if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
            self.context.model_id = model.config.name_or_path.replace("/", "_")
        else:
            # For custom models without config, use the class name
            self.context.model_id = model.__class__.__name__

        # Initialize components using context
        self.layers = LanguageModelLayers(self.context)
        self.lm_tokenizer = LanguageModelTokenizer(self.context)
        self.activations = LanguageModelActivations(self.context)

        if store is not None:
            self.context.store = store
        else:
            from amber.store import LocalStore
            self.context.store = LocalStore(Path.cwd() / "store" / self.context.model_id)

        self._activation_text_trackers = []

    @property
    def model(self) -> nn.Module | None:
        return self.context.model

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        return self.context.tokenizer

    @property
    def model_id(self) -> str:
        return self.context.model_id

    @property
    def store(self) -> Store | None:
        return self.context.store

    def tokenize(self, texts: Sequence[str], **kwargs: Any):
        return self.lm_tokenizer.tokenize(texts, **kwargs)

    def _inference(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            discard_output: bool = False,
            save_inputs: bool = False,
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

        try:
            for _tracker in self._activation_text_trackers:
                if hasattr(_tracker, "set_current_texts"):
                    _tracker.set_current_texts(texts)
        except Exception:
            logger.exception("Error setting current texts on activation tracker")
            pass

        # Temporarily disable controllers if requested
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
        finally:
            # Re-enable controllers that were disabled
            for controller in controllers_to_restore:
                controller.enable()

        if discard_output:
            if save_inputs:
                input_ids = enc.get("input_ids")
                attn = enc.get("attention_mask")
                if input_ids is not None:
                    input_ids = input_ids.detach().to("cpu", non_blocking=(device_type == "cuda"))
                if attn is not None:
                    attn = attn.detach().to("cpu", non_blocking=(device_type == "cuda"))
                return input_ids, attn
            return None

        return output, enc

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
            discard_output=False,
            save_inputs=False,
            with_controllers=with_controllers
        )

    def register_activation_text_tracker(self, tracker: Any) -> None:
        if tracker not in self._activation_text_trackers:
            self._activation_text_trackers.append(tracker)

    def unregister_activation_text_tracker(self, tracker: Any) -> None:
        try:
            self._activation_text_trackers.remove(tracker)
        except ValueError:
            pass

    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
            tokenizer_params: dict = None,
            model_params: dict = None,
    ) -> "LanguageModel":
        if tokenizer_params is None:
            tokenizer_params = {}
        if model_params is None:
            model_params = {}

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_params)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)

        lm = cls(model, tokenizer)
        return lm

    @classmethod
    def from_local(cls, model_path: str, tokenizer_path: str):
        pass
