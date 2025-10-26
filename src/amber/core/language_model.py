from pathlib import Path
from typing import Callable, Sequence, Any, Dict

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from amber.core.language_model_layers import LanguageModelLayers
from amber.core.language_model_tokenizer import LanguageModelTokenizer
from amber.core.language_model_activations import LanguageModelActivations
from amber.store import Store, LocalStore
from amber.utils import get_logger

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
        self._model = model
        self.model_name = model.__class__.__name__
        self._tokenizer = tokenizer
        self.layers = LanguageModelLayers(self, model)
        self.lm_tokenizer = LanguageModelTokenizer(model, tokenizer)
        self.activations = LanguageModelActivations(self)
        self.store = store or LocalStore(Path.cwd() / "store" / self.model_name)

        self._activation_text_trackers: list[Any] = []

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        return self._tokenizer

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

        first_param = next(self._model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        device_type = str(getattr(device, 'type', 'cpu'))

        if device_type == "cuda":
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        else:
            enc = {k: v.to(device) for k, v in enc.items()}

        self._model.eval()

        # Provide current texts to any registered trackers prior to forward
        try:
            for _tracker in getattr(self, "_activation_text_trackers", []):
                if hasattr(_tracker, "set_current_texts"):
                    _tracker.set_current_texts(texts)
        except Exception:
            logger.exception("Error setting current texts on activation tracker")
            pass

        with torch.inference_mode():
            if autocast and device_type == "cuda":
                amp_dtype = autocast_dtype or torch.float16
                with torch.autocast(device_type, dtype=amp_dtype):
                    output = self._model(**enc)
            else:
                output = self._model(**enc)

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
    ):
        return self._inference(
            texts,
            tok_kwargs=tok_kwargs,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
            discard_output=False,
            save_inputs=False
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
        model = AutoModelWithLMHead.from_pretrained(model_name, **model_params)
        lm = cls(model=model, tokenizer=tokenizer)
        lm.model_name = model_name
        setattr(lm, "hf_repo_id", model_name)
        return lm

    @classmethod
    def from_local(cls, model_path: str, tokenizer_path: str):
        pass
