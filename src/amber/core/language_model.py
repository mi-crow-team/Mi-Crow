from pathlib import Path
from typing import Callable, Sequence, Any

from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from amber.core.language_model_layers import LanguageModelLayers
from amber.core.language_model_tokenizer import LanguageModelTokenizer
from amber.core.language_model_activations import LanguageModelActivations
from amber.store import Store, LocalStore


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
        self.model = model
        self.model_name = model.__class__.__name__
        self.tokenizer = tokenizer
        self.layers = LanguageModelLayers(model)
        self.lm_tokenizer = LanguageModelTokenizer(model, tokenizer)
        self.activations = LanguageModelActivations(self)
        self.store = store or LocalStore(Path.cwd() / "store" / self.model_name)

    def get_model(self) -> nn.Module:
        return self.model

    def get_tokenizer(self) -> AutoTokenizer | None:
        return self.tokenizer

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def tokenize(self, texts: Sequence[str], **kwargs: Any):
        return self.lm_tokenizer.tokenize(texts, **kwargs)

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
        return cls(model=model, tokenizer=tokenizer)  # TODO: fix the type hinting

    @classmethod
    def from_local(cls, model_path: str, tokenizer_path: str):
        """Load model and tokenizer from local directories without network access.

        Parameters:
            model_path: Filesystem path to a directory containing a saved HF causal LM.
            tokenizer_path: Filesystem path to a directory containing a saved HF tokenizer.
        """
        # Import locally to respect potential monkeypatching of transformers in tests
        from transformers import AutoTokenizer as _AT, AutoModelForCausalLM as _AM

        tokenizer = _AT.from_pretrained(tokenizer_path, local_files_only=True)
        model = _AM.from_pretrained(model_path, local_files_only=True)
        return cls(model=model, tokenizer=tokenizer)
