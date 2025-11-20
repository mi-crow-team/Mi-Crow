from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, Any, Dict, List, TYPE_CHECKING

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from amber.language_model.language_model_layers import LanguageModelLayers
from amber.language_model.language_model_tokenizer import LanguageModelTokenizer
from amber.language_model.language_model_activations import LanguageModelActivations
from amber.language_model.language_model_context import LanguageModelContext
from amber.language_model.language_model_contracts import ModelMetadata
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
        """Tokenize texts using the language model tokenizer.
        
        Args:
            texts: Sequence of text strings to tokenize
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized encodings
        """
        return self.lm_tokenizer.tokenize(texts, **kwargs)

    def _get_device(self) -> torch.device:
        """Get the device from model parameters."""
        first_param = next(self.model.parameters(), None)
        return first_param.device if first_param is not None else torch.device("cpu")

    def _prepare_tokenizer_kwargs(self, tok_kwargs: Dict | None) -> Dict[str, Any]:
        """Prepare tokenizer keyword arguments with defaults."""
        if tok_kwargs is None:
            tok_kwargs = {}
        return {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            **tok_kwargs,
        }

    def _move_encodings_to_device(self, enc: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Move encodings to the specified device."""
        device_type = str(device.type)
        if device_type == "cuda":
            return {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        return {k: v.to(device) for k, v in enc.items()}

    def _inference(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
    ):
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before running inference")

        tok_kwargs = self._prepare_tokenizer_kwargs(tok_kwargs)
        enc = self.tokenize(texts, **tok_kwargs)

        device = self._get_device()
        device_type = str(device.type)
        enc = self._move_encodings_to_device(enc, device)

        self.model.eval()

        self._setup_trackers(texts)

        controllers_to_restore = self._prepare_controllers(with_controllers)

        try:
            output = self._run_model_forward(enc, autocast, device_type, autocast_dtype)
            return output, enc
        finally:
            self._restore_controllers(controllers_to_restore)

    def _setup_trackers(self, texts: Sequence[str]) -> None:
        """Setup input trackers for current texts."""
        if self._input_tracker is not None and self._input_tracker.enabled:
            self._input_tracker.set_current_texts(texts)

    def _prepare_controllers(self, with_controllers: bool) -> List[Any]:
        """Prepare controllers for inference, disabling if needed."""
        controllers_to_restore = []
        if not with_controllers:
            controllers = self.layers.get_controllers()
            for controller in controllers:
                if controller.enabled:
                    controller.disable()
                    controllers_to_restore.append(controller)
        return controllers_to_restore

    def _restore_controllers(self, controllers_to_restore: List[Any]) -> None:
        """Restore controllers that were disabled."""
        for controller in controllers_to_restore:
            controller.enable()

    def _extract_logits_from_output(self, output: Any) -> torch.Tensor:
        """Extract logits tensor from model output."""
        if hasattr(output, 'logits'):
            return output.logits
        elif isinstance(output, tuple) and len(output) > 0:
            return output[0]
        elif isinstance(output, torch.Tensor):
            return output
        else:
            raise ValueError(f"Unable to extract logits from output type: {type(output)}")

    def _run_model_forward(
            self,
            enc: Dict[str, torch.Tensor],
            autocast: bool,
            device_type: str,
            autocast_dtype: torch.dtype | None
    ) -> Any:
        """Run model forward pass with optional autocast."""
        with torch.inference_mode():
            if autocast and device_type == "cuda":
                amp_dtype = autocast_dtype or torch.float16
                with torch.autocast(device_type, dtype=amp_dtype):
                    return self.model(**enc)
            return self.model(**enc)

    def forwards(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
    ):
        """Run forward pass on texts.
        
        Args:
            texts: Input texts to process
            tok_kwargs: Optional tokenizer keyword arguments
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Optional dtype for autocast
            with_controllers: Whether to use controllers during inference
            
        Returns:
            Tuple of (model_output, encodings)
        """
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
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for decoding but is None")

        output, enc = self._inference(
            texts,
            tok_kwargs=tok_kwargs,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
            with_controllers=with_controllers
        )

        logits = self._extract_logits_from_output(output)
        predicted_token_ids = logits.argmax(dim=-1)

        decoded_texts = []
        for i in range(predicted_token_ids.shape[0]):
            token_ids = predicted_token_ids[i].cpu().tolist()
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            decoded_texts.append(decoded_text)

        return decoded_texts

    def get_input_tracker(self) -> "InputTracker | None":
        """Get the input tracker instance if it exists.
        
        Returns:
            InputTracker instance or None
        """
        return self._input_tracker

    def get_all_detector_metadata(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Tensor]]]:
        """Get metadata from all registered detectors.
        
        Returns:
            Tuple of (detectors_metadata, detectors_tensor_metadata)
        """
        detectors = self.layers.get_detectors()
        detectors_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        detectors_tensor_metadata: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

        for detector in detectors:
            detectors_metadata[detector.layer_signature] = detector.metadata
            detectors_tensor_metadata[detector.layer_signature] = detector.tensor_metadata

        return detectors_metadata, detectors_tensor_metadata

    def save_detector_metadata(self, run_name: str, batch_idx: int) -> str:
        """Save detector metadata to store.
        
        Args:
            run_name: Name of the run
            batch_idx: Batch index
            
        Returns:
            Path where metadata was saved
        """
        if self.store is None:
            raise ValueError("Store must be provided or set on the language model")
        detectors_metadata, detectors_tensor_metadata = self.get_all_detector_metadata()
        return self.store.put_detector_metadata(run_name, batch_idx, detectors_metadata, detectors_tensor_metadata)

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

    def save_model(self, path: Path | str | None = None) -> Path:
        """
        Save the model and its metadata to the store.
        
        Args:
            path: Optional path to save the model. If None, defaults to {model_id}/model.pt
                  relative to the store base path.
                  
        Returns:
            Path where the model was saved
            
        Raises:
            ValueError: If store is not set
        """
        if self.store is None:
            raise ValueError("Store must be provided or set on the language model")
        
        if path is None:
            save_path = Path(self.store.base_path) / self.model_id / "model.pt"
        else:
            save_path = Path(path)
            if not save_path.is_absolute():
                save_path = Path(self.store.base_path) / save_path
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect hooks information
        hooks_info: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for layer_signature, hook_types in self.context._hook_registry.items():
            layer_key = str(layer_signature)
            for hook_type, hooks_list in hook_types.items():
                for hook, _ in hooks_list:
                    hook_info = {
                        "hook_id": hook.id,
                        "hook_type": hook.hook_type.value if hasattr(hook.hook_type, 'value') else str(hook.hook_type),
                        "layer_signature": str(hook.layer_signature) if hook.layer_signature is not None else None,
                        "hook_class": hook.__class__.__name__,
                        "enabled": hook.enabled,
                    }
                    hooks_info[layer_key].append(hook_info)
        
        model_state_dict = self.model.state_dict()
        
        metadata = ModelMetadata(
            model_id=self.model_id,
            hooks=dict(hooks_info),
            model_path=str(save_path)
        )
        
        payload = {
            "model_state_dict": model_state_dict,
            "metadata": asdict(metadata),
        }
        
        torch.save(payload, save_path)
        logger.info(f"Saved model to {save_path}")
        
        return save_path

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
    def from_local_torch(cls, model_path: str, tokenizer_path: str, store: Store) -> "LanguageModel":
        """
        Load a language model from local HuggingFace paths.
        
        Args:
            model_path: Path to the model directory or file
            tokenizer_path: Path to the tokenizer directory or file
            store: Store instance for persistence
            
        Returns:
            LanguageModel instance
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        return cls(model, tokenizer, store)

    @classmethod
    def from_local(cls, saved_path: Path | str, store: Store, model_id: str | None = None) -> "LanguageModel":
        """
        Load a language model from a saved file (created by save_model).
        
        Args:
            saved_path: Path to the saved model file (.pt file)
            store: Store instance for persistence
            model_id: Optional model identifier. If not provided, will use the model_id from saved metadata.
                     If provided, will be used to load the model architecture from HuggingFace.
                     
        Returns:
            LanguageModel instance
            
        Raises:
            FileNotFoundError: If the saved file doesn't exist
            ValueError: If the saved file format is invalid or model_id is required but not provided
        """
        saved_path = Path(saved_path)
        if not saved_path.exists():
            raise FileNotFoundError(f"Saved model file not found: {saved_path}")
        
        # Load the saved payload
        payload = torch.load(saved_path, map_location='cpu')
        
        # Validate payload structure
        if "model_state_dict" not in payload:
            raise ValueError(f"Invalid saved model format: missing 'model_state_dict' key in {saved_path}")
        if "metadata" not in payload:
            raise ValueError(f"Invalid saved model format: missing 'metadata' key in {saved_path}")
        
        model_state_dict = payload["model_state_dict"]
        metadata_dict = payload["metadata"]
        
        # Get model_id from metadata or use provided one
        saved_model_id = metadata_dict.get("model_id")
        if model_id is None:
            if saved_model_id is None:
                raise ValueError(
                    f"model_id not found in saved metadata and not provided. "
                    f"Please provide model_id parameter."
                )
            model_id = saved_model_id
        
        # Load model and tokenizer from HuggingFace using model_id
        # This assumes model_id is a valid HuggingFace model name
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as e:
            raise ValueError(
                f"Failed to load model '{model_id}' from HuggingFace. "
                f"Error: {e}. "
                f"Please ensure model_id is a valid HuggingFace model name."
            ) from e
        
        # Load the saved state dict into the model
        model.load_state_dict(model_state_dict)
        
        # Create LanguageModel instance
        lm = cls(model, tokenizer, store, model_id=model_id)
        
        # Note: Hooks are not automatically restored as they require hook instances
        # The hook metadata is available in metadata_dict["hooks"] if needed
        
        logger.info(f"Loaded model from {saved_path} (model_id: {model_id})")
        
        return lm
