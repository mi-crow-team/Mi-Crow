# Lightweight shim module to allow tests to monkeypatch local loading paths.
# It re-exports the relevant HF classes so that LanguageModel.from_local
# can import from amber.adapters.language_model and tests can monkeypatch here.
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
