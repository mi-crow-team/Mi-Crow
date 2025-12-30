# Raport implementacji modułów i automatyzacji

## 1. Procedury automatyzacji

### 1.1 Zarządzanie zależnościami (UV)

**Opis**: Projekt używa UV jako menedżera pakietów i zależności.

**Konfiguracja**: 
- Plik `pyproject.toml` zawiera sekcję `[project]` z podstawowymi zależnościami
- Grupy zależności zdefiniowane w `[project.optional-dependencies]`:
  - `dev`: pre-commit, ruff, pytest, pytest-cov
  - `docs`: mkdocs, mkdocs-material, mkdocstrings, mkdocs-section-index, mkdocs-redirects, mkdocs-literate-nav, mkdocs-gen-files, mike
- Dodatkowa grupa `[dependency-groups]` z `dev` zawierająca pytest-cov

**Komendy**:
- `uv sync` - synchronizacja zależności zgodnie z `uv.lock`
- `uv add <package>` - dodanie nowej zależności
- `uv lock` - aktualizacja pliku `uv.lock`

**Plik uv.lock**: Automatycznie generowany plik blokujący wersje wszystkich zależności dla reprodukowalności środowiska.

### 1.2 Pre-commit hooks

**Konfiguracja**: Plik `.pre-commit-config.yaml` zawiera lokalne hooki:
- `ruff` (linter): `uv run ruff check --fix --show-fixes`
- `ruff-format` (formatter): `uv run ruff format`

**Instalacja**: `pre-commit install` - automatyczne uruchamianie przed każdym commitem.

**Działanie**: Hooks uruchamiane są automatycznie przed commitami, sprawdzając i formatując kod Python.

### 1.3 Ruff - linting i formatowanie

**Konfiguracja**: Sekcja `[tool.ruff]` w `pyproject.toml`:
- `line-length = 120`
- `target-version = ["py310", "py311", "py312"]`
- `[tool.ruff.lint]`: `select = ["E", "F", "I", "W", "C"]` (błędy, formatowanie, importy, ostrzeżenia, złożoność)
- `[tool.ruff.format]`: `quote-style = "double"`, `indent-style = "space"`

**Komendy**:
- `uv run ruff check` - sprawdzanie błędów
- `uv run ruff format` - formatowanie kodu

### 1.4 Testy i coverage (pytest)

**Konfiguracja**: Sekcja `[tool.pytest.ini_options]` w `pyproject.toml`:
- `addopts = "--maxfail=1 -q --cov=mi_crow --cov-report=term-missing --cov-report=xml --cov-fail-under=90"`
- `testpaths = ["tests"]`
- `pythonpath = ["src"]`

**Pokrycie kodu**: 
- Wymagane minimum: 90% (`fail-under=90`)
- Raporty: terminal (z brakującymi liniami), XML (`coverage.xml`), HTML (`htmlcov/`)

**Markery testów**:
- `--unit` - tylko testy jednostkowe z `tests/unit/`
- `--e2e` - tylko testy end-to-end z `tests/e2e/`

**Środowisko testowe**:
- Fixtures w `tests/conftest.py`: `temp_store`, `mock_store`, `mock_model`, `mock_tokenizer`, `mock_language_model`, `sample_dataset`, `sample_classification_dataset`
- Parametryzowane fixtures dla strategii ładowania: `loading_strategy`, `non_streaming_strategy`

### 1.5 GitHub Actions

**Workflow testów** (`.github/workflows/tests.yml`):
- Trigger: push/PR na branche `main`, `stage`
- Macierz: Python 3.10, 3.11, 3.12
- Kroki:
  1. Checkout kodu
  2. Instalacja Python i UV
  3. Instalacja projektu z grupą `dev`
  4. Weryfikacja instalacji pakietu
  5. Uruchomienie pytest z coverage (próg 85% w CI)
  6. Upload artefaktów: `coverage.xml` i `htmlcov/`

**Workflow dokumentacji** (`.github/workflows/docs.yml`):
- Trigger: push na `main`, workflow_dispatch
- Kroki:
  1. Checkout kodu
  2. Instalacja Python i UV
  3. Cache zależności
  4. Instalacja projektu z grupą `docs`
  5. Budowa dokumentacji: `mkdocs build --strict`
  6. Deploy do GitHub Pages

**Badge CI**: Widoczny w `README.md`: `![CI](https://github.com/AdamKaniasty/Inzynierka/actions/workflows/tests.yml/badge.svg?branch=main)`

### 1.6 Dokumentacja (MkDocs)

**Konfiguracja**: Plik `mkdocs.yml`:
- Theme: Material
- Plugins: `search`, `mkdocstrings` (Google style), `section-index`, `redirects`
- URL: https://adamkaniasty.github.io/Inzynierka/
- Automatyczny deploy przez GitHub Actions na GitHub Pages

## 2. Implementacja modułów

### 2.1 Moduł mi_crow.datasets

**BaseDataset** (`src/mi_crow/datasets/base_dataset.py`):
- Klasa abstrakcyjna bazowa dla wszystkich datasetów
- Parametry inicjalizacji:
  - `ds`: `Dataset | IterableDataset` - dataset HuggingFace
  - `store`: `Store` - instancja Store do cache'owania
  - `loading_strategy`: `LoadingStrategy` - strategia ładowania (domyślnie `MEMORY`)
- Metody abstrakcyjne: `__getitem__()`, `__len__()`, `__iter__()`

**LoadingStrategy** (`src/mi_crow/datasets/loading_strategy.py`):
- Enum z trzema strategiami:
  - `MEMORY`: Ładowanie całego datasetu do pamięci (najszybszy dostęp, najwyższe użycie pamięci)
  - `DYNAMIC_LOAD`: Zapis na dysk, odczyt dynamiczny przez memory-mapped Arrow files (obsługuje len/getitem, niższe użycie pamięci)
  - `ITERABLE_ONLY`: Prawdziwy streaming używając IterableDataset (najniższe użycie pamięci, brak wsparcia dla len/getitem)

**TextDataset** (`src/mi_crow/datasets/text_dataset.py`):
- Dataset tekstowy dziedziczący z `BaseDataset`
- Parametry:
  - `ds`: `Dataset | IterableDataset`
  - `store`: `Store`
  - `loading_strategy`: `LoadingStrategy` (domyślnie `MEMORY`)
  - `text_field`: `str` - nazwa kolumny z tekstem (domyślnie `"text"`)

**ClassificationDataset** (`src/mi_crow/datasets/classification_dataset.py`):
- Dataset z kategoriami dziedziczący z `BaseDataset`
- Parametry:
  - `ds`: `Dataset | IterableDataset`
  - `store`: `Store`
  - `loading_strategy`: `LoadingStrategy` (domyślnie `MEMORY`)
  - `text_field`: `str` - nazwa kolumny z tekstem (domyślnie `"text"`)
  - `category_field`: `Union[str, List[str]]` - nazwa kolumny(ów) z kategorią/etykietą (domyślnie `"category"`), obsługuje pojedyncze lub wielokrotne etykiety

### 2.2 Moduł mi_crow.hooks

**Hook** (`src/mi_crow/hooks/hook.py`):
- Klasa abstrakcyjna bazowa dla wszystkich hooków
- Parametry inicjalizacji:
  - `layer_signature`: `str | int | None` - nazwa lub indeks warstwy do podpięcia hooka
  - `hook_type`: `HookType | str` - typ hooka (domyślnie `HookType.FORWARD`)
  - `hook_id`: `str | None` - unikalny identyfikator (auto-generowany jeśli nie podany)
- `HookType` Enum: `FORWARD`, `PRE_FORWARD`

**Detector** (`src/mi_crow/hooks/detector.py`):
- Klasa abstrakcyjna dziedzicząca z `Hook` do wykrywania/zapisywania aktywacji
- Parametry:
  - `hook_type`: `HookType | str` (domyślnie `HookType.FORWARD`)
  - `hook_id`: `str | None`
  - `store`: `Store | None` - opcjonalny Store do zapisywania metadanych
  - `layer_signature`: `str | int | None`
- Atrybuty: `metadata: Dict[str, Any]`, `tensor_metadata: Dict[str, torch.Tensor]`

**Controller** (`src/mi_crow/hooks/controller.py`):
- Klasa abstrakcyjna dziedzicząca z `Hook` do modyfikacji aktywacji
- Parametry:
  - `hook_type`: `HookType | str` (domyślnie `HookType.FORWARD`)
  - `hook_id`: `str | None`
  - `layer_signature`: `str | int | None`
- Może modyfikować wejścia (pre_forward) lub wyjścia (forward) warstw

**LayerActivationDetector** (`src/mi_crow/hooks/implementations/activation_saver.py`):
- Implementacja `Detector` do zapisywania aktywacji warstw
- Zapisuje aktywacje do Store podczas forward pass

**FunctionController** (`src/mi_crow/hooks/implementations/function_controller.py`):
- Implementacja `Controller` do kontroli funkcji
- Pozwala na modyfikację aktywacji przez funkcję użytkownika

### 2.3 Moduł mi_crow.language_model

**LanguageModel** (`src/mi_crow/language_model/language_model.py`):
- Główna klasa wrappera modelu językowego
- Parametry inicjalizacji:
  - `model`: `nn.Module` - model PyTorch
  - `tokenizer`: `PreTrainedTokenizerBase` - tokenizer HuggingFace
  - `store`: `Store` - instancja Store do persystencji
  - `model_id`: `str | None` - opcjonalny identyfikator modelu (auto-ekstrahowany jeśli nie podany)
- Metody:
  - `tokenize()` - tokenizacja tekstów
  - `forward()` - forward pass modelu
  - `generate()` - generowanie tekstu
  - `save_model()` - zapis modelu
- Factory methods:
  - `from_huggingface(model_name, store, tokenizer_params, model_params)` - ładowanie z HuggingFace Hub
  - `from_local_torch(model_path, tokenizer_path, store)` - ładowanie z lokalnych ścieżek HuggingFace
  - `from_local(saved_path, store, model_id)` - ładowanie z zapisanego pliku

**LanguageModelLayers** (`src/mi_crow/language_model/layers.py`):
- Zarządzanie warstwami i hookami
- Metody do rejestracji/odrejestracji hooków na warstwach

**LanguageModelActivations** (`src/mi_crow/language_model/activations.py`):
- Zarządzanie aktywacjami modelu
- Dostęp do zapisanych aktywacji z Store

**LanguageModelTokenizer** (`src/mi_crow/language_model/tokenizer.py`):
- Wrapper tokenizera
- Ujednolicony interfejs do operacji tokenizacji

**InferenceEngine** (`src/mi_crow/language_model/inference.py`):
- Silnik inferencji dla LanguageModel
- Parametry metody `execute_inference()`:
  - `texts`: `Sequence[str]` - sekwencja tekstów wejściowych
  - `tok_kwargs`: `Dict | None` - opcjonalne parametry tokenizera
  - `autocast`: `bool` - użycie automatic mixed precision (domyślnie `True`)
  - `autocast_dtype`: `torch.dtype | None` - opcjonalny dtype dla autocast
  - `with_controllers`: `bool` - użycie controllerów podczas inferencji (domyślnie `True`)

**LanguageModelContext** (`src/mi_crow/language_model/context.py`):
- Dataclass z kontekstem współdzielonym dla LanguageModel i jego komponentów
- Zawiera: `model`, `tokenizer`, `model_id`, `store`, `device`, `dtype`, rejestry hooków

### 2.4 Moduł mi_crow.mechanistic.sae (moduły uczenia maszynowego)

**Sae** (`src/mi_crow/mechanistic/sae/sae.py`):
- Klasa abstrakcyjna dziedzicząca z `Controller` i `Detector`
- Parametry inicjalizacji:
  - `n_latents`: `int` - liczba neuronów latentnych
  - `n_inputs`: `int` - liczba wejść
  - `hook_id`: `str | None` - identyfikator hooka
  - `device`: `str` - urządzenie (domyślnie `'cpu'`)
  - `store`: `Store | None` - opcjonalny Store
- Metody abstrakcyjne:
  - `encode(x: torch.Tensor) -> torch.Tensor` - enkodowanie wejścia
  - `decode(x: torch.Tensor) -> torch.Tensor` - dekodowanie latentów
  - `forward(x: torch.Tensor) -> torch.Tensor` - forward pass
  - `modify_activations(module, inputs, output) -> torch.Tensor | None` - modyfikacja aktywacji (Controller)
  - `save(name: str)` - zapis modelu

**TopKSae** (`src/mi_crow/mechanistic/sae/modules/topk_sae.py`):
- Implementacja TopK SAE dziedzicząca z `Sae`
- Dodatkowy parametr: `k: int` - liczba aktywnych neuronów (TopK)
- Metody:
  - `train(store, run_id, layer_signature, config)` - trening SAE używając aktywacji z Store
  - `save(name, path)` - zapis modelu (overcomplete state dict + metadata mi_crow)
  - `load(path)` - statyczna metoda do ładowania TopKSae z pliku

**SaeTrainer** (`src/mi_crow/mechanistic/sae/sae_trainer.py`):
- Klasa trenująca SAE używając funkcji z biblioteki `overcomplete`
- Metoda `train(store, run_id, layer_signature, config)` - główna metoda treningu

**SaeTrainingConfig** (`src/mi_crow/mechanistic/sae/sae_trainer.py`):
- Dataclass z konfiguracją treningu SAE
- Parametry treningu:
  - `epochs`: `int` (domyślnie 1)
  - `batch_size`: `int` (domyślnie 1024)
  - `lr`: `float` (domyślnie 1e-3)
  - `l1_lambda`: `float` (domyślnie 0.0)
  - `device`: `str | torch.device` (domyślnie "cpu")
  - `dtype`: `Optional[torch.dtype]`
- Parametry zaawansowane:
  - `use_amp`: `bool` - automatic mixed precision (domyślnie `True`)
  - `amp_dtype`: `Optional[torch.dtype]`
  - `grad_accum_steps`: `int` (domyślnie 1)
  - `clip_grad`: `float` (domyślnie 1.0)
  - `monitoring`: `int` - poziom monitorowania (0=silent, 1=basic, 2=detailed, domyślnie 1)
  - `max_batches_per_epoch`: `Optional[int]`
  - `scheduler`: `Optional[Any]` - learning rate scheduler
  - `max_nan_fallbacks`: `int` (domyślnie 5)
- Parametry wandb:
  - `use_wandb`: `bool` (domyślnie `False`)
  - `wandb_project`: `Optional[str]` - nazwa projektu (domyślnie "sae-training" jeśli nie ustawione)
  - `wandb_entity`: `Optional[str]` - nazwa entity/team
  - `wandb_name`: `Optional[str]` - nazwa runa (domyślnie run_id jeśli nie ustawione)
  - `wandb_tags`: `Optional[list[str]]` - dodatkowe tagi
  - `wandb_config`: `Optional[dict[str, Any]]` - dodatkowa konfiguracja
  - `wandb_mode`: `str` - tryb wandb: "online", "offline", "disabled" (domyślnie "online")
  - `wandb_slow_metrics_frequency`: `int` - częstotliwość logowania wolnych metryk (L0, dead features) (domyślnie 50)

**AutoencoderContext** (`src/mi_crow/mechanistic/sae/autoencoder_context.py`):
- Dataclass z kontekstem SAE
- Parametry:
  - `autoencoder`: `Sae` - instancja SAE
  - `n_latents`: `int` - liczba neuronów latentnych
  - `n_inputs`: `int` - liczba wejść
  - `lm`: `Optional[LanguageModel]` - opcjonalny model językowy
  - `lm_layer_signature`: `Optional[int | str]` - sygnatura warstwy LM
  - `model_id`: `Optional[str]` - identyfikator modelu
  - `device`: `str` (domyślnie 'cpu')
  - `experiment_name`: `Optional[str]`
  - `run_id`: `Optional[str]`
  - `text_tracking_enabled`: `bool` (domyślnie `False`)
  - `text_tracking_k`: `int` (domyślnie 5)
  - `text_tracking_negative`: `bool` (domyślnie `False`)
  - `store`: `Optional[Store]`
  - `tied`: `bool` (domyślnie `False`)
  - `bias_init`: `float` (domyślnie 0.0)
  - `init_method`: `str` (domyślnie "kaiming")

**AutoencoderConcepts** (`src/mi_crow/mechanistic/sae/concepts/autoencoder_concepts.py`):
- Zarządzanie konceptami SAE
- Parametry manipulacji konceptami:
  - `multiplication`: `nn.Parameter` - mnożnik dla każdego neuronu (domyślnie ones)
  - `bias`: `nn.Parameter` - bias dla każdego neuronu (domyślnie ones)
- Metody:
  - `enable_text_tracking()` - włączenie śledzenia tekstów
  - `disable_text_tracking()` - wyłączenie śledzenia tekstów
- Atrybut: `dictionary: ConceptDictionary | None` - słownik konceptów

**ConceptDictionary** (`src/mi_crow/mechanistic/sae/concepts/concept_dictionary.py`):
- Słownik konceptów z neuronami i ich tekstami

### 2.5 Moduł mi_crow.store

**Store** (`src/mi_crow/store/store.py`):
- Klasa abstrakcyjna do przechowywania tensorów
- Parametry inicjalizacji:
  - `base_path`: `Path | str` - bazowa ścieżka katalogu (domyślnie "")
  - `runs_prefix`: `str` - prefix dla katalogu runs (domyślnie "runs")
  - `dataset_prefix`: `str` - prefix dla katalogu datasets (domyślnie "datasets")
  - `model_prefix`: `str` - prefix dla katalogu models (domyślnie "models")
- Organizacja danych hierarchiczna:
  - Runs: grupowanie najwyższego poziomu według `run_id`
  - Batches: w ramach każdego runa, dane zorganizowane według `batch_index`
  - Layers: w ramach każdego batcha, tensory zorganizowane według `layer_signature`
  - Keys: w ramach każdej warstwy, tensory identyfikowane przez klucz (np. "activations")
- Metody abstrakcyjne:
  - `put_tensor(key, tensor)` - zapis pojedynczego tensora
  - `get_tensor(key)` - odczyt pojedynczego tensora
  - `put_detector_metadata(run_id, batch_index, metadata, tensor_metadata)` - zapis metadanych detektora
  - `get_detector_metadata(run_id, batch_index)` - odczyt metadanych detektora
  - `get_detector_metadata_by_layer_by_key(run_id, batch_index, layer, key)` - odczyt konkretnego tensora

**LocalStore** (`src/mi_crow/store/local_store.py`):
- Implementacja lokalnego przechowywania dziedzicząca z `Store`
- Używa `safetensors.torch` do zapisu tensorów
- Metadane zapisywane jako pliki JSON
- Tensory zapisywane jako pliki safetensors
- Struktura katalogów: `base_path/runs_prefix/run_id/batch_index/layer_signature/`

**StoreDataloader** (`src/mi_crow/store/store_dataloader.py`):
- DataLoader-like klasa do iteracji po aktywacjach w Store
- Parametry:
  - `store`: `Store` - instancja Store
  - `run_id`: `str` - ID runa
  - `layer`: `str` - sygnatura warstwy
  - `key`: `str` - klucz tensora (domyślnie "activations")
  - `batch_size`: `int` (domyślnie 32)
  - `dtype`: `Optional[torch.dtype]` - opcjonalny dtype do castowania
  - `max_batches`: `Optional[int]` - opcjonalny limit liczby batchy na epokę
- Może być iterowany wielokrotnie (wymagane przez `overcomplete.train_sae`)




