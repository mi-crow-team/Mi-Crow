---
name: Plan raportu implementacji modułów i automatyzacji
overview: ""
todos:
  - id: 41a71e17-1a66-482a-927c-8b8739cedd36
    content: Utworzenie pliku docs/automation.md z opisem wszystkich procedur automatyzacji (UV, pre-commit, ruff, pytest, GitHub Actions, MkD
    status: pending
  - id: ab21d40e-3967-484a-8a10-8dd60a424cb5
    content: Utworzenie pliku docs/modules.md z dokumentacją wszystkich modułów, ich parametrów i konfiguracjiAktualizacja mkdocs.yml - dodanie nowych sekcji do nawigacjiAktualizacja docs/index.md - dodanie linków do nowych sekcji dokumentacjiUtworzenie pliku docs/automation.md z opisem wszystkich procedur automatyzacji (UV, pre-commit, ruff, pytest, GitHub Actions, MkDocs)Utworzenie pliku docs/modules.md z dokumentacją wszystkich modułów, ich parametrów i konfiguracjiAktualizacja mkdocs.yml - dodanie nowych sekcji do nawigacjiAktualizacja docs/index.md - dodanie linków do nowych sekcji dokumentacUtworzenie pliku docs/automation.md z opisem wszystkich procedur automatyzacji (UV, pre-commit, ruff, pytest, GitHub Actions, MkDocs)Utworzenie pliku docs/modules.md z dokumentacją wszystkich modułów, ich parametrów i konfiguracjiAktualizacja mkdocs.yml - dodanie nowych sekcji do nawigacjiAktualizacja docs/index.md - dodanie linków do nowych sekcji dokumentacji
    status: pending
---

# Plan raportu implementacji modułów i automatyzacji

## Cel

Utworzenie jednego pliku Markdown z raportem na zajęcia opisującym:

1. Procedury automatyzacji (GitHub Actions, UV, pre-commit, ruff, pytest, środowisko testowe)
2. Implementację modułów (stan implementacji, parametry, opisy modeli ML)

## Struktura raportu

Jeden plik Markdown (np. `raport.md` lub `raport_implementacji.md`) w katalogu głównym projektu z następującymi sekcjami:

## 1. Procedury automatyzacji

### 1.1 Zarządzanie zależnościami (UV)

- Opis użycia UV jako menedżera pakietów
- Konfiguracja w `pyproject.toml`
- Komendy: `uv sync`, `uv add`, `uv lock`
- Grupy zależności: `dev`, `docs`
- Plik `uv.lock` i synchronizacja wersji

### 1.2 Pre-commit hooks

- Konfiguracja w `.pre-commit-config.yaml`
- Hooks: `ruff` (linter) i `ruff-format` (formatter)
- Automatyczne uruchamianie przed commitami
- Instalacja: `pre-commit install`

### 1.3 Ruff - linting i formatowanie

- Konfiguracja w `pyproject.toml` sekcja `[tool.ruff]`
- Ustawienia: line-length=120, target-version, select rules
- Uruchamianie: `uv run ruff check`, `uv run ruff format`

### 1.4 Testy i coverage (pytest)

- Konfiguracja w `pyproject.toml` sekcja `[tool.pytest.ini_options]`
- Wymagane pokrycie: 90% (fail-under=90)
- Markery testów: `--unit`, `--e2e`
- Raporty: terminal, XML, HTML
- Środowisko testowe: fixtures w `conftest.py`, testy jednostkowe i e2e

### 1.5 GitHub Actions

- Opis workflow CI/CD (jeśli dostępne)
- Badge CI widoczny w README.md
- Automatyczne testy i deployment (jeśli skonfigurowane)

### 1.6 Dokumentacja (MkDocs)

- Konfiguracja w `mkdocs.yml`
- Theme: Material
- Plugins: mkdocstrings, search, section-index
- Deploy: GitHub Pages (https://adamkaniasty.github.io/Inzynierka/)

## 2. Implementacja modułów

### 2.1 Moduł `amber.datasets`

- **BaseDataset**: Klasa abstrakcyjna z LoadingStrategy (MEMORY, DYNAMIC_LOAD, ITERABLE_ONLY)
- **TextDataset**: Dataset tekstowy z parametrem `text_field`
- **ClassificationDataset**: Dataset z kategoriami, parametry `text_field`, `category_field` (single/multiple)
- **LoadingStrategy**: Enum z trzema strategiami ładowania

### 2.2 Moduł `amber.hooks`

- **Hook**: Klasa abstrakcyjna, parametry: `layer_signature`, `hook_type` (FORWARD/PRE_FORWARD), `hook_id`
- **Detector**: Hook do wykrywania/zapisywania aktywacji
- **Controller**: Hook do modyfikacji aktywacji
- **LayerActivationDetector**: Implementacja zapisywania aktywacji warstw
- **FunctionController**: Implementacja kontrolera funkcji

### 2.3 Moduł `amber.language_model`

- **LanguageModel**: Główna klasa wrappera modelu
- Parametry inicjalizacji: `model`, `tokenizer`, `store`, `model_id`
- Metody: `tokenize()`, `forwards()`, `generate()`, `save_model()`
- Factory methods: `from_huggingface()`, `from_local_torch()`, `from_local()`
- **LanguageModelLayers**: Zarządzanie warstwami i hookami
- **LanguageModelActivations**: Zarządzanie aktywacjami
- **LanguageModelTokenizer**: Wrapper tokenizera
- **InferenceEngine**: Silnik inferencji z parametrami `autocast`, `autocast_dtype`, `with_controllers`

### 2.4 Moduł `amber.mechanistic.sae` (moduły uczenia maszynowego)

- **Sae**: Klasa abstrakcyjna SAE
- Parametry: `n_latents`, `n_inputs`, `hook_id`, `device`, `store`
- Metody abstrakcyjne: `encode()`, `decode()`, `forward()`, `modify_activations()`, `save()`
- **TopKSae**: Implementacja TopK SAE
- Dodatkowy parametr: `k` (liczba aktywnych neuronów)
- Metody: `train()`, `save()`, `load()`
- **SaeTrainer**: Klasa trenująca SAE
- **SaeTrainingConfig**: Konfiguracja treningu
- Parametry treningu: `epochs`, `batch_size`, `lr`, `l1_lambda`, `device`, `dtype`
- Parametry zaawansowane: `use_amp`, `amp_dtype`, `grad_accum_steps`, `clip_grad`, `monitoring`
- Parametry wandb: `use_wandb`, `wandb_project`, `wandb_entity`, `wandb_name`, `wandb_tags`, `wandb_mode`
- **AutoencoderContext**: Kontekst SAE z parametrami: `n_latents`, `n_inputs`, `device`, `text_tracking_enabled`, `text_tracking_k`
- **AutoencoderConcepts**: Zarządzanie konceptami (multiplication, bias)

### 2.5 Moduł `amber.store`

- **Store**: Klasa abstrakcyjna do przechowywania tensorów
- Parametry: `base_path`, `runs_prefix`, `dataset_prefix`, `model_prefix`
- Organizacja: runs -> batches -> layers -> keys
- **LocalStore**: Implementacja lokalnego przechowywania (safetensors)