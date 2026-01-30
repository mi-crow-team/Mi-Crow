<div align="center">
  <img src="docs/logo-white.svg" alt="Mi-Crow Logo" width="200">
</div>

# Mi-Crow: Mechanistic Interpretability for Large Language Models

[![CI](https://github.com/AdamKaniasty/Inzynierka/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AdamKaniasty/Inzynierka/actions)
[![PyPI](https://img.shields.io/pypi/v/mi-crow)](https://pypi.org/project/mi-crow/)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](http://mi-crow-team.github.io/Mi-Crow/)

**Mi-Crow** is a Python library for mechanistic interpretability research on Large Language Models (LLMs). Designed for researchers, it provides a unified interface for analyzing and controlling model behavior through Sparse Autoencoders (SAEs), activation hooks, and concept manipulation.

## Features

- **Unified Model Interface** - Work with any HuggingFace language model through a consistent API
- **Sparse Autoencoder Training** - Train SAEs to extract interpretable features from model activations
- **Hook System** - Intercept and manipulate model activations with minimal performance overhead
- **Concept Discovery & Manipulation** - Discover and control model behavior through learned concepts
- **Hierarchical Data Persistence** - Efficient storage and management of large-scale experiment data
- **Research Focused** - Comprehensive testing (85%+ coverage), extensive documentation, and designed for interpretability research workflows

## Installation

### Install from PyPI

```bash
pip install mi-crow
```

### Install from Source

```bash
git clone https://github.com/AdamKaniasty/Mi-Crow.git
cd Mi-Crow
pip install -e .
```

### Requirements

- **Python 3.12+** (required for modern type hints and features)
- **PyTorch** - Tensor operations and neural networks
- **Transformers** - Model loading and tokenization
- **Accelerate** - Distributed and mixed-precision training
- **Datasets** - Dataset loading and processing
- **overcomplete** - SAE implementations

## Quick Start

### Basic Usage

```python
from mi_crow.language_model import LanguageModel

# Initialize a language model
lm = LanguageModel(model_id="bielik")

# Run inference
outputs = lm.forwards(["Hello, world!"])

# Access activations and outputs
print(outputs.logits)
```

### Training an SAE

```python
from mi_crow.language_model import LanguageModel
from mi_crow.mechanistic.sae import SaeTrainer
from mi_crow.mechanistic.sae.modules import TopKSae

# Load model and collect activations
lm = LanguageModel(model_id="bielik")
activations = lm.save_activations(
    dataset=["Your text data here"],
    layers=["transformer_h_0_attn_c_attn"]
)

# Train SAE
trainer = SaeTrainer(
    model=lm,
    layer="transformer_h_0_attn_c_attn",
    sae_class=TopKSae,
    hyperparams={"epochs": 10, "batch_size": 256}
)
sae = trainer.train(activations)
```

### Concept Manipulation

```python
# Load concepts and manipulate model behavior
concepts = lm.load_concepts(sae_id="your_sae_id")
concepts.manipulate(neuron_idx=0, scale_factor=1.5)

# Run inference with concept manipulation
outputs = lm.forwards(
    ["Your prompt"],
    with_controllers=True,
    concept_config=concepts
)
```

## Documentation

- **Full Documentation**: [adamkaniasty.github.io/Inzynierka](https://adamkaniasty.github.io/Inzynierka/)
- **GitHub Repository**: [github.com/AdamKaniasty/Mi-Crow](https://github.com/AdamKaniasty/Mi-Crow/)
- **Example Notebooks**: See `examples/` directory for Jupyter notebook tutorials

## Architecture

Mi-Crow follows a modular design with five core components:

1. **`language_model/`** - Unified interface for language models
   - Model initialization from HuggingFace Hub or local files
   - Unified inference interface with mixed-precision support
   - Architecture-agnostic layer abstraction

2. **`hooks/`** - Flexible hook system for activation interception
   - Detectors for observing activations
   - Controllers for modifying model behavior
   - Support for FORWARD and PRE_FORWARD hooks

3. **`mechanistic/`** - SAE training and concept manipulation
   - Sparse Autoencoder training (TopK, L1 variants)
   - Concept dictionary management
   - Concept-based model steering

4. **`store/`** - Hierarchical data persistence
   - Efficient tensor storage in safetensors format
   - Batch iteration for large datasets
   - Metadata management

5. **`datasets/`** - Dataset loading and processing
   - HuggingFace dataset integration
   - Local file dataset support

## Example Workflow

See the example notebooks in the `examples/` directory:

1. **`01_train_sae_model.ipynb`** - Train an SAE on model activations
2. **`02_attach_sae_and_save_texts.ipynb`** - Collect top activating texts
3. **`03_load_concepts.ipynb`** - Load and manipulate concepts

## Development

### Running Tests

The project uses pytest for testing. Tests are organized into unit tests and end-to-end tests.

### Running All Tests

```bash
pytest
```

### Running Specific Test Suites

Run only unit tests:
```bash
pytest --unit -q
```

Run only end-to-end tests:
```bash
pytest --e2e -q
```

You can also use pytest markers:
```bash
pytest -m unit -q
pytest -m e2e -q
```

Or specify the test directory directly:
```bash
pytest tests/unit -q
pytest tests/e2e -q
```

### Test Coverage

The test suite is configured to require at least 85% code coverage. Coverage reports are generated in both terminal and XML formats.

The project maintains **85%+ code coverage** requirement.

### Code Quality

- **Linting**: Ruff for code formatting and linting
- **Pre-commit Hooks**: Automated quality checks
- **Type Hints**: Extensive use of Python type annotations
- **CI/CD**: GitHub Actions for automated testing and deployment

## Citation

If you use Mi-Crow in your research, please cite:

```bibtex
@thesis{kaniasty2025microw,
  title={Mechanistic Interpretability for Large Language Models: A Production-Ready Framework},
  author={Kaniasty, Adam and Kowalski, Hubert},
  year={2025},
  school={Warsaw University of Technology},
  note={Engineering Thesis}
}
```

## License

See the main repository for license information: [Mi-Crow License](https://github.com/AdamKaniasty/Mi-Crow/)

## Contact

- **Maintainers**: Adam Kaniasty, Hubert Kowalski
- **Email**: adam.kaniasty@gmail.com
- **GitHub**: [@AdamKaniasty](https://github.com/AdamKaniasty)

## Acknowledgments

This work was developed in collaboration with the **Bielik** team and represents a contribution to the open-source mechanistic interpretability community.

---

## Backend (FastAPI) quickstart

Install server-only dependencies (kept out of the core library) with uv:
```bash
uv sync --group server
```

Run the API:
```bash
uv run --group server uvicorn server.main:app --reload
```

Smoke-test the server endpoints:
```bash
uv run --group server pytest tests/server/test_api.py --cov=server --cov-fail-under=0
```

### SAE API usage

- Configure artifact location (optional): `export SERVER_ARTIFACT_BASE_PATH=/path/to/mi_crow_artifacts` (defaults to `~/.cache/mi_crow_server`)
- Load a model: `curl -X POST http://localhost:8000/models/load -H "Content-Type: application/json" -d '{"model_id":"bielik"}'`
- Save activations from dataset (stored in `LocalStore` under `activations/<model>/<run_id>`):
  - HF dataset: `{"dataset":{"type":"hf","name":"ag_news","split":"train","text_field":"text"}}`
  - Local files: `{"dataset":{"type":"local","paths":["/path/to/file.txt"]}}`
  - Example: `curl -X POST http://localhost:8000/sae/activations/save -H "Content-Type: application/json" -d '{"model_id":"bielik","layers":["dummy_root"],"dataset":{"type":"local","paths":["/tmp/data.txt"]},"sample_limit":100,"batch_size":4,"shard_size":64}'` → returns a manifest path, run_id, token counts, and batch metadata.
- List activation runs: `curl "http://localhost:8000/sae/activations?model_id=bielik"`
- Start SAE training (async job, uses `SaeTrainer`): `curl -X POST http://localhost:8000/sae/train -H "Content-Type: application/json" -d '{"model_id":"bielik","activations_path":"/path/to/manifest.json","layer":"<layer_name>","sae_class":"TopKSae","hyperparams":{"epochs":1,"batch_size":256}}'` → returns `job_id`
- Check job status: `curl http://localhost:8000/sae/train/status/<job_id>` (returns `sae_id`, `sae_path`, `metadata_path`, progress, and logs)
- Cancel a job (best-effort): `curl -X POST http://localhost:8000/sae/train/cancel/<job_id>`
- Load an SAE: `curl -X POST http://localhost:8000/sae/load -H "Content-Type: application/json" -d '{"model_id":"bielik","sae_path":"/path/to/sae.json"}'`
- List SAEs: `curl "http://localhost:8000/sae/saes?model_id=bielik"`
- Run SAE inference (optionally save top texts and apply concept config): `curl -X POST http://localhost:8000/sae/infer -H "Content-Type: application/json" -d '{"model_id":"bielik","sae_id":"<sae_id>","save_top_texts":true,"top_k_neurons":5,"concept_config_path":"/path/to/concepts.json","inputs":[{"prompt":"hi"}]}'` → returns outputs, top neuron summary, sae metadata, and saved top-texts path when requested.
- Per-token latents: add `"return_token_latents": true` (default off) to include top-k neuron activations per token.
- List concepts: `curl "http://localhost:8000/sae/concepts?model_id=bielik&sae_id=<sae_id>"`
- Load concepts from a file (validated against SAE latents): `curl -X POST http://localhost:8000/sae/concepts/load -H "Content-Type: application/json" -d '{"model_id":"bielik","sae_id":"<sae_id>","source_path":"/path/to/concepts.json"}'`
- Manipulate concepts (saves a config file for inference-time scaling): `curl -X POST http://localhost:8000/sae/concepts/manipulate -H "Content-Type: application/json" -d '{"model_id":"bielik","sae_id":"<sae_id>","edits":{"0":1.2}}'`
- List concept configs: `curl "http://localhost:8000/sae/concepts/configs?model_id=bielik&sae_id=<sae_id>"`
- Preview concept config (validate without saving): `curl -X POST http://localhost:8000/sae/concepts/preview -H "Content-Type: application/json" -d '{"model_id":"bielik","sae_id":"<sae_id>","edits":{"0":1.2}}'`
- Delete activation run or SAE (requires API key if set): `curl -X DELETE "http://localhost:8000/sae/activations/<run_id>?model_id=bielik" -H "X-API-Key: <key>"` and `curl -X DELETE "http://localhost:8000/sae/saes/<sae_id>?model_id=bielik" -H "X-API-Key: <key>"`
- Health/metrics summary: `curl http://localhost:8000/health/metrics` (in-memory job counts; no persistence, no auth)

Notes:
- Job manager is in-memory/lightweight: jobs disappear on process restart; idempotency is best-effort via payload key.
- Training/inference currently run in-process threads; add your own resource guards when running heavy models.
- Optional API key protection: set `SERVER_API_KEY=<value>` to require `X-API-Key` on protected endpoints (delete).