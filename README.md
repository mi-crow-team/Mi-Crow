![CI](https://github.com/AdamKaniasty/Inzynierka/actions/workflows/tests.yml/badge.svg?branch=main)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://adamkaniasty.github.io/Inzynierka/)

## Running Tests

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