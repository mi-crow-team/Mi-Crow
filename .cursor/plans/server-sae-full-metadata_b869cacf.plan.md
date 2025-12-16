---
name: server-sae-full-metadata
overview: "Wire server to amber SAE without touching core library: LocalStore-based activations, real SAE training/inference, concept application, and enriched responses."
todos: []
---

# Server SAE Integration (No Core Library Changes)

## Scope

b

- Only modify server code; reuse amber APIs as-is.
- Use LocalStore for activations, real SaeTrainer for training, and Sae hook for inference/top-texts.
- Concept handling via existing amber concept classes; no edits to amber package.

## Plan

1) **Activation Pipeline → LocalStore**

- Update `/sae/activations/save` to write activations into `amber.store.LocalStore` under `artifact_base_path` using `run_id`/batch/layer keys.
- Keep manifest with run_id, layers, dataset info, token counts, batch indices.
- Add layer-existence check (already present) and bounds on batch/sample size.

2) **SAE Training (real)**

- In `/sae/train`, load LocalStore with run_id/layer from manifest; build `SaeTrainingConfig` from payload; call `SaeTrainer.train` on the selected SAE class (chosen by payload or default) without modifying amber code.
- Save SAE checkpoint via its `save` method alongside training metadata (history, config, metrics) in `artifact_base_path/sae/<model>/<run>/`.
- Return sae_id/sae_path/metadata_path from the job result.

3) **SAE Inference & Top Texts**

- In `/sae/infer`, load SAE via `Sae.load`, register the SAE hook on target layer, run the LM forward; capture neuron activations from the SAE detector metadata.
- Compute top-N neurons per prompt and write top-texts JSON (neuron_id, activation stats, example texts) to `top_texts_dir`.
- Response: generated text/tokens, optional logits/probs, top-neuron summary + path to top-texts file, sae_id reference.

4) **Concepts Integration**

- `/sae/concepts/load`: parse user file into amber concepts/dictionary; validate against SAE dims; persist normalized concept file under concepts dir.
- `/sae/concepts/manipulate`: accept weights/edits; build a concept-config mapping to neuron weights; persist config.
- `/sae/infer` accepts concept_config to adjust activations/decoder via SAE hook before generation (no amber code changes—use available hook points).

5) **Jobs & Observability**

- Keep in-memory JobManager; ensure train job result carries sae_id/paths/metadata. Add minimal logging around train start/end and inference timings.

6) **Testing & Docs**

- Extend server tests with small dummy SAE class implementing `save/load/encode/decode` to cover training/inference flows without touching amber; mock LocalStore I/O.
- Update README with new expectations: activations in LocalStore, real training path, concept flow, top-text outputs, and job responses.

## Notes

- No changes to amber core; all integrations via its public classes (LocalStore, SaeTrainer, Sae, concepts).
- Keep payload schemas mostly stable; only enrich responses with sae/metadata paths and top-text summaries.