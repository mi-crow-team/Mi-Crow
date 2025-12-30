import torch
from pathlib import Path

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from mi_crow.store.local_store import LocalStore


def main() -> None:
    """
    Small demo for the "music / hearing-music" neuron.

    - Loads Bielik LM and the trained TopKSae from this experiment's store.
    - Attaches the SAE as a controller on layer 16 post-attention layernorm.
    - Runs one baseline generation (no controllers).
    - Runs one generation with neuron 12 amplified.
    - Prints both generations for side-by-side comparison.
    """

    # --- Config ---
    model_id = "speakleash/Bielik-1.5B-v3.0-Instruct"
    layer_signature = "llamaforcausallm_model_layers_16_post_attention_layernorm"

    # Path to the trained SAE from 02_train_sae.py (meta.json confirms this run)
    sae_run_dir = Path(__file__).parent / "store" / "runs" / "training_20251213_005514"
    sae_model_path = sae_run_dir / "model.pt"

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # "Music" neuron index discovered from top_texts.json
    music_neuron_idx = 12
    amplification_factor = 10.0  # >1 boosts the neuron; 0 would knock it out

    # Prompt closely mirroring the top-activating story for neuron 12
    prompt = (
        "Once upon a time, there was a girl named Lily who loved to play the flute. "
        "One day she met a deaf cat who could not hear her play. "
        "Write a detailed story about how Lily finds a way for the cat to feel the music."
    )

    print(f"Using device: {device}")

    # --- Load language model ---
    print("Loading language model...")
    store_base = Path(__file__).parent / "store"
    store = LocalStore(base_path=store_base)
    lm = LanguageModel.from_huggingface(model_id, store=store)
    lm.model.to(device)
    print(f"Model loaded: {lm.model_id}")

    # --- Load trained TopKSae ---
    print("Loading trained TopKSae...")
    if not sae_model_path.exists():
        raise FileNotFoundError(f"SAE model not found at: {sae_model_path}")

    sae_hook: TopKSae = TopKSae.load(sae_model_path)
    sae_hook.sae_engine.to(device)
    print(
        f"TopKSae loaded: {sae_hook.context.n_inputs} -> "
        f"{sae_hook.context.n_latents} (k={sae_hook.k})"
    )

    # --- Attach SAE as controller on the target layer ---
    print(f"Registering SAE hook on layer: {layer_signature}")
    lm.layers.register_hook(layer_signature, sae_hook)
    sae_hook.context.lm = lm
    sae_hook.context.lm_layer_signature = layer_signature

    # --- Helper: reset concept manipulation to neutral (no effect) ---
    def reset_manipulation() -> None:
        with torch.no_grad():
            sae_hook.concepts.multiplication[:] = 1.0
            sae_hook.concepts.bias[:] = 0.0

    reset_manipulation()

    # --- 1) Baseline generation (no controllers) ---
    print("\n=== BASELINE (no SAE controllers) ===")
    baseline_outputs = lm.generate(
        [prompt],
        with_controllers=False,  # <- no SAE modifications
        autocast=False,
    )
    baseline_text = baseline_outputs[0]
    print(baseline_text)

    # --- 2) Manipulated generation (music neuron amplified) ---
    print("\n=== MANIPULATED (music neuron amplified) ===")

    reset_manipulation()
    with torch.no_grad():
        sae_hook.concepts.multiplication[music_neuron_idx] = amplification_factor

    manipulated_outputs = lm.generate(
        [prompt],
        with_controllers=True,  # <- SAE active
        autocast=False,
    )
    manipulated_text = manipulated_outputs[0]
    print(manipulated_text)

    # --- Simple side-by-side printout ---
    print("\n" + "=" * 80)
    print("PROMPT:")
    print(prompt)
    print("\n--- BASELINE ---\n")
    print(baseline_text)
    print("\n--- MANIPULATED (music neuron amplified) ---\n")
    print(manipulated_text)
    print("=" * 80)


if __name__ == "__main__":
    main()


