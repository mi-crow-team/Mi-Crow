import torch
from pathlib import Path

from amber.language_model.language_model import LanguageModel
from amber.mechanistic.sae.modules.topk_sae import TopKSae, TopKSaeTrainingConfig
from amber.store.local_store import LocalStore
from amber.utils import get_logger

logger = get_logger(__name__)

MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct"
STORE_DIR = Path(__file__).parent / "store"

N_LATENTS_MULTIPLIER = 4
TOP_K = 8

EPOCHS = 10
BATCH_SIZE_TRAIN = 32
LR = 1e-3
L1_LAMBDA = 1e-4

DEVICE = "mps"
DTYPE = torch.float16 if torch.cuda.is_available() else None


def main():
    logger.info("🚀 Starting SAE Training")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store)
    lm.model.to(DEVICE)

    logger.info(f"✅ Model loaded: {lm.model_id}")

    run_id_file = STORE_DIR / "run_id.txt"
    if not run_id_file.exists():
        logger.error("❌ Error: run_id.txt not found. Please run 01_save_activations.py first.")
        return

    with open(run_id_file, "r") as f:
        RUN_ID = f.read().strip()

    logger.info(f"📁 Using run ID: {RUN_ID}")

    batches = store.list_run_batches(RUN_ID)
    if not batches:
        logger.error(f"❌ Error: No batches found for run ID: {RUN_ID}")
        return

    logger.info(f"📦 Found {len(batches)} batches")

    sample_batch = store.get_run_batch(RUN_ID, batches[0])
    if isinstance(sample_batch, dict):
        activations = sample_batch.get("activations")
    elif isinstance(sample_batch, list):
        activations = sample_batch[0] if sample_batch else None
    else:
        activations = sample_batch

    if activations is None:
        logger.error("❌ Error: Could not find activations in batch")
        return

    if isinstance(activations, torch.Tensor):
        n_inputs = activations.shape[-1]
    else:
        logger.error("❌ Error: Activations is not a tensor")
        return

    n_latents = n_inputs * N_LATENTS_MULTIPLIER

    logger.info(f"📊 Activation dimension: {n_inputs}")
    logger.info(f"📊 SAE latents: {n_latents} ({N_LATENTS_MULTIPLIER}x overcomplete)")
    logger.info(f"📊 TopK: {TOP_K}")

    layer_signature = store.get_run_metadata(RUN_ID).get("layer_signatures")[0]

    logger.info(f"🎯 Target layer: {layer_signature}")

    logger.info("🏗️  Creating TopKSAE...")
    sae = TopKSae(
        n_latents=n_latents,
        n_inputs=n_inputs,
        k=TOP_K,
        device=DEVICE,
        store=store,
    )
    logger.info(f"✅ SAE created: {n_inputs} -> {n_latents} (TopK={TOP_K})")

    logger.info("🏋️ Training TopKSAE...")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Batch size: {BATCH_SIZE_TRAIN}")
    logger.info(f"   Learning rate: {LR}")
    logger.info(f"   L1 lambda: {L1_LAMBDA}")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Monitoring: detailed (level 2)")
    logger.info(f"   Memory efficient: True")

    use_amp = DEVICE != "cpu"
    if not use_amp:
        logger.warning(f"   ⚠️  AMP disabled for CPU device")

    config = TopKSaeTrainingConfig(
        k=TOP_K,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        lr=LR,
        l1_lambda=L1_LAMBDA,
        device=DEVICE,
        dtype=DTYPE,
        verbose=True,
        use_amp=use_amp,
        amp_dtype=DTYPE if use_amp else None,
        clip_grad=1.0,
        monitoring=2,
        memory_efficient=True,
    )

    logger.info("📦 Checking dataloader...")
    from amber.store.store_dataloader import StoreDataloader
    test_dataloader = StoreDataloader(
        store=store,
        run_id=RUN_ID,
        layer=layer_signature,
        key="activations",
        batch_size=BATCH_SIZE_TRAIN,
        dtype=DTYPE,
        max_batches=1,
    )
    logger.info("   Testing dataloader with 1 batch...")
    try:
        test_batch = next(iter(test_dataloader))
        logger.info(
            f"   ✅ Dataloader works! Batch shape: {test_batch.shape if hasattr(test_batch, 'shape') else type(test_batch)}")
    except Exception as e:
        logger.error(f"   ❌ Dataloader error: {e}")
        return

    logger.info("🚀 Starting training (this may take a while)...")

    result = sae.train(store, RUN_ID, layer_signature, config)
    training_run_id = result.get('training_run_id')

    logger.info("✅ Training function returned!")

    if training_run_id:
        logger.info(f"💾 Training outputs saved to: store/runs/{training_run_id}/")
        logger.info(f"   - Model: store/runs/{training_run_id}/model.pt")
        logger.info(f"   - History: store/runs/{training_run_id}/history.json")
        logger.info(f"   - Metadata: store/runs/{training_run_id}/meta.json")

if __name__ == "__main__":
    main()
