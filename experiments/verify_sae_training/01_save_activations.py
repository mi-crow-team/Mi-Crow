import logging
import torch
from pathlib import Path
from datetime import datetime

from mi_crow.datasets import TextDataset
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = get_logger(__name__)

MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct"
HF_DATASET = "roneneldan/TinyStories"
DATA_SPLIT = "train"
TEXT_FIELD = "text"
DATA_LIMIT = 1000
MAX_LENGTH = 128
BATCH_SIZE_SAVE = 16

LAYER_SIGNATURE = "llamaforcausallm_model_layers_16_post_attention_layernorm"

STORE_DIR = Path(__file__).parent / "store"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    RUN_ID = f"activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("🚀 Starting Activation Saving")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📊 Dataset: {HF_DATASET}")
    logger.info(f"💾 Run ID: {RUN_ID}")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store)
    lm.model.to(DEVICE)

    logger.info(f"✅ Model loaded: {lm.model_id}")
    logger.info(f"📱 Device: {DEVICE}")
    logger.info(f"📁 Store location: {lm.store.base_path}")
    logger.info(f"🎯 Target layer: {LAYER_SIGNATURE}")
    logger.info(f"📍 Layer type: resid_mid (after attention, before MLP)")

    logger.info("📥 Loading dataset...")
    dataset = TextDataset.from_huggingface(
        HF_DATASET,
        split=DATA_SPLIT,
        store=store,
        text_field=TEXT_FIELD,
        limit=DATA_LIMIT,
    )
    logger.info(f"✅ Loaded {len(dataset)} text samples")

    logger.info("💾 Saving activations...")
    run_name = lm.activations.save_activations_dataset(
        dataset,
        layer_signature=LAYER_SIGNATURE,
        run_name=RUN_ID,
        batch_size=BATCH_SIZE_SAVE,
        max_length=MAX_LENGTH,
        autocast=False,
        verbose=True,
    )

    batches = lm.store.list_run_batches(run_name)
    logger.info(f"✅ Activations saved!")
    logger.info(f"📁 Run name: {run_name}")
    logger.info(f"📦 Saved {len(batches)} batches to store")
    logger.info(f"💾 Run ID saved to: {STORE_DIR / 'run_id.txt'}")
    with open(STORE_DIR / "run_id.txt", "w") as f:
        f.write(run_name)


if __name__ == "__main__":
    main()
