import contextlib
import io
import os
import sys
from pathlib import Path

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store.local_store import LocalStore


def main():
    # Setup store
    STORE_DIR = Path("store")
    store = LocalStore(base_path=STORE_DIR)

    # Models to inspect
    MODELS = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "speakleash/Bielik-4.5B-v3.0-Instruct",
        "speakleash/Bielik-1.5B-v3.0-Instruct",
    ]

    # Output directory
    output_dir = STORE_DIR / "runs" / "layer_names"
    os.makedirs(output_dir, exist_ok=True)

    errors = []

    for model_id in MODELS:
        try:
            print(f"\n[INFO] Loading model: {model_id}", file=sys.stderr)
            lm = LanguageModel.from_huggingface(model_id, store=store)

            output_lines = [
                "=" * 80,
                f"Model: {model_id}",
                "=" * 80,
                "",
            ]

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lm.layers.print_layer_names()
            layer_names_str = buf.getvalue()
            output_lines.append(layer_names_str.rstrip())
            output_lines.append("")

            model_filename = model_id.replace("/", "__") + "_layers.txt"
            output_path = output_dir / model_filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            print(f"[INFO] Saved layer names to {output_path}", file=sys.stderr)
        except Exception as e:
            import traceback

            err_msg = f"[ERROR] Failed for model {model_id}: {e}\n{traceback.format_exc()}"
            print(err_msg, file=sys.stderr)
            errors.append((model_id, str(e)))

    if errors:
        print("\n[SUMMARY] Some models failed:", file=sys.stderr)
        for model_id, err in errors:
            print(f"  - {model_id}: {err}", file=sys.stderr)
    else:
        print("\n[SUMMARY] All models processed successfully.", file=sys.stderr)


if __name__ == "__main__":
    main()
