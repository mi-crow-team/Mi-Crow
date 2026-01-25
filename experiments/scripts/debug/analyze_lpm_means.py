# ruff: noqa
import argparse
from pathlib import Path

import torch


def analyze_lpm_model(model_path: str):
    print(f"Loading LPM model from: {model_path}")
    path = Path(model_path)
    if not path.exists():
        print(f"Error: File not found at {model_path}")
        return

    # Load state
    state = torch.load(path, map_location="cpu")

    # Metadata / Config
    config = state.get("config", {})
    print("\n" + "=" * 50)
    print("MODEL CONFIGURATION")
    print("=" * 50)
    for k, v in config.items():
        print(f"{k:20}: {v}")

    # Prototypes
    prototypes = state.get("prototypes", {})
    if not prototypes:
        print("\nNo prototypes found in the model!")
        return

    print("\n" + "=" * 50)
    print("PROTOTYPE ANALYSIS")
    print("=" * 50)

    labels = list(prototypes.keys())
    print(f"Found {len(labels)} classes: {labels}")

    proto_tensors = {}
    for label in labels:
        tensor = prototypes[label]
        proto_tensors[label] = tensor

        # Basic stats
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        norm_val = torch.norm(tensor).item()

        print(f"\nClass: {label}")
        print(f"  Shape: {list(tensor.shape)}")
        print(f"  Norm:  {norm_val:.4f}")
        print(f"  Mean:  {mean_val:.4f}")
        print(f"  Std:   {std_val:.4f}")
        print(f"  Min/Max: {tensor.min().item():.4f} / {tensor.max().item():.4f}")

        # Check for anomalies
        if torch.isnan(tensor).any():
            print("  WARNING: Contains NaNs!")
        if torch.isinf(tensor).any():
            print("  WARNING: Contains Infs!")

        # Print raw values
        print("\n  Raw Values:")
        with torch.no_grad():
            torch.set_printoptions(threshold=10_000, linewidth=200, precision=4, sci_mode=False)
            print(f"  {tensor}")

    # Comparison between prototypes
    if len(labels) >= 2:
        print("\n" + "=" * 50)
        print("INTER-CLASS COMPARISON")
        print("=" * 50)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                p1, p2 = proto_tensors[l1], proto_tensors[l2]

                # Distance
                dist = torch.norm(p1 - p2).item()

                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()

                print(f"{l1} vs {l2}:")
                print(f"  Euclidean Distance: {dist:.4f}")
                print(f"  Cosine Similarity:  {cos_sim:.4f}")

                # Relative distance (distance / mean norm)
                mean_norm = (torch.norm(p1).item() + torch.norm(p2).item()) / 2
                rel_dist = dist / mean_norm if mean_norm > 0 else 0
                print(f"  Relative Distance:  {rel_dist:.4f} (distance / avg_norm)")

    # Precision Matrix analysis if available
    precision_matrix = state.get("precision_matrix")
    if precision_matrix is not None:
        print("\n" + "=" * 50)
        print("PRECISION MATRIX ANALYSIS (Mahalanobis)")
        print("=" * 50)
        print(f"Shape: {list(precision_matrix.shape)}")
        print(f"Mean:  {precision_matrix.mean().item():.4f}")
        print(f"Std:   {precision_matrix.std().item():.4f}")
        print(f"Norm (Frobenius): {torch.norm(precision_matrix).item():.4f}")

        # Eigenvalues can show conditioning
        try:
            eigs = torch.linalg.eigvalsh(precision_matrix)
            print(f"Eigenvalues: min={eigs.min().item():.4e}, max={eigs.max().item():.4e}")
            cond_num = eigs.max().item() / eigs.min().item() if eigs.min().item() != 0 else float("inf")
            print(f"Condition Number: {cond_num:.4e}")
        except Exception as e:
            print(f"Could not calculate eigenvalues: {e}")

    print("\n" + "=" * 50)
    print("HYPOTHESIS TESTING FOR SINGLE-CLASS BIAS")
    print("=" * 50)
    if len(labels) == 2:
        # If one norm is significantly larger than the other,
        # it might affect classification if the input vectors are close to zero or shifted.
        l1, l2 = labels[0], labels[1]
        n1, n2 = torch.norm(proto_tensors[l1]).item(), torch.norm(proto_tensors[l2]).item()

        if abs(n1 - n2) / max(n1, n2) > 0.5:
            print(f"Potential Bias: The norms of the prototypes are very different ({n1:.2f} vs {n2:.2f}).")
            print("This could cause one class to dominate if input vectors have inconsistent scales.")

        # Check if one prototype is much closer to the origin
        if n1 < n2 * 0.1:
            print(f"Potential Bias: {l1} is much closer to the origin than {l2}.")
        elif n2 < n1 * 0.1:
            print(f"Potential Bias: {l2} is much closer to the origin than {l1}.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LPM model prototypes.")
    parser.add_argument("model_path", type=str, help="Path to the .pt model file")
    args = parser.parse_args()

    analyze_lpm_model(args.model_path)
