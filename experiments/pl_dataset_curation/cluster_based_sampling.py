import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def intelligent_sampling(df, k_target, text_col="text"):
    """
    Samples K rows using Hierarchical K-means with Resampling.
    Based on: arXiv:2405.15613v2 (Meta FAIR)
    """
    print(f"Embedding {len(df)} sentences...")
    # Using a RoBERTa model suitable for Polish
    model = SentenceTransformer("sdadas/st-polish-paraphrase-from-mpnet")
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)

    # Parameters for the hierarchy
    # For a target of 520, we use a 2-level hierarchy:
    # Level 1 (Leaf): ~200 clusters to find fine-grained concepts
    # Level 2 (Root): ~50 clusters to group those concepts broadly
    n_leaf_clusters = min(len(df) // 5, 200)
    n_root_clusters = min(n_leaf_clusters // 2, 50)

    print(f"Level 1: Clustering into {n_leaf_clusters} leaf clusters...")
    kmeans_l1 = KMeans(n_clusters=n_leaf_clusters, random_state=42, n_init="auto").fit(embeddings)
    df["leaf_cluster"] = kmeans_l1.labels_
    leaf_centroids = kmeans_l1.cluster_centers_

    print(f"Level 2: Clustering {n_leaf_clusters} centroids into {n_root_clusters} root clusters...")
    kmeans_l2 = KMeans(n_clusters=n_root_clusters, random_state=42, n_init="auto").fit(leaf_centroids)

    # Map leaf clusters to root clusters
    leaf_to_root = dict(enumerate(kmeans_l2.labels_))
    df["root_cluster"] = df["leaf_cluster"].map(leaf_to_root)

    # Hierarchical Balanced Sampling Logic
    # 1. Distribute k_target evenly among root clusters
    samples_per_root = k_target // n_root_clusters
    remainder = k_target % n_root_clusters

    sampled_indices = []

    # Process each root cluster
    for root_id in range(n_root_clusters):
        root_quota = samples_per_root + (1 if root_id < remainder else 0)
        root_df = df[df["root_cluster"] == root_id]

        # 2. Within each root, distribute quota among its leaf clusters
        child_leafs = root_df["leaf_cluster"].unique()
        samples_per_leaf = root_quota // len(child_leafs)
        leaf_remainder = root_quota % len(child_leafs)

        for i, leaf_id in enumerate(child_leafs):
            leaf_quota = samples_per_leaf + (1 if i < leaf_remainder else 0)
            leaf_df = root_df[root_df["leaf_cluster"] == leaf_id]

            # 3. Sample 'r' (random) or 'c' (closest to centroid) from leaf
            # We'll use random sampling within the cluster to ensure diversity
            n_to_take = min(len(leaf_df), leaf_quota)
            sampled_indices.extend(leaf_df.sample(n=n_to_take, random_state=42).index)

    # If we are still short (due to small clusters), fill with random remaining
    if len(sampled_indices) < k_target:
        needed = k_target - len(sampled_indices)
        remaining_df = df.drop(sampled_indices)
        sampled_indices.extend(remaining_df.sample(n=needed, random_state=42).index)

    return df.loc[sampled_indices[:k_target]].copy()


if __name__ == "__main__":
    # --- POLEMO ---

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "polemo_neutral.csv")
    df_neutral = pd.read_csv(input_csv)

    # Filter for neutral label (as per your instruction)
    df_neutral = df_neutral[df_neutral["text_harm_label"] == 0]

    # Perform intelligent sampling
    k_target = 260
    curated_df = intelligent_sampling(df_neutral, k_target)

    # Clean up temporary columns and save
    curated_df = curated_df[["text", "text_harm_label"]]
    save_path = os.path.join(script_dir, "sampled_neutral_260.csv")
    curated_df.to_csv(save_path, index=False)
    print(f"Success! Saved {len(curated_df)} rows to '{save_path}'.")

    # --- AEGIS ---
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # k_target = 260
    # input_csv_aegis = os.path.join(script_dir, "aegis_safe_train.csv")
    # df_aegis = pd.read_csv(input_csv_aegis)
    # curated_aegis_df = intelligent_sampling(df_aegis, k_target, text_col="prompt")
    # curated_aegis_df = curated_aegis_df[["prompt", "prompt_label"]]
    # save_path_aegis = os.path.join(script_dir, "sampled_aegis_safe_260.csv")
    # curated_aegis_df.to_csv(save_path_aegis, index=False)
    # print(f"Success! Saved {len(curated_aegis_df)} rows to '{save_path_aegis}'.")
