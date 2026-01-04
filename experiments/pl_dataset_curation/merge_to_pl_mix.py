import os

import pandas as pd

# Use the directory where this script is located as base
base_dir = os.path.dirname(os.path.abspath(__file__))
neutral_path = os.path.join(base_dir, "sampled_neutral_520_processed.csv")
gadzi_path = os.path.join(base_dir, "gadzi_jezyk_processed.csv")
out_path = os.path.join(base_dir, "pl_mix.csv")

# Load datasets
neutral_df = pd.read_csv(neutral_path)
gadzi_df = pd.read_csv(gadzi_path)


# Map labels
def map_label(label):
    if label == 0 or label == "0":
        return "unharmful"
    if label == 1 or label == "1":
        return "harmful"
    return label


# Process neutral dataset
neutral_df["text_harm_label"] = neutral_df["text_harm_label"].apply(map_label)
neutral_df["text_harm_category"] = "unharmful"


# Process gadzi jezyk dataset
gadzi_df["text_harm_label"] = gadzi_df["text_harm_label"].apply(map_label)
# Ensure the harm category column is named consistently
if "prompt_harm_category" in gadzi_df.columns:
    gadzi_df = gadzi_df.rename(columns={"prompt_harm_category": "text_harm_category"})

# Concatenate
df = pd.concat([neutral_df, gadzi_df], ignore_index=True)

# Shuffle the merged DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
cols = ["text", "text_harm_label", "text_harm_category"]
df[cols].to_csv(out_path, index=False)
print(f"Merged and shuffled dataset saved to {out_path}")
