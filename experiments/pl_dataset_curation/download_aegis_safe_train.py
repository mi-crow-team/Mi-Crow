import csv
import os

from datasets import load_dataset

# Use the directory where this script is located as base
base_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(base_dir, "aegis_safe_train.csv")

# Load the train split of the dataset
# No config needed for this dataset, just split="train"
ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")

# Convert to pandas DataFrame
train_df = ds.to_pandas()

# Keep only required columns
cols = ["prompt", "prompt_label", "violated_categories"]
filtered_df = train_df[cols]

# Filter where prompt_label == "safe"
filtered_df = filtered_df[filtered_df["prompt_label"] == "safe"]

# Filter out very long prompts (>500 chars), very short prompts (<10 chars), and '[REDACTED]'
filtered_df = filtered_df[filtered_df["prompt"].str.len() <= 4000]
filtered_df = filtered_df[filtered_df["prompt"].str.len() >= 25]
filtered_df = filtered_df[filtered_df["prompt"] != "[REDACTED]"]

# Remove rows with non-empty violated_categories
filtered_df = filtered_df[filtered_df["violated_categories"].isna() | (filtered_df["violated_categories"] == "")]

# Drop violated_categories column
filtered_df = filtered_df.drop(columns=["violated_categories"])

# Save to CSV with quoting to handle newlines safely
filtered_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
print(f"Saved filtered train split to {out_path}")
