import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Use the directory where this script is located as base
base_dir = os.path.dirname(os.path.abspath(__file__))
neutral_path = os.path.join(base_dir, "sampled_neutral_260_processed.csv")
gadzi_path = os.path.join(base_dir, "gadzi_jezyk_processed.csv")
aegis_path = os.path.join(base_dir, "pl_sampled_aegis_safe_260.csv")
out_path = os.path.join(base_dir, "pl_mix.csv")


# Load datasets
neutral_df = pd.read_csv(neutral_path)
gadzi_df = pd.read_csv(gadzi_path)
aegis_df = pd.read_csv(aegis_path)


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


# Process aegis dataset: map columns and values
aegis_df = aegis_df.rename(columns={"prompt": "text", "prompt_label": "text_harm_label"})
aegis_df["text_harm_label"] = "unharmful"  # all are 'safe', map to 'unharmful'
aegis_df["text_harm_category"] = "unharmful"

# Concatenate all
df = pd.concat([neutral_df, gadzi_df, aegis_df], ignore_index=True)

# Shuffle the merged DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# Save merged dataset
cols = ["text", "text_harm_label", "text_harm_category"]
df[cols].to_csv(out_path, index=False)
print(f"Merged and shuffled dataset saved to {out_path}")

# Train/test split with stratification

# Use both harm label and category for stratification
# Create stratification column combining label and category
stratify_cols = df["text_harm_label"] + "_" + df["text_harm_category"].fillna("")

# Check minimum class size and handle rare classes
class_counts = stratify_cols.value_counts()
min_class_size = class_counts.min()

# Print the counts
print("Class counts for stratification:")
print(class_counts)

# Identify rare and common classes
rare_classes = class_counts[class_counts == 1].index
common_classes = class_counts[class_counts > 1].index

rare_mask = stratify_cols.isin(rare_classes)
common_mask = stratify_cols.isin(common_classes)

df_common = df[common_mask].copy()
df_rare = df[rare_mask].copy()

if len(df_common) > 0:
    stratify_by = stratify_cols[common_mask]
    train_common, test_common = train_test_split(df_common, test_size=0.2, random_state=42, stratify=stratify_by)
else:
    # If all classes are rare, just do a random split
    train_common, test_common = train_test_split(df, test_size=0.2, random_state=42)

# Add all rare rows to train set (to avoid test set with single samples)
train_df = pd.concat([train_common, df_rare], ignore_index=True)
test_df = test_common

train_path = os.path.join(base_dir, "pl_mix_train.csv")
test_path = os.path.join(base_dir, "pl_mix_test.csv")
train_df[cols].to_csv(train_path, index=False)
test_df[cols].to_csv(test_path, index=False)
print(f"Train and test splits saved to {train_path} and {test_path}")
