# Dataset configuration mapping
DATASET_CONFIGS = {
    "wgmix_test": {
        "store_path": "store/datasets/wgmix_test",
        "text_field": "prompt",
        "category_field": "prompt_harm_label",
        "description": "WildGuardMix Test (English)",
    },
    "plmix_test": {
        "store_path": "store/datasets/plmix_test",
        "text_field": "text",
        "category_field": "text_harm_label",
        "description": "PL Mix Test (Polish)",
    },
}
