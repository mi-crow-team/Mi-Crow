import os

import pandas as pd
import requests


def download_and_parse_polemo_manual():
    # Base URL for the raw data files on Hugging Face
    base_url = "https://huggingface.co/datasets/clarin-pl/polemo2-official/resolve/main/data"

    # We want the 'all' domain and 'sentence' configuration for the 'train' split
    # This matches the 'all_sentence' configuration you were trying to load
    target_file_url = f"{base_url}/all.sentence.train.txt"

    print(f"Downloading raw data from: {target_file_url}")
    response = requests.get(target_file_url)

    if response.status_code != 200:
        print(f"Failed to download. Status code: {response.status_code}")
        return

    content = response.content.decode("utf-8")
    data = []

    # Parsing logic based on polemo2-official.py
    # The script splits by space, takes the last element as label, and rest as text
    for line in content.splitlines():
        if not line.strip():
            continue

        splitted_line = line.split(" ")
        text = " ".join(splitted_line[:-1])

        # Clean the target label according to the original script logic
        raw_target = splitted_line[-1].strip().replace("minus_m", "minus").replace("plus_m", "plus").split("_")[-1]

        # We only want 'zero' (neutral) labels
        # In the original labels: 0=zero, 1=minus, 2=plus, 3=amb
        if raw_target == "zero":
            data.append({"text": text, "text_harm_label": 0})

    df = pd.DataFrame(data)
    print(f"Successfully parsed {len(df)} neutral observations.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "polemo_neutral.csv")
    df.to_csv(output_filename, index=False, encoding="utf-8")
    print(f"Saved to {output_filename}")


if __name__ == "__main__":
    download_and_parse_polemo_manual()
