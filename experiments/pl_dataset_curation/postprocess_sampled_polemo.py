import csv
import os
import re

INPUT_FILE = "sampled_neutral_260.csv"
OUTPUT_FILE = "sampled_neutral_260_processed.csv"

# Define quote characters to remove from start/end
QUOTE_CHARS = [
    '"',
    "'",
    "„",
    "“",
    "”",
    "«",
    "»",
    "‚",
    "‘",
    "’",
    "‟",
    "‹",
    "›",
    "″",
    "‶",
    "〝",
    "〞",
    "〟",
    "“",
    "”",
    "„",
    "‟",
]


def clean_text(text):
    """
    Clean text by:
    - Removing triple quotes
    - Removing leading/trailing quotes (including Polish „ and ")
    - Normalizing whitespace
    - Normalizing ellipsis
    - Fixing spacing around punctuation
    """
    if not isinstance(text, str):
        return text

    # First, strip any leading/trailing whitespace
    text = text.strip()

    # Remove triple quotes (single, double, and down quotes)
    text = re.sub(r'("""|\'\'\'|„„„|"""|"""|«««|»»»)', "", text)

    # Remove leading/trailing quote chars repeatedly until none remain
    # This pattern includes Polish opening quotes „ and closing quotes "
    pattern = r'^["\'„""«»‚' '‟‹›″‶〝〞〟\\s]+|["\'„""«»‚' r"‟‹›″‶〝〞〟\s]+$"
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(pattern, "", text)

    # Normalize ellipsis: replace … with ... and multiple dots with exactly ...
    text = re.sub(r"…", "...", text)
    text = re.sub(r"\.{4,}", "...", text)

    # Replace multiple consecutive spaces (including various types) with single space
    text = re.sub(r"\s+", " ", text)

    # Replace exactly 2 dots with 1 dot (but preserve exactly 3 dots)
    text = re.sub(r"\.{2}(?!\.)", ".", text)

    # Fix common spacing issues around punctuation
    # Remove spaces before commas, periods, exclamation, question marks
    text = re.sub(r"\s+([,\.!?;:])", r"\1", text)

    # Ensure single space after punctuation when followed by a letter
    text = re.sub(r"([,;:])(?=[^\s])", r"\1 ", text)
    text = re.sub(r"([\.!?])(?=[A-ZĄĆĘŁŃÓŚŹŻ])", r"\1 ", text)

    # Final strip to remove any edge whitespace
    text = text.strip()

    return text


def process_csv(input_path, output_path):
    """
    Process the CSV file by cleaning the text column.
    """
    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = list(reader)

    processed_rows = []
    for row in rows:
        # Only clean the 'text' column, keep label as is
        if "text" in row:
            row["text"] = clean_text(row["text"])
        processed_rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Processed {len(processed_rows)} rows")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, INPUT_FILE)
    output_path = os.path.join(base_dir, OUTPUT_FILE)

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        exit(1)

    process_csv(input_path, output_path)
    print(f"Processed CSV saved to {output_path}")
