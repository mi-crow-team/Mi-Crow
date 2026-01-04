"""Does not work"""

import os

from datasets import load_dataset


def download_and_prepare_polemo():
    print("Pobieranie zestawu danych Polemo2 (all_sentence)...")

    # 1. Pobranie podzbioru 'all_sentence' (pojedyncze zdania) i splitu 'train'
    dataset = load_dataset("clarin-pl/polemo2-official", "all_sentence", split="train", trust_remote_code=True)

    # 2. Konwersja do formatu Pandas DataFrame
    df = dataset.to_pandas()

    # W Polemo2 etykiety (target) oznaczają:
    # 0: neutralny (neutral)
    # 1: pozytywny (positive)
    # 2: negatywny (negative)
    # 3: ambiwalentny (ambiguous)

    print(f"Liczba wszystkich rekordów w treningowym zestawie: {len(df)}")

    # 3. Filtrowanie: wybieramy tylko teksty neutralne (target == 0)
    df_neutral = df[df["target"] == 0].copy()

    # 4. Transformacja kolumn:
    # Zmieniamy nazwę 'target' na 'text_harm_label'
    # (wszystkie będą miały wartość 0, co u Ciebie oznacza grupę neutralną/bezpieczną)
    df_neutral = df_neutral.rename(columns={"target": "text_harm_label"})

    # Wybieramy tylko interesujące nas kolumny
    df_neutral = df_neutral[["text", "text_harm_label"]]

    print(f"Liczba tekstów neutralnych po przefiltrowaniu: {len(df_neutral)}")

    # 5. Zapis do pliku CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "polemo_neutral.csv")
    df_neutral.to_csv(output_filename, index=False, encoding="utf-8")

    print(f"Plik '{output_filename}' został pomyślnie zapisany.")
    return output_filename


if __name__ == "__main__":
    download_and_prepare_polemo()
