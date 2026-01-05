import os

import pandas as pd
from datasets import load_dataset


def prepare_gadzi_jezyk():
    print("Pobieranie zestawu danych Gadzi Język...")

    # 1. Załadowanie zestawu danych z Hugging Face
    dataset = load_dataset("JerzyPL/GadziJezyk", split="train")
    df = dataset.to_pandas()

    print(f"Załadowano {len(df)} rekordów.")

    # Wyświetlamy kolumny, aby potwierdzić obecność spacji
    print(f"Wykryte kolumny: {df.columns.tolist()}")

    # 2. Definicja mapowania kategorii (uwzględniamy spacje w kluczach)
    mapping = {
        " Kat 1": "verbal_abuse",
        " Kat 2": "vulgar",
        " Kat 3": "sexual",
        " Kat 4": "crime",
        " Kat 5": "self_harm",
    }

    def get_harm_categories_list(row):
        categories = []
        # Sprawdzamy wszystkie kategorie - zbieramy wszystkie, które mają wartość 1
        for col, label in mapping.items():
            try:
                if pd.notna(row[col]) and int(row[col]) == 1:
                    categories.append(label)
            except (ValueError, TypeError):
                continue

        # Jeśli brak kategorii, zwróć unknown, w przeciwnym razie połącz średnikiem
        # Średnik jest bezpieczniejszy w CSV niż przecinek
        return ";".join(categories) if categories else "unknown"

    print("Przetwarzanie wielu kategorii i transformacja formatu...")

    # 3. Tworzenie nowego DataFrame
    # Używamy nazw kolumn ze spacjami zgodnie z Twoim odkryciem (" Zapytanie")
    new_df = pd.DataFrame()
    new_df["text"] = df[" Zapytanie"]
    new_df["text_harm_label"] = 1
    new_df["prompt_harm_category"] = df.apply(get_harm_categories_list, axis=1)

    # 4. Zapis do pliku CSV
    # Używamy quoting=csv.QUOTE_ALL lub po prostu domyślnego separatora,
    # ponieważ średnik wewnątrz kolumny nie zepsuje struktury opartej na przecinkach.
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gadzi_jezyk_processed.csv")
    new_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Sukces! Plik zapisany w: {output_path}")
    print("Podgląd rekordów z wieloma kategoriami:\n", new_df[new_df["prompt_harm_category"].str.contains(";")].head())


if __name__ == "__main__":
    prepare_gadzi_jezyk()
