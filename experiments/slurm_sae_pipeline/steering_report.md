# Steering Report: Example from Interactive Naming Inference

This report documents a **representative example** from the SAE concept-steering pipeline (`09_interactive_naming_inference.py`): a Polish review sentence, the concepts detected, the chosen steerable concept, and the baseline vs. steered continuations. It also explains where **top texts** for these features are stored and how they are used.

---

## 1. Input sentence (prompt)

**Source:** Polemo-2 test split (review corpus).

**Sentence:**
```
Reklamują się jako miejsce noclegowe w górach - i to jest najlepsza recenzja - nie jest to na pewno hotel ani pensjonat , Po wycieczce pieszej lub dniu spędzonym na stoku wystarczy wypić 200 g destylatu i jakoś do rana się przetrzyma : ) Dla rodzin z dziećmi - odradzam .
```

**Prompt used for generation:** The same sentence, with trailing space (no extra punctuation).

Roughly: *"They advertise as mountain accommodation — and that's the best review — it's certainly not a hotel or guesthouse. After a hike or a day on the slopes, 200 g of spirits gets you through till morning : ) For families with children — I don't recommend."*

---

## 2. Detected concepts

The pipeline runs the Bielik 1.5B LM with an SAE on **layer 15**, extracts top‑k active features for the prompt, names them via an LLM using **top texts** (see Section 5), and validates that the proposed concepts fit the sentence. Below are the **11 concepts** kept for this example.

| Feature idx | Concept name           | Score |
|-------------|------------------------|-------|
| 537         | czasownik              | 0.85  |
| 4936        | obsługa klienta        | 0.80  |
| 990         | procent                | 0.95  |
| 638         | doświadczenie osobiste | 0.90  |
| **2866**    | **ocena**              | **0.95** |
| 3118        | dla                    | 0.95  |
| 5168        | cena                   | 0.95  |
| 178         | negacja                | 0.95  |
| 1555        | osoba                  | 0.85  |
| 5803        | czasownik              | 0.85  |
| 1038        | emotikon               | 0.90  |

*(“Czasownik” concepts are excluded from steering but kept for validation.)*

---

## 3. Most steerable concept

An LLM is asked which of the **non‑czasownik** concepts is most likely to change the model’s continuation when boosted. For this sentence, the chosen concept is:

- **`ocena`** (feature **2866**)

So we expect steering on **ocena** to have the strongest effect on the generated text (e.g. explicit ratings, recommendations, or evaluative language).

---

## 4. Steering manipulations

For each steerable concept, the pipeline:

1. Runs **baseline** generation (no steering).
2. Runs **steered** generation (concept direction added to activations at the relevant layer).
3. Optionally runs a **bias check**; `bias_check_passed` records that result.

All runs use the **same** prompt and sampling setup; only steering changes.

### 4.1 Baseline (no steering)

All concepts share the same baseline continuation (3 samples):

| Sample | Continuation |
|--------|--------------|
| 1 | `Wyświetl więcej opinii (1)\nNajnowsza opinia o: Hotel Active & Spa, Kudowa-Zdrój\nNa` |
| 2 | `!!\n- Adres: Bystra 16, Szczyrk\n- Tel. +48 (33) <phone>-` |
| 3 | `!!\nOdpowiedź od użytkownika Aga (Zarejestrowany) Data: 23-10-2018\nNie` |

The model often continues with review‑UI style text (more reviews, address, reply metadata) rather than fluid natural language.

### 4.2 Steered outputs (per concept)

#### **ocena** (2866) — most steerable

- **bias_check_passed:** yes  
- **Steered samples:**
  1. `Nie polecam. .\n\n .To samo dla mnie max po prostu wstyd jak reszta hotelu też nie było żadnych sprząc, łazieki`
  2. `bo tak samo jak ktoś ma problemy ze znalezieniem miejsca czy rezerwacji. Jesli nawet po otwarciu drzwi i okien są brudne . wszystko poza`
  3. `.To już było jaksowa zajeznica dla mnie bardzo niska jakość toaaynia notorycznie brudna oraz brak wody (`  

Steering toward **ocena** yields more **evaluative** content (“Nie polecam”, “niska jakość”, “notorycznie brudna”), consistent with the concept.

#### **obsługa klienta** (4936)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “siedzę sama w pokoju”, “jakość obsługi czy obsługa kuchni” — more **service‑related** language.

#### **procent** (990)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “Dojazd kosztuje ok 15:.40 zł”, “obsługa hotelowa była fatalna” — **numeric/cost** and **evaluative** fragments.

#### **doświadczenie osobiste** (638)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “Nie ma możliwości zrobienia rezerwacji”, “17 lat temu”, “dla dzieci które chcą kupić” — more **first‑person / experience** framing.

#### **dla** (3118)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “dla mnie za drogi”, “dla mnie koszmarowa obsługa” — **“dla”**‑centric, often “dla mnie”.

#### **cena** (5168)

- **bias_check_passed:** **no**  
- **Steered samples:** e.g. “kupić bilet”, “płatna woda”, “Nie polecam” — **price/cost** and recommendation; bias check failed for this concept.

#### **negacja** (178)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “nie ma wyboru”, “za drogi” — more **negation** and negative phrasing.

#### **osoba** (1555)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “siedzieć samemu”, “ktoś chce”, “ktoś był” — **person‑focused** language.

#### **emotikon** (1038)

- **bias_check_passed:** yes  
- **Steered samples:** e.g. “cholera”, “!!!”, “Ogólniai brak” — **emotional / emphatic** tone.

---

## 5. Top texts: what they are and where they are stored

### 5.1 Role of top texts

**Top texts** are the highest‑activation **contexts** (text + token + score) for each SAE feature, collected during **`03_run_inference`** over a large corpus. They are used to:

1. **Name** features (e.g. → “ocena”, “obsługa klienta”) via an LLM in `09_interactive_naming_inference`.
2. **Build** concept dictionaries in `07_create_dictionaries_from_texts` (e.g. `dictionaries/bielik-1.5b/layer_15/`).

So the same top texts that back the **names** in this example also back the dictionaries.

### 5.2 Storage and format

- **Produced by:** `03_run_inference` (with text tracking enabled).  
- **Typical location:** `store/runs/top_texts_collection_<YYYYMMDD_HHMMSS>/`  
  - Example: `store/runs/top_texts_collection_20260119_225330`  
  - Logs also reference Slurm paths such as  
    `.../experiments/slurm_sae_pipeline/store/runs/top_texts_collection_20260119_225346`.

- **Files:** `top_texts_layer_<layer_idx>_llamaforcausallm_model_layers_<L>_batch_<N>.json`  
  - For this example, **layer 15** → `top_texts_layer_0_llamaforcausallm_model_layers_15_batch_*.json` (exact batch depends on the run).

- **Schema:** Each JSON is a map `feature_idx (string) → list of records`:

```json
{
  "2866": [
    {
      "text": "Naprawdę polecam ten hotel",
      "score": 15.9,
      "token_idx": 0,
      "token_str": "Naprawdę"
    },
    {
      "text": "Byłem naprawdę zadowolony z usług",
      "score": 16.2,
      "token_idx": 1,
      "token_str": "naprawdę"
    }
  ],
  "537": [ ... ],
  ...
}
```

- **Fields:**
  - `text`: full span where the feature fired  
  - `score`: activation value  
  - `token_idx`: position of the highlighted token in the span  
  - `token_str`: token string  

A minimal **reference** structure (for features 0, 1, 2) is in:

`experiments/slurm_sae_pipeline/test_top_texts/top_texts_layer_0_llamaforcausallm_model_layers_15_batch_1.json`

The **same** structure applies to the feature indices used in this report; only the keys (and content) differ.

#### Example: top texts (same format, different features)

The file above uses the same schema as the top texts for this example’s features. Below is the **top‑5** list for **feature 1** (“wyrażenie_sprawdzające” / “Naprawdę”) as stored there — purely to illustrate the format:

| # | Score | Token     | Text |
|---|-------|-----------|------|
| 1 | 18.3  | Naprawdę  | **[Naprawdę]** piękny widok z okna |
| 2 | 17.1  | naprawdę  | To jest **[naprawdę]** dobre jedzenie |
| 3 | 16.8  | Naprawdę  | **[Naprawdę]** warto odwiedzić to miejsce |
| 4 | 16.2  | naprawdę  | Byłem **[naprawdę]** zadowolony z usług |
| 5 | 15.9  | Naprawdę  | **[Naprawdę]** polecam ten hotel |

Raw JSON excerpt:

```json
"1": [
  { "text": "Naprawdę piękny widok z okna", "score": 18.3, "token_idx": 0, "token_str": "Naprawdę" },
  { "text": "To jest naprawdę dobre jedzenie", "score": 17.1, "token_idx": 2, "token_str": "naprawdę" },
  { "text": "Naprawdę warto odwiedzić to miejsce", "score": 16.8, "token_idx": 0, "token_str": "Naprawdę" },
  { "text": "Byłem naprawdę zadowolony z usług", "score": 16.2, "token_idx": 1, "token_str": "naprawdę" },
  { "text": "Naprawdę polecam ten hotel", "score": 15.9, "token_idx": 0, "token_str": "Naprawdę" }
]
```

For this report’s **ocena** (2866), **obsługa klienta** (4936), etc., the top texts have the **same** structure and live under keys `"2866"`, `"4936"`, … in the layer‑15 `top_texts_*_batch_*.json` from the 03 run (see paths above).

### 5.3 Top texts for this example’s features

The concepts in **Section 2** correspond to these **feature indices**; their top texts live in the `top_texts_layer_0_llamaforcausallm_model_layers_15_batch_*.json` files from the 03 run that was used for this interactive naming run:

| Feature idx | Concept name           | Top texts key in JSON |
|-------------|------------------------|------------------------|
| 537         | czasownik              | `"537"`                |
| 4936        | obsługa klienta        | `"4936"`               |
| 990         | procent                | `"990"`                |
| 638         | doświadczenie osobiste | `"638"`                |
| **2866**    | **ocena**              | **`"2866"`**           |
| 3118        | dla                    | `"3118"`               |
| 5168        | cena                   | `"5168"`               |
| 178         | negacja                | `"178"`                |
| 1555        | osoba                  | `"1555"`               |
| 5803        | czasownik              | `"5803"`               |
| 1038        | emotikon               | `"1038"`               |

To **inspect** top texts for these features:

1. Use the **same** `--top_texts_dir` or `--top_texts_file` as in the `09` run (e.g. the `top_texts_collection_*` directory or the specific `top_texts_layer_0_..._batch_*.json` for layer 15).  
2. Open the layer‑15 top texts JSON and read the keys above.  
3. Each value is a list of `{ "text", "score", "token_idx", "token_str" }` objects, as in the schema.

`07_create_dictionaries_from_texts` can also be run with that `--top_texts_dir` to (re)build dictionaries and optionally `concepts_report.txt`, which includes **sample** top texts per feature.

### 5.4 Top texts used to name **ocena** (2866) and **wykrzyknik** (3432)

The two concepts highlighted in this report — **ocena** (Example 1, most steerable) and **wykrzyknik** (Example 2) — were **named by an LLM** given the top texts below. Those texts are the **basis** for the names: the LLM sees the same excerpts and proposes a short Polish concept label.

**Source:** `store/runs/top_texts_collection_20260119_225330/top_texts_layer_0_llamaforcausallm_model_layers_15_batch_480.json` (latest layer‑15 batch used by `09` when loading from that directory).

#### **ocena** (2866)

Top activating contexts (review/opinion snippets containing "jako", "oceniam", "jakość", etc.):

| # | Score | Text |
|---|-------|------|
| 1 | 38.0 | Nie wiem czy jest dobrym lekarzem , ale **jako** osoba mi po prostu nie odpowiada . |
| 2 | 35.7 | Jednak nie w konwencji dziennikarskiej , czy demaskatorskiej , ale raczej **jako** ukazanie luźnych migawek z życia Wielkiej Brytanii w XXI wieku . |
| 3 | 35.2 | : Analiza mat I , wykład i ćwiczenia : Polecam … polecam go **jako** ćwiczeniowca . |
| 4 | 34.7 | Byłem ostatnio ze znajomymi w Krakowie i **jako** bazę… |
| 5 | 33.9 | Na wielki plus zaliczam 4 letni serwis oraz **jakość** serwisu . |
| 6 | 33.8 | … prywatnie , nie tylko **jako** lekarza mojego i mojej rodziny . |
| 7 | 33.1 | … **oceniam** z perspektywy poczekalni , a nie jako pacjent . |
| 8 | 32.9 | Od 9 lat stosuję Afobam i **jakoś** żyję . |

The shared thread is **evaluation / "jako" / "jakość" / "oceniam"** — hence the name **ocena** (rating, evaluation).

#### **wykrzyknik** (3432)

Top activating contexts (emphatic fragments ending in "!" or "! ! !"):

| # | Score | Text |
|---|-------|------|
| 1 | 29.4 | … SPA czynne tylko do 21 . 00 , z którego pan wyprasza już o 20 . 45 . **! ! ! !** |
| 2 | 29.1 | **OSTRZEŻENIE ! ! ! !** |
| 3 | 29.0 | BRUNO pościel nie wymieniona **! ! !** |
| 4 | 28.9 | … w basenie " **! ?** |
| 5 | 28.9 | … animator Dogan **!** |
| 6 | 28.6 | Nikomu nie polecam . . . **! ! ! ! !** |
| 7 | 28.1 | … złe jedzenie i bar czynny do 24 . 00 a nie wolno wnosić jedzenia i napoi **! ! !** |
| 8 | 28.0 | Zostałem o tym poinformowany w dość nieuprzejmy sposób ( " Tu nie wolno robić zdjęć , jest tabliczka **!** " ) . |

The shared thread is **exclamation marks** and **emphatic, often negative** tone — hence the name **wykrzyknik** (exclamation).

---

## 6. Second example: „auto można zaparkować przy ulicy” — **wykrzyknik (3432)**

A second representative example from the same pipeline, with focus on the **wykrzyknik** (exclamation) concept (feature **3432**).

### 6.1 Input sentence

**Sentence:**
```
a tym czasem auto można zaparkować przy ulicy ! ! !
```

**Prompt used:** `a tym czasem auto można zaparkować przy ulicy ! !  ` (trailing space).

Roughly: *"And meanwhile you can park the car on the street ! ! !"* — a short, emphatic parking tip with multiple exclamation marks.

### 6.2 Detected concepts

| Feature idx | Concept name     | Score |
|-------------|------------------|-------|
| 537         | czasownik        | 0.85  |
| **3432**    | **wykrzyknik**   | **0.95** |
| 1588        | przy             | 0.90  |
| 1555        | osoba            | 0.85  |
| 4542        | rekomendacja     | 0.85  |
| 2295        | posiadanie       | 0.95  |
| 2351        | czasownik        | 0.85  |
| 5878        | czasownik-zwrotny| 0.95  |
| 984         | przed            | 0.95  |
| 2617        | lokalizacja      | 0.85  |
| 5518        | czasownik        | 0.90  |

**Most steerable concept (LLM-chosen):** `lokalizacja` (2617). The **wykrzyknik** (3432) manipulation is documented below as requested.

### 6.3 Baseline (no steering)

All concepts share the same baseline continuation (3 samples):

| Sample | Continuation |
|--------|--------------|
| 1 | `I teraz w zasadzie mamy dwie możliwości ; zrobić to co jest wskazane na rysunku (i nie uszkodzić auta), albo zmienić trasę i pojechać tak jak` |
| 2 | `„Nie ma nic gorszego niż kierowca z automatyczną skrzynią biegów, który nie potrafi wrzucić trzeciego biegu!" - to tylko jeden` |
| 3 | `Nasi kierowcy mają do dyspozycji kilka miejsc parkingowych na terenie obiektu. Można również skorzystać z miejsca postojowego znajdującego się za obiektem lub też zostawić` |

The model continues with **driving/parking** content (options, gearbox joke, parking availability).

### 6.4 **wykrzyknik** (3432) — steered output

- **bias_check_passed:** yes  
- **Steered samples:**
  1. `!!\n nie ma miejsca na parkingu bo kasa parkingowy do auta tylko parking pod bramkiem i już się trzeba było odworać że jak z`
  2. `nie chce dojs []] . Na parkin. po prostu i kasa na koniec parkingu dla niego jest ponad 15 złotych , co`
  3. `. A my od wczoraj przyjeźchaliśmy specjalnie na telefon i nie ma żadnych auta ani nawet scisku pod biurkiem , tylko woda`

Steering toward **wykrzyknik** keeps **parking/car** themes but shifts the tone: more **fragmented**, **emphatic** phrasing (“!!”, “po prostu”, “specjalnie”), complaints about parking (“nie ma miejsca”, “kasa … ponad 15 złotych”), and **exclamation‑like** rhythm. The bias check passed.

### 6.5 Other manipulations (brief)

- **lokalizacja** (2617): most steerable; bias check **failed**. Steered → “parking miejkowy”, “nie podjeżść … brudny i zaniedbany”, location-oriented but noisier.
- **przy** (1588): bias check **failed**. Steered → “przy”‑heavy, spatial; e.g. “14km od Krakowa”, “przy drodze”.
- **osoba** (1555): more person‑focused continuations (e.g. “kierowcy”, “my”).

---

## 7. Summary

- **Example 1 (mountain accommodation):** Negative Polish review; **ocena** (2866) most steerable; steering → evaluative sentiment (“Nie polecam”, “niska jakość”, “brudna”).  
- **Example 2 (parking):** “Auto można zaparkować przy ulicy ! ! !”; **lokalizacja** (2617) most steerable, **wykrzyknik** (3432) highlighted. Steering **wykrzyknik** → parking/car content with more emphatic, exclamatory tone (“!!”, “po prostu”, “specjalnie”), bias check passed.  
- **Top texts:** Stored in `store/runs/top_texts_collection_*/top_texts_layer_0_llamaforcausallm_model_layers_15_batch_*.json`; same schema for all features (e.g. 2866, 3432).

---

*Generated from `interactive_naming_results/interactive_naming_results.json` and the SAE pipeline in `experiments/slurm_sae_pipeline/`.*
