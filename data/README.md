# Datasets

The datasets are not included in this repository. Download them from their sources and place them as shown below.
Use a different location by setting `D2D_DATA=/path/to/data`.

```
data/
├── semeval16/
│   ├── restaurants_train_english.xml
│   ├── restaurants_test_english.xml
│   ├── laptops_train_english.xml
│   └── laptops_test_english.xml
├── semeval16_arabic/
│   ├── hotels_train_arabic.xml
│   └── hotels_test_arabic.xml
├── oats_coursera/
│   ├── Coursera.xml          # full reviews (used to recover review IDs)
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
├── finnlp_sentiment/
│   ├── train.csv
│   ├── dev.csv               # not used
│   └── test.csv
└── truevoice_intent/
    └── train.parquet
```

| Folder | Dataset | Source | Terms |
|---|---|---|---|
| `semeval16/` | SemEval-2016 Task 5, subtask 1, English restaurants and laptops (Pontiki et al., 2016) | SemEval-2016 ABSA release; we used the XML files mirrored in the `YaxinCui/ABSADataset` GitHub repository | Terms of the SemEval organizers |
| `semeval16_arabic/` | SemEval-2016 Task 5, Arabic hotels | Same mirror (`YaxinCui/ABSADataset`) | Terms of the SemEval organizers |
| `oats_coursera/` | OATS, Coursera domain (Chebolu et al., LREC-COLING 2024) | The authors' OATS release | Terms of the OATS authors |
| `finnlp_sentiment/` | ThaiFin: paragraphs from Thai Form 56-1 annual reports, 16 aspects | `nlp-chula/finnlp-sentiment` (FinNLP project, supported by the Capital Market Development Fund, Thailand) | CC BY 4.0 |
| `truevoice_intent/` | TrueVoice call-center utterances with intent labels | <https://github.com/PyThaiNLP/truevoice-intent> | Provided courtesy of TrueVoice, **for educational purposes only** |

How each dataset is turned into discovery, mapping, and test sets is defined in [`d2d/data.py`](../d2d/data.py):

- **SemEval (English):** category = entity of the sentence's first opinion; official train/test splits.
- **Arabic hotels:** a sentence is kept only if all its opinions concern the same entity.
- **OATS Coursera:** category = entity of the first opinion; opinions labeled `POLARITY` are ignored. The official dev and test splits are merged into one test set.
- **ThaiFin:** only paragraphs with exactly one aspect; official train/test splits.
- **TrueVoice:** category = `Object`. The classes `garbage`, `contact`, and `officer` are removed. A fixed random test set of 2,000 utterances is drawn with seed 12345.

Texts without a usable single label stay in the unlabeled discovery set but are excluded from mapping and testing.
