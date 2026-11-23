"""Loading the six datasets into one common format.

Every loader returns rows with:
  sid    text ID
  rid    review ID (texts from the same review are resampled together in the bootstrap)
  text   the text itself
  split  "train" (discovery and mapping set) or "test"
  cat    gold category, or "NONE" for texts without a usable single label
Texts labeled NONE stay in the discovery set but are excluded from mapping and testing.
"""
import ast
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATA_DIR, DATASETS

# ThaiFin aspect columns (FinNLP Form 56-1 data).
THAIFIN_ASPECTS = ["Brand", "Dividend", "Economics", "Environment", "Financing", "Governance", "Investment", "Legal", "M&A",
                   "Others", "Political", "Product/Service", "Profit/Loss", "Rating", "Social&People", "Technology"]
# TrueVoice classes that do not name a topic.
TRUEVOICE_DROP = {"garbage", "contact", "officer"}
TRUEVOICE_TEST_SIZE = 2000
TRUEVOICE_SPLIT_SEED = 12345


def _semeval_english(domain):
    """SemEval-2016 Task 5 (English): category = entity of the sentence's first opinion."""
    rows = []
    for split in ("train", "test"):
        root = ET.parse(DATA_DIR / f"semeval16/{domain}_{split}_english.xml").getroot()
        for rv in root.iter("Review"):
            for s in rv.iter("sentence"):
                ops = [o.get("category").split("#")[0] for o in s.iter("Opinion")]
                rows.append(dict(sid=s.get("id"), rid=rv.get("rid"), text=s.find("text").text, split=split,
                                 cat=ops[0] if ops else "NONE"))
    return rows


def _coursera():
    """OATS Coursera: the official dev and test splits are merged into one test set (dev first).

    Category = entity of the first opinion; opinions labeled POLARITY (annotation artifacts) are ignored.
    """
    d = DATA_DIR / "oats_coursera"
    review_of = {}
    for rv in ET.parse(d / "Coursera.xml").getroot().iter("Review"):
        for s in rv.iter("sentence"):
            review_of.setdefault(s.find("text").text.strip(), rv.get("rid"))
    rows = []
    for part, split in (("train", "train"), ("dev", "test"), ("test", "test")):
        with open(d / f"{part}.txt", encoding="utf8") as fh:
            for i, line in enumerate(fh):
                text, quads = line.rstrip("\n").split("####")
                ops = [q[1] for q in ast.literal_eval(quads) if not q[1].startswith("polarity")]
                cat = ops[0].upper().replace(" ", "#", 1).split("#")[0] if ops else "NONE"
                rows.append(dict(sid=f"{part}_{i}", rid=review_of[text.strip()], text=text, split=split, cat=cat))
    return rows


def _arhotels():
    """SemEval-2016 Arabic hotels: a sentence is kept only if all its opinions concern the same entity."""
    rows = []
    for split in ("train", "test"):
        root = ET.parse(DATA_DIR / f"semeval16_arabic/hotels_{split}_arabic.xml").getroot()
        for rv in root.iter("Review"):
            for s in rv.iter("sentence"):
                ents = {o.get("category").split("#")[0] for o in s.iter("Opinion")}
                rows.append(dict(sid=s.get("id"), rid=rv.get("rid"), text=s.find("text").text, split=split,
                                 cat=ents.pop() if len(ents) == 1 else "NONE"))
    return rows


def _thaifin():
    """ThaiFin: paragraphs with exactly one aspect; official train and test splits (dev is not used)."""
    rows = []
    for split in ("train", "test"):
        d = pd.read_csv(DATA_DIR / f"finnlp_sentiment/{split}.csv", index_col=0)
        single = d[THAIFIN_ASPECTS].sum(axis=1) == 1
        for i, (text, one, cat) in enumerate(zip(d.paragraph, single, d[THAIFIN_ASPECTS].idxmax(axis=1))):
            rows.append(dict(sid=f"{split}_{i}", rid=f"{split}_{i}", text=text, split=split, cat=cat if one else "NONE"))
    return rows


def truevoice_frame():
    """TrueVoice utterances after cleaning, and the fixed random order that defines the test set."""
    d = pd.read_parquet(DATA_DIR / "truevoice_intent/train.parquet")
    d = d[~d.Object.isin(TRUEVOICE_DROP)]
    d = d.assign(texts=d.texts.str.replace("<PHONE_NUMBER_REMOVED>", "", regex=False).str.strip()).reset_index(drop=True)
    order = np.random.default_rng(TRUEVOICE_SPLIT_SEED).permutation(len(d))
    return d, order


def _truevoice():
    """TrueVoice: category = object of the request; a random 2,000 utterances form the test set."""
    d, order = truevoice_frame()
    test, rest = order[:TRUEVOICE_TEST_SIZE], order[TRUEVOICE_TEST_SIZE:]
    return ([dict(sid=f"t{i}", rid=f"t{i}", text=d.texts[i], split="test", cat=d.Object[i]) for i in test] +
            [dict(sid=f"t{i}", rid=f"t{i}", text=d.texts[i], split="train", cat=d.Object[i]) for i in rest])


LOADERS = {"restaurants": lambda: _semeval_english("restaurants"), "laptops": lambda: _semeval_english("laptops"),
           "coursera": _coursera, "arhotels": _arhotels, "thaifin": _thaifin, "truevoice": _truevoice}


@dataclass
class Corpus:
    """A dataset split into the parts the framework uses."""
    name: str
    df: pd.DataFrame       # all texts (columns sid, rid, text, split, cat)
    disc: np.ndarray       # boolean mask: unlabeled discovery set
    dev: np.ndarray        # row indices of the labeled mapping set
    test: np.ndarray       # row indices of the test set
    cats: list             # sorted category names

    @property
    def spec(self):
        return DATASETS[self.name]

    def test_view(self):
        """Test texts, gold labels, and bootstrap groups, as used for LLM scoring.

        Single-text datasets (ThaiFin, TrueVoice) use the integer text ID as the group, which fixes
        the order of groups in the bootstrap and so reproduces the reported confidence intervals.
        """
        t = self.df.iloc[self.test]
        groups = t.rid.values
        if self.name in ("thaifin", "truevoice"):
            groups = np.array([int("".join(ch for ch in r if ch.isdigit())) for r in groups])
        return t.text.values, t.cat.values, groups


def load_corpus(name):
    """Load one dataset, drop empty texts, and define the discovery, mapping, and test sets."""
    df = pd.DataFrame(LOADERS[name]())
    df = df[df.text.notna() & (df.text.str.strip() != "")].reset_index(drop=True)
    disc = (df.split == "train").values
    labeled = (df.cat != "NONE").values
    return Corpus(name=name, df=df, disc=disc, dev=df.index[disc & labeled].values, test=df.index[~disc & labeled].values,
                  cats=sorted(set(df.cat) - {"NONE"}))


def load_categories(name):
    """Category inventory: {NAME: {"definition": str, "seeds": [str, str, str]}} in prompt order."""
    return json.loads(DATASETS[name].category_file.read_text(encoding="utf8"))
