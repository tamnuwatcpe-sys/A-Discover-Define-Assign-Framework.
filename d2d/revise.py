"""Validate and Revise stages.

A small labeled validation sample shows which categories are weak (F1 below a threshold). For each weak
category, the LLM (standing in for the analyst) rewrites the definition from the validation texts it got
wrong. A revision is kept only if validation macro-F1 improves; the test set is never used here.
"""
import copy

import numpy as np
from sklearn.metrics import f1_score

from .llm import chat, classify, generate
from .metrics import macro_f1

WEAK_F1 = 0.6       # categories below this validation F1 are revised
N_EXAMPLES = 6      # misclassified examples shown per direction
MAX_CHARS = 300     # long texts (ThaiFin paragraphs) are truncated in the revision prompt


def weak_categories(categories, y_val, pred_val, labels, threshold=WEAK_F1):
    """Categories that occur in the validation sample (as gold or prediction) and have F1 < threshold."""
    per_cat = dict(zip(labels, f1_score(y_val, pred_val, labels=labels, average=None, zero_division=0)))
    return [c for c in categories if c in per_cat and ((y_val == c).any() or (pred_val == c).any()) and per_cat[c] < threshold]


def revision_prompt(spec, categories, cat, missed, wrong):
    """Ask for a new definition of `cat` given texts it missed and texts wrongly assigned to it."""
    lines = "\n".join(f"- {k}: {v['definition']}" for k, v in categories.items())
    ex1 = "\n".join(f'{i + 1}. "{t[:MAX_CHARS]}" (assigned to {p})' for i, (t, p) in enumerate(missed)) or "(none)"
    ex2 = "\n".join(f'{i + 1}. "{t[:MAX_CHARS]}" (actually {g})' for i, (t, g) in enumerate(wrong)) or "(none)"
    return (f"You maintain a category scheme for classifying {spec.unit}s from {spec.domain}. The texts are in {spec.language}.\n"
            f"Current categories:\n{lines}\n\n"
            f"A classifier using these definitions often gets the category {cat} wrong.\n"
            f"Texts that belong to {cat} but were assigned to another category:\n{ex1}\n\n"
            f"Texts that were assigned to {cat} but belong to another category:\n{ex2}\n\n"
            f"Rewrite the definition of {cat} in one or two sentences so that it can be told apart from the categories it is confused "
            f"with. Say what it includes and, if useful, what it excludes and where that belongs. Write in English, do not quote the "
            f"example texts, and answer with the new definition only.")


def revise_definitions(tok, model, spec, categories, weak, texts, y_val, pred_val):
    """Return a copy of the inventory in which every weak category has an LLM-rewritten definition."""
    new = copy.deepcopy(categories)
    prompts = []
    for c in weak:
        missed = [(t, p) for t, g, p in zip(texts, y_val, pred_val) if g == c and p != c][:N_EXAMPLES]
        wrong = [(t, g) for t, g, p in zip(texts, y_val, pred_val) if p == c and g != c][:N_EXAMPLES]
        prompts.append(chat(tok, model, revision_prompt(spec, categories, c, missed, wrong)))
    answers = generate(tok, model, prompts, max_new_tokens=160, batch_size=4)
    for c, a in zip(weak, answers):
        new[c]["definition"] = a.strip().split("\n")[0].strip()  # keep the first line only
    return new


def validate_and_revise(tok, model, spec, categories, val_texts, y_val, test_texts, y_test, test_pred_v0, batch_size=8):
    """One Assign-Validate-Revise round. Returns a dict with validation and test macro-F1 before and after."""
    labels = sorted(set(y_test))
    pred_v0, _ = classify(tok, model, spec, categories, val_texts, "def", batch_size)
    r = dict(test_v0=macro_f1(y_test, test_pred_v0, labels), val_v0=macro_f1(y_val, pred_v0, labels))
    weak = weak_categories(categories, y_val, pred_v0, labels)
    revised = revise_definitions(tok, model, spec, categories, weak, val_texts, y_val, pred_v0)
    r.update(weak=weak, revised={c: revised[c]["definition"] for c in weak})
    if weak:  # re-assign with the revised inventory
        pred_v1, _ = classify(tok, model, spec, revised, val_texts, "def", batch_size)
        test_v1, _ = classify(tok, model, spec, revised, test_texts, "def", batch_size)
    else:
        pred_v1, test_v1 = pred_v0, test_pred_v0
    r.update(val_v1=macro_f1(y_val, pred_v1, labels), test_v1=macro_f1(y_test, test_v1, labels))
    return r, test_v1


def apply_check(r):
    """Keep the revision only if validation macro-F1 went up."""
    r["accepted"] = bool(r["val_v1"] > r["val_v0"])
    r["test_final"] = r["test_v1"] if r["accepted"] else r["test_v0"]
    return r
