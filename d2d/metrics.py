"""Macro-F1 and the cluster (review-level) bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def macro_f1(y, pred, labels):
    """Macro-F1 over `labels`; predictions outside `labels` (e.g. UNMAPPED) count as errors."""
    return f1_score(y, pred, labels=labels, average="macro", zero_division=0)


def macro_f1_index(y, pred, n_cats, eval_cats):
    """Fast macro-F1 on integer labels.

    Class `n_cats` means noise or unmapped (always an error); the mean is taken over `eval_cats`,
    the categories present in the test set. Gives the same value as `macro_f1`.
    """
    cm = np.bincount(y * (n_cats + 1) + pred, minlength=(n_cats + 1) ** 2).reshape(n_cats + 1, n_cats + 1)
    tp = np.diag(cm)[:n_cats]
    denom = cm.sum(0)[:n_cats] + cm.sum(1)[:n_cats]
    return np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0)[eval_cats])


def bootstrap_indices(groups, rng, n_boot=1000):
    """Resample whole groups (e.g. reviews) with replacement; returns a list of index arrays."""
    by = pd.Series(np.arange(len(groups))).groupby(groups).indices
    keys = np.array(list(by))
    return [np.concatenate([by[k] for k in rng.choice(keys, len(keys))]) for _ in range(n_boot)]


def percentile_ci(values, level=95):
    """Two-sided percentile interval."""
    a = (100 - level) / 2
    return np.percentile(values, a), np.percentile(values, 100 - a)
