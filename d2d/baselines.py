"""Mapping clusters to categories, and the supervised and random baselines."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .embeddings import unit


def cluster_map(cluster_ids, labels):
    """Map every cluster to the most common category among its labeled texts (noise, -1, is ignored)."""
    d = pd.DataFrame(dict(c=cluster_ids, y=labels))
    d = d[d.c >= 0]
    return d.groupby("c").y.agg(lambda t: t.value_counts().index[0]).to_dict()


def mapped_predictions(lab, map_idx, map_y, test_idx, enc, unmapped):
    """Predict test categories (as indices via `enc`) from cluster IDs mapped on (map_idx, map_y)."""
    mp = cluster_map(lab[map_idx], map_y)
    return np.array([enc.get(mp.get(c, ""), unmapped) for c in lab[test_idx]])


def centroids(E, y, cats):
    """Unit-length mean embedding of every category (a tiny constant vector if a category has no texts)."""
    return unit(np.array([E[y == c].mean(0) if (y == c).any() else np.zeros(E.shape[1]) + 1e-9 for c in cats]))


def logistic_regression(E_train, y_train, E_test):
    """Logistic regression on e5 embeddings (C = 2, balanced class weights)."""
    return LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(E_train, y_train).predict(E_test)


def stratified_random(y_map, n, rng):
    """Random labels drawn with the category frequencies of the mapping set."""
    prev = pd.Series(y_map).value_counts(normalize=True)
    return rng.choice(prev.index, n, p=prev.values)
