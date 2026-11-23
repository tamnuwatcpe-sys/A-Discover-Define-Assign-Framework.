"""CAt (Tulkens and van Cranenburgh, 2020): seeded aspect classification by attention over word vectors.

Adaptation used in the paper: word vectors are e5 embeddings of single words, the aspect candidates are the
200 most frequent words of the discovery set, and the label words are the words of each category's seed phrases.
The RBF kernel width gamma is chosen by two-fold cross-validation (split by review) on the mapping set.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .embeddings import encode, unit
from .metrics import macro_f1_index

GAMMAS = (2, 5, 10, 20)
N_CANDIDATES = 200


class CAt:
    def __init__(self, tokens, disc, seeds, tokenize):
        """tokens: list of token lists (one per text); seeds: {category: [seed phrases]}."""
        self.tokens, self.seed_cats = tokens, list(seeds)
        freq = Counter(w for t, d in zip(tokens, disc) if d for w in t)
        label_words = [[w for ph in seeds[c] for w in tokenize(ph)] for c in self.seed_cats]
        vocab = sorted({w for t in tokens for w in t} | {w for ws in label_words for w in ws})
        self.wv = dict(zip(vocab, encode(vocab)))
        self.candidates = np.array([self.wv[w] for w, _ in freq.most_common(N_CANDIDATES)])
        self.labels = unit(np.array([np.mean([self.wv[w] for w in ws if w in self.wv], 0) for ws in label_words]))

    def predict(self, gamma, enc, unknown):
        """Category index (via `enc`) for every text; texts with no known word get `unknown`."""
        out = np.full(len(self.tokens), unknown)
        for i, t in enumerate(self.tokens):
            W = np.array([self.wv[w] for w in t if w in self.wv])
            if not len(W):
                continue
            # ||x - y||^2 = 2 - 2 cos(x, y) for unit vectors
            sc = np.exp(-gamma * np.maximum(2 - 2 * W @ self.candidates.T, 0)).sum(1)
            att = np.exp(sc - sc.max())
            att /= att.sum()
            out[i] = enc[self.seed_cats[(self.labels @ (att @ W)).argmax()]]
        return out

    def select_gamma(self, enc, n_cats, eval_cats, dev, y_dev, dev_groups, gammas=GAMMAS):
        """Pick gamma by two-fold cross-validation on the mapping set; returns (gamma, predictions, cv scores)."""
        half = np.isin(dev_groups, pd.unique(dev_groups)[::2])
        preds = {g: self.predict(g, enc, n_cats) for g in gammas}
        cv = {g: np.mean([macro_f1_index(y_dev[h], p[dev][h], n_cats, eval_cats) for h in (half, ~half)])
              for g, p in preds.items()}
        best = max(cv, key=cv.get)
        return best, preds[best], cv
