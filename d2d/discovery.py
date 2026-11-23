"""Discover stage: the nine clustering methods compared in the paper.

Unsupervised (no category knowledge):
  A1     k-means on TF-IDF vectors
  A2     NMF on TF-IDF vectors
  B1     LDA trained on whole reviews and applied to texts (topic probability < 0.3 -> noise)
  C1     k-means on e5 sentence embeddings
  C3     BERTopic (UMAP + HDBSCAN on e5), unseen texts take the majority topic of their 10 nearest neighbors
  C3R    C3, with noise points reassigned the same way
  E1top  evidence-accumulation consensus clustering over C1, A2, and B1
Seeded (three seed phrases per category):
  D1     k-means on e5 with one initial centroid per category at the mean embedding of its seeds
  A2S    NMF initialized with the seed terms

Every method fits on the discovery set and returns one cluster ID per text (-1 = noise).
"""
import copy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .embeddings import encode, unit

METHODS = ["A1", "A2", "B1", "C1", "C3", "C3R", "E1top", "D1", "A2S"]  # E1top must follow C1, A2, and B1
UNSUPERVISED = ["A1", "A2", "B1", "C1", "C3", "C3R", "E1top"]
SEEDED = ["D1", "A2S", "CAt"]
NAMES = {"A1": "k-means (TF-IDF)", "A2": "NMF (TF-IDF)", "B1": "LDA", "C1": "k-means (e5)", "C3": "BERTopic",
         "C3R": "BERTopic, noise reassigned", "E1top": "Consensus clustering", "D1": "Seeded k-means", "A2S": "Seeded NMF",
         "CAt": "Contrastive attention"}


def _split(s):
    return s.split()


class Discovery:
    """Holds the shared representations of one corpus and runs the discovery methods on it."""

    def __init__(self, corpus, emb, tokenize, seeds, label_dir):
        self.df, self.disc, self.E = corpus.df, corpus.disc, emb
        self.tokenize, self.seeds, self.label_dir = tokenize, seeds, label_dir
        self.seed_cats = list(seeds)
        self.lex = [" ".join(tokenize(t)) for t in self.df.text]  # tokenized text joined by spaces
        self.df = self.df.assign(lex=self.lex)
        lex = self.df.lex
        # TF-IDF of unigrams and bigrams, fitted on the discovery set.
        tv = TfidfVectorizer(tokenizer=_split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
        tv.fit(lex[self.disc])
        self.X, self.vocab = tv.transform(lex), tv.vocabulary_
        # Word counts for LDA, which is trained on whole reviews.
        cv = CountVectorizer(tokenizer=_split, lowercase=False, token_pattern=None, min_df=5)
        reviews = self.df[self.disc].groupby("rid").lex.apply(" ".join)
        cv.fit(reviews)
        self.R_counts, self.Xc = cv.transform(reviews), cv.transform(lex)
        self.nn = self._neighbors()
        self._seed_centroids = None
        self._bertopic = {}

    # ---------- helpers ----------
    def _neighbors(self, n=10):
        """The n nearest discovery texts of every text (excluding the text itself)."""
        S = self.E @ self.E[self.disc].T
        pos = np.full(len(self.df), -1)
        pos[self.disc] = np.arange(self.disc.sum())
        S[np.arange(len(self.df))[self.disc], pos[self.disc]] = -9
        return np.argsort(-S, axis=1)[:, :n]

    def _nmf_labels(self, nmf):
        W = nmf.transform(self.X)
        lab = W.argmax(1)
        lab[W.sum(1) == 0] = -1
        return lab

    # ---------- methods ----------
    def A1(self, k, s):
        lab = KMeans(k, n_init=3, random_state=s).fit(self.X[self.disc]).predict(self.X)
        lab[np.asarray(self.X.sum(1)).ravel() == 0] = -1  # texts with no known word are noise
        return lab

    def A2(self, k, s):
        return self._nmf_labels(NMF(k, init="nndsvda", random_state=s, max_iter=400).fit(self.X[self.disc]))

    def A2S(self, k, s):
        # One row of H per category, built from the words of its seed phrases.
        H0 = np.zeros((len(self.seed_cats), len(self.vocab)))
        for r, c in enumerate(self.seed_cats):
            for phrase in self.seeds[c]:
                for w in self.tokenize(phrase):
                    if w in self.vocab:
                        H0[r, self.vocab[w]] += 1
        H0 = H0 / np.maximum(H0.max(1, keepdims=True), 1e-9)
        rng, Xd = np.random.default_rng(s), self.X[self.disc]
        H0 = H0[:min(k, len(H0))]
        # Remaining components start at random discovery texts.
        extra = Xd[rng.choice(Xd.shape[0], k - len(H0), replace=False)].toarray() if k > len(H0) else np.empty((0, self.X.shape[1]))
        H = np.vstack([H0, extra]) + 1e-4
        W = np.maximum(np.asarray(Xd @ H.T), 0) + 1e-4
        return self._nmf_labels(NMF(k, init="custom", random_state=s, max_iter=400).fit(Xd, W=W, H=H))

    def B1(self, k, s):
        T = LatentDirichletAllocation(k, learning_method="batch", max_iter=30, random_state=s).fit(self.R_counts).transform(self.Xc)
        lab = T.argmax(1)
        lab[(T.max(1) < 0.3) | (np.asarray(self.Xc.sum(1)).ravel() == 0)] = -1
        return lab

    def C1(self, k, s):
        return KMeans(k, n_init=3, random_state=s).fit(self.E[self.disc]).predict(self.E)

    def D1(self, k, s):
        if self._seed_centroids is None:
            self._seed_centroids = unit(np.array([encode(self.seeds[c]).mean(0) for c in self.seed_cats]))
        C = self._seed_centroids[:min(k, len(self._seed_centroids))]
        rng = np.random.default_rng(s)
        extra = (self.E[self.disc][rng.choice(self.disc.sum(), k - len(C), replace=False)] if k > len(C)
                 else np.empty((0, self.E.shape[1])))
        return KMeans(k, init=np.vstack([C, extra]), n_init=1, random_state=s).fit(self.E[self.disc]).predict(self.E)

    def _fit_bertopic(self, s, lang):
        if s not in self._bertopic:
            from bertopic import BERTopic
            from bertopic.dimensionality import BaseDimensionalityReduction
            from hdbscan import HDBSCAN
            from umap import UMAP
            U = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=s).fit_transform(self.E[self.disc])
            docs = self.df.lex[self.disc].replace("", "-").tolist()
            # The English setting strips non-Latin characters, so Arabic and Thai use the multilingual setting.
            m = BERTopic(umap_model=BaseDimensionalityReduction(), language="english" if lang == "en" else "multilingual",
                         hdbscan_model=HDBSCAN(min_cluster_size=30, min_samples=5),
                         vectorizer_model=CountVectorizer(tokenizer=_split, lowercase=False, token_pattern=None),
                         calculate_probabilities=False, verbose=False).fit(docs, U)
            self._bertopic[s] = (m, docs)
        return self._bertopic[s]

    def C3(self, k, s, lang, reassign=False):
        m, docs = self._fit_bertopic(s, lang)
        m = copy.deepcopy(m)
        if len(set(m.topics_) - {-1}) > k:
            m.reduce_topics(docs, nr_topics=k + 1)
        lab = np.full(len(self.df), -1)
        lab[self.disc] = m.topics_
        T = lab[self.disc][self.nn]  # topics of each text's nearest discovery neighbors
        if reassign:  # C3R: every text, noise included, takes the majority non-noise topic of its neighbors
            return np.array([np.bincount(r[r >= 0]).argmax() if (r >= 0).any() else -1 for r in T])
        # UMAP cannot transform unseen texts, so they take the majority topic (noise included) of their neighbors.
        lab[~self.disc] = [np.bincount(r + 1).argmax() - 1 for r in T[~self.disc]]
        return lab

    def E1top(self, k, s):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        Ls = [np.load(self.label_dir / f"{m}_k{k}_s{s}.npy") for m in ("C1", "A2", "B1")]
        samp = np.where(self.disc)[0]
        co = np.zeros((len(samp), len(samp)))  # co-association: how often two texts share a cluster
        for L in Ls:
            l = L[samp]
            co += (l[:, None] == l[None, :]) & (l[:, None] >= 0)
        co /= len(Ls)
        np.fill_diagonal(co, 1)
        cl = fcluster(linkage(squareform(1 - co, checks=False), "average"), k, "maxclust") - 1
        # Assign every text to the consensus cluster that best matches its clusters in the three base methods.
        score = np.zeros((len(self.df), cl.max() + 1))
        for L in Ls:
            P = np.zeros((cl.max() + 1, L.max() + 2))
            np.add.at(P, (cl, L[samp] + 1), 1)
            P /= np.bincount(cl)[:, None]
            P[:, 0] = 0
            score += P[:, L + 1].T
        return score.argmax(1)

    def run(self, method, k, s, lang):
        if method in ("C3", "C3R"):
            return self.C3(k, s, lang, reassign=method == "C3R")
        return getattr(self, method)(k, s)

    def run_all(self, ks, seeds, lang):
        """Run every method for every k and seed, caching labels as {method}_k{k}_s{seed}.npy."""
        self.label_dir.mkdir(parents=True, exist_ok=True)
        for m in METHODS:
            for s in seeds:
                for k in ks:
                    p = self.label_dir / f"{m}_k{k}_s{s}.npy"
                    if not p.exists():
                        np.save(p, np.asarray(self.run(m, k, s, lang)).astype(np.int32))
