"""Discover stage on one dataset: all clustering methods, CAt, supervised baselines, and the bootstrap.

    python scripts/run_discovery.py restaurants      # restaurants | laptops | coursera | arhotels | thaifin | truevoice

Writes to results/discovery/<dataset>/:
  emb.npy                   e5 embeddings of every text (cache)
  labels/*.npy              cluster IDs per method, k, and seed (cache)
  runs.csv                  test macro-F1 of every method, k, and seed (clusters mapped by majority vote)
  table_macro_by_k.csv      the same, averaged over seeds
  baselines.csv             stratified random, nearest centroid, logistic regression, and CAt
  cat_pred_test.npy         CAt predictions on the test set (used when scoring the LLMs)
  percat_k<K>.csv           per-category F1 at the main k
  equal_labels_538.csv      mapping and supervised baselines with 538 labels (five draws)
  bootstrap.csv             paired differences with 95% review-level bootstrap intervals
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d.baselines import centroids, logistic_regression, mapped_predictions, stratified_random  # noqa: E402
from d2d.cat import CAt  # noqa: E402
from d2d.config import DATASETS, SEEDS  # noqa: E402
from d2d.data import load_categories, load_corpus  # noqa: E402
from d2d.discovery import METHODS, Discovery  # noqa: E402
from d2d.embeddings import corpus_embeddings  # noqa: E402
from d2d.metrics import bootstrap_indices, macro_f1_index  # noqa: E402
from d2d.text import get_tokenizer  # noqa: E402

warnings.filterwarnings("ignore")
N_EQUAL_LABELS = 538  # label budget for the equal-label comparison


def main(name):
    spec = DATASETS[name]
    out, ks, k_main = spec.out_dir, list(spec.ks), spec.k_main
    label_dir = out / "labels"
    corpus = load_corpus(name)
    df, DEV, TEST, CATS = corpus.df, corpus.dev, corpus.test, corpus.cats
    print(f"{name}: {len(df)} texts | discovery {corpus.disc.sum()} | mapping {len(DEV)} | test {len(TEST)} | {len(CATS)} categories")

    # ---------- Discover: run every method ----------
    E = corpus_embeddings(corpus)
    seeds = {c: v["seeds"] for c, v in load_categories(name).items()}
    tokenize = get_tokenizer(spec.lang)
    disc = Discovery(corpus, E, tokenize, seeds, label_dir)
    disc.run_all(ks, SEEDS, spec.lang)

    # ---------- Map clusters and score on the test set ----------
    y_dev, y = df.cat.values[DEV], df.cat.values[TEST]
    enc = {c: i for i, c in enumerate(CATS)}
    K = len(CATS)                                   # index K = noise or unmapped
    yt = np.array([enc[c] for c in y])
    EV = np.array(sorted({enc[c] for c in y}))      # average over categories present in the test set
    mf1 = lambda t, p: macro_f1_index(t, p, K, EV)  # noqa: E731

    def predict(lab, idx=DEV, ys=y_dev):
        return mapped_predictions(lab, idx, ys, TEST, enc, K)

    P = {(m, k, s): predict(np.load(label_dir / f"{m}_k{k}_s{s}.npy")) for m in METHODS for k in ks for s in SEEDS}
    R = pd.DataFrame([dict(method=m, k=k, seed=s, macro=mf1(yt, p), coverage=(p < K).mean()) for (m, k, s), p in P.items()])
    R.to_csv(out / "runs.csv", index=False)
    R.pivot_table(index="method", columns="k", values="macro").round(3).sort_values(k_main, ascending=False).to_csv(out / "table_macro_by_k.csv")

    # ---------- Baselines (all labels of the mapping set) ----------
    # One generator serves both the random baseline and the bootstrap below, in this order.
    rng = np.random.default_rng(0)
    base = {"CHANCE_stratified": np.mean([mf1(yt, np.array([enc[c] for c in stratified_random(y_dev, len(y), rng)]))
                                          for _ in range(500)]),
            "SUP_e5_centroid": mf1(yt, (E[TEST] @ centroids(E[DEV], y_dev, CATS).T).argmax(1)),
            "SUP_e5_LR": mf1(yt, np.array([enc[c] for c in logistic_regression(E[DEV], y_dev, E[TEST])]))}

    # ---------- CAt ----------
    cat = CAt([l.split() for l in disc.df.lex], corpus.disc, seeds, tokenize)
    ydv = np.array([enc[c] for c in y_dev])
    gamma, cat_all, cv = cat.select_gamma(enc, K, EV, DEV, ydv, df.rid.values[DEV])
    cat_test = cat_all[TEST]
    for s in SEEDS:  # CAt is deterministic; it is stored under every seed so that it can enter the bootstrap
        P[("CAt", k_main, s)] = cat_test
    np.save(out / "cat_pred_test.npy", np.array([CATS[i] if i < K else "UNMAPPED" for i in cat_test]))
    base["CAt"] = mf1(yt, cat_test)
    print("CAt gamma", gamma, "cv", {g: round(v, 3) for g, v in cv.items()}, "test", round(base["CAt"], 3))
    pd.Series(base).round(4).to_csv(out / "baselines.csv")

    # ---------- Per-category F1 at the main k (for Figure 4) ----------
    def f1cat(pred):
        return f1_score(yt, pred, labels=EV, average=None, zero_division=0)
    pc = {m: np.mean([f1cat(P[(m, k_main, s)]) for s in SEEDS], 0) for m in ("C1", "A2", "C3", "D1")}
    pc["CAt"] = f1cat(cat_test)
    pc = pd.DataFrame(pc, index=[CATS[i] for i in EV])
    pc.insert(0, "share", pd.Series(y).value_counts(normalize=True).reindex(pc.index).values)
    pc.sort_values("share", ascending=False).round(4).to_csv(out / f"percat_k{k_main}.csv")

    # ---------- Equal label budget: 538 random mapping labels, five draws ----------
    eq = []
    for draw in range(5):
        sub = np.random.default_rng(100 + draw).choice(len(DEV), min(N_EQUAL_LABELS, len(DEV)), replace=False)
        dsub, ysub = DEV[sub], y_dev[sub]
        row = {m: np.mean([mf1(yt, predict(np.load(label_dir / f"{m}_k{k_main}_s{s}.npy"), dsub, ysub)) for s in SEEDS]) for m in METHODS}
        row["SUP_e5_centroid"] = mf1(yt, (E[TEST] @ centroids(E[dsub], ysub, CATS).T).argmax(1))
        row["SUP_e5_LR"] = mf1(yt, np.array([enc[c] for c in logistic_regression(E[dsub], ysub, E[TEST])]))
        row["CAt"], row["draw"] = base["CAt"], draw
        eq.append(row)
    pd.DataFrame(eq).drop(columns="draw").agg(["mean", "std"]).T.round(4).to_csv(out / f"equal_labels_{N_EQUAL_LABELS}.csv")

    # ---------- Review-level bootstrap of paired differences (discovery methods averaged over seeds) ----------
    BS = bootstrap_indices(df.rid.values[TEST], rng)

    def score(m, k, ii):
        return np.mean([mf1(yt[ii], P[(m, k, s)][ii]) for s in SEEDS])

    def diff(a, b, k):
        full = np.arange(len(yt))
        d = np.array([score(a, k, ii) - score(b, k, ii) for ii in BS])
        return dict(a=a, b=b, k=k, diff=round(score(a, k, full) - score(b, k, full), 4),
                    lo=round(np.percentile(d, 2.5), 4), hi=round(np.percentile(d, 97.5), 4))

    mean = R.groupby(["method", "k"]).macro.mean()
    best_word = max(["A1", "A2", "B1"], key=lambda m: mean[(m, k_main)])
    pairs = [("C1", best_word, k_main), ("D1", "C1", k_main), ("D1", "C1", ks[-1]), ("A2S", "A2", k_main),
             ("E1top", "C1", k_main), ("C3R", "C3", k_main), ("C1", "C3R", k_main), ("C3", "C1", k_main),
             ("CAt", "C1", k_main), ("CAt", "D1", k_main)]
    D = pd.DataFrame([diff(a, b, k) for a, b, k in pairs])
    D.to_csv(out / "bootstrap.csv", index=False)
    print(R[R.k == k_main].groupby("method").macro.mean().round(3).sort_values(ascending=False).to_string())
    print({k: round(v, 3) for k, v in base.items()})
    print(D.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        sys.exit(f"usage: python scripts/run_discovery.py [{'|'.join(DATASETS)}]")
    main(sys.argv[1])
