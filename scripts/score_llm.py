"""Score the cached LLM answers against the gold labels and compare each model with CAt.

    python scripts/score_llm.py

Needs results/llm/*.npy (scripts/run_llm.py) and results/discovery/<dataset>/cat_pred_test.npy
(scripts/run_discovery.py). Writes results/llm/results.csv with, per model, dataset, and condition:
macro-F1, the share of unparsed answers, CAt's macro-F1, the difference from CAt with a 95% bootstrap
interval, and per-category F1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d.config import DATASETS, RESULTS_DIR  # noqa: E402
from d2d.data import load_corpus  # noqa: E402
from d2d.llm import CONDITIONS, UNMAPPED  # noqa: E402
from d2d.metrics import bootstrap_indices, macro_f1  # noqa: E402


def main():
    out = RESULTS_DIR / "llm"
    tests = {}
    for name in DATASETS:
        _, y, groups = load_corpus(name).test_view()
        cat = np.load(DATASETS[name].out_dir / "cat_pred_test.npy")
        assert len(cat) == len(y), name
        tests[name] = (y, groups, cat)

    rows = []
    for short in sorted({p.name.split("_")[0] for p in out.glob("*_def.npy")}):
        rng = np.random.default_rng(0)  # same resamples for every model
        for name, (y, groups, cat) in tests.items():
            labels = sorted(set(y))
            BS = bootstrap_indices(groups, rng)
            for cond in CONDITIONS:
                p = out / f"{short}_{name}_{cond}.npy"
                if not p.exists():
                    continue
                pred = np.load(p)
                assert len(pred) == len(y), (short, name, cond)
                r = dict(model=short, dataset=name, cond=cond, macro=macro_f1(y, pred, labels),
                         unparsed=(pred == UNMAPPED).mean(), cat=macro_f1(y, cat, labels))
                d = np.array([macro_f1(y[ii], pred[ii], labels) - macro_f1(y[ii], cat[ii], labels) for ii in BS])
                r.update(llm_minus_cat=r["macro"] - r["cat"], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5))
                per_cat = f1_score(y, pred, labels=labels, average=None, zero_division=0)
                r["percat"] = json.dumps(dict(zip(labels, np.round(per_cat, 3))))
                rows.append(r)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "results.csv", index=False)
    print(df[df.cond == "def"].pivot_table(index="model", columns="dataset", values="macro", aggfunc="first").round(3).to_string())


if __name__ == "__main__":
    main()
