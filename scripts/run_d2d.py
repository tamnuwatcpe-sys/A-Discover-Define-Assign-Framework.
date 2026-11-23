"""End-to-end evaluation of the Assign-Validate-Revise cycle against alternatives with the same labels.

    python scripts/run_d2d.py                                    # all datasets, 3 draws of 300 labels
    python scripts/run_d2d.py --datasets laptops --device cuda:1
    python scripts/run_d2d.py --baselines-only                   # recompute the label-budget baselines only

For every dataset and draw, 300 texts are drawn from the labeled mapping set as the validation sample.
  D2D v0      Qwen3-8B with the initial definitions (test answers cached by scripts/run_llm.py)
  Validate    validation macro-F1 and per-category F1; categories with F1 < 0.6 are weak
  Revise      Qwen3-8B rewrites the definitions of weak categories from validation errors only
  D2D v1      test texts re-assigned with the revised definitions
  D2D final   v1 if validation macro-F1 improved, otherwise v0
Alternatives with the same 300 labels: k-means (e5, main k, 5 seeds) clusters mapped by majority vote,
a nearest-centroid classifier, and logistic regression. The test set is used only for reporting.
Writes results/d2d/<dataset>_draw<d>.json and results/d2d/results.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d.baselines import cluster_map, logistic_regression  # noqa: E402
from d2d.config import DATASETS, MAIN_MODEL, RESULTS_DIR, SEEDS, short_name  # noqa: E402
from d2d.data import load_categories, load_corpus  # noqa: E402
from d2d.metrics import macro_f1  # noqa: E402
from d2d.revise import apply_check, validate_and_revise  # noqa: E402


def same_label_baselines(corpus, E, val, y_val, y_test, labels):
    """k-means mapping, nearest centroid, and logistic regression given only the validation labels."""
    spec, TEST = corpus.spec, corpus.test
    km = []
    for s in SEEDS:
        lab = np.load(spec.out_dir / "labels" / f"C1_k{spec.k_main}_s{s}.npy")
        mp = cluster_map(lab[val], y_val)
        km.append(macro_f1(y_test, np.array([mp.get(c, "UNMAPPED") for c in lab[TEST]]), labels))
    cen = {c: E[val][y_val == c].mean(0) for c in sorted(set(y_val))}
    C = np.array(list(cen.values()))
    return dict(kmeans_map=float(np.mean(km)),
                centroid=macro_f1(y_test, np.array(list(cen))[np.argmax(E[TEST] @ C.T, 1)], labels),
                logreg=macro_f1(y_test, logistic_regression(E[val], y_val, E[TEST]), labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--labels", type=int, default=300, help="size of the validation sample")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-memory", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--baselines-only", action="store_true", help="reuse saved D2D results; recompute the alternatives")
    args = ap.parse_args()

    out = RESULTS_DIR / "d2d"
    out.mkdir(parents=True, exist_ok=True)
    if not args.baselines_only:
        from d2d.llm import load_model
        tok, model = load_model(MAIN_MODEL, args.device, args.max_memory)

    rows = []
    for name in args.datasets.split(","):
        corpus, cats, spec = load_corpus(name), load_categories(name), DATASETS[name]
        df, DEV = corpus.df, corpus.dev
        test_texts, y_test, _ = corpus.test_view()
        labels = sorted(set(y_test))
        E = np.load(spec.out_dir / "emb.npy")
        v0_test = np.load(RESULTS_DIR / "llm" / f"{short_name(MAIN_MODEL)}_{name}_def.npy")
        for draw in range(args.draws):
            f = out / f"{name}_draw{draw}.json"
            if f.exists() and not args.baselines_only:
                rows.append(json.loads(f.read_text()))
                continue
            if args.baselines_only and not f.exists():
                continue
            val = np.random.default_rng(1000 + draw).choice(DEV, min(args.labels, len(DEV)), replace=False)
            y_val, val_texts = df.cat.values[val], df.text.values[val]
            if args.baselines_only:
                r = json.loads(f.read_text())
            else:
                r, test_v1 = validate_and_revise(tok, model, spec, cats, val_texts, y_val, test_texts, y_test, v0_test,
                                                 args.batch_size)
                r = dict(dataset=name, draw=draw, B=len(val), **r)
                np.save(out / f"{name}_draw{draw}_test_v1.npy", test_v1)
            r.update(same_label_baselines(corpus, E, val, y_val, y_test, labels))
            r = apply_check(r)
            f.write_text(json.dumps(r, ensure_ascii=False, indent=1))
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "revised"}, flush=True)
    res = pd.DataFrame(rows).drop(columns=["revised", "weak"], errors="ignore")
    res.to_csv(out / "results.csv", index=False)
    print(res.groupby("dataset").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
