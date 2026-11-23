"""Tables 3-5 of the paper and the summary statistics quoted in the text.

    python scripts/make_tables.py

Writes Markdown and CSV files to results/tables/:
  table3_methods.*      macro-F1 of discovery methods, LLMs, and supervised baselines
  table4_llms.*         macro-F1 by LLM and prompt condition, and the smallest gain over CAt
  table5_end_to_end.*   end-to-end evaluation with 300 labels (mean and SD over three draws)
  summary.txt           Kendall's tau of method rankings and the share of categories with F1 = 0
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d.config import DATASETS, LARGE_MODEL, LOCAL_MODELS, RESULTS_DIR, short_name  # noqa: E402
from d2d.discovery import METHODS, NAMES  # noqa: E402

OUT = RESULTS_DIR / "tables"
DS = list(DATASETS)
LOCAL = [short_name(m) for m in LOCAL_MODELS]
LARGE = short_name(LARGE_MODEL)


def f3(x):
    return f"{x:.3f}"


def save(df, stem):
    df.to_csv(OUT / f"{stem}.csv")
    (OUT / f"{stem}.md").write_text(df.to_markdown() + "\n", encoding="utf8")


def discovery_means():
    """Mean macro-F1 over seeds at the main k, per method and dataset."""
    out = {}
    for name in DS:
        spec = DATASETS[name]
        R = pd.read_csv(spec.out_dir / "runs.csv")
        out[name] = R[R.k == spec.k_main].groupby("method").macro.mean()
    return pd.DataFrame(out)


def table3(disc, llm):
    rows = {}
    for m in ["C1", "C3", "C3R", "A2", "E1top", "A1", "B1"]:
        rows[f"Unsupervised: {m} {NAMES[m]}"] = disc.loc[m].map(f3)
    base = {n: pd.read_csv(DATASETS[n].out_dir / "baselines.csv", index_col=0).iloc[:, 0] for n in DS}
    eq = {n: pd.read_csv(DATASETS[n].out_dir / "equal_labels_538.csv", index_col=0)["mean"] for n in DS}
    rows["Seeded: CAt"] = pd.Series({n: f3(base[n]["CAt"]) for n in DS})
    for m in ["D1", "A2S"]:
        rows[f"Seeded: {m} {NAMES[m]}"] = disc.loc[m].map(f3)
    rows["LLM: Qwen3-8B"] = pd.Series({n: f3(llm.loc[("Qwen3-8B", n), "macro"]) for n in DS})
    rows["LLM: range of 7 open-weight models"] = pd.Series({n: "–".join(f3(f(llm.loc[[(m, n) for m in LOCAL], "macro"]))
                                                                        for f in (min, max)) for n in DS})
    rows[f"LLM: {LARGE}"] = pd.Series({n: f3(llm.loc[(LARGE, n), "macro"]) for n in DS})
    rows["Supervised: logistic regression (e5)"] = pd.Series({n: f3(base[n]["SUP_e5_LR"]) for n in DS})
    rows["Supervised: nearest centroid (e5)"] = pd.Series({n: f3(base[n]["SUP_e5_centroid"]) for n in DS})
    rows["Supervised: nearest centroid, 538 labels"] = pd.Series({n: f3(eq[n]["SUP_e5_centroid"]) for n in DS})
    rows["Floor: stratified random"] = pd.Series({n: f3(base[n]["CHANCE_stratified"]) for n in DS})
    save(pd.DataFrame(rows).T[DS], "table3_methods")


def table4(llm_all):
    rows = {}
    q = llm_all[llm_all.model == "Qwen3-8B"].set_index(["cond", "dataset"]).macro
    for cond, label in (("names", "names only"), ("def", "names + definitions"), ("def_seeds", "definitions + seeds")):
        rows[f"Qwen3-8B, {label}"] = pd.Series({n: f3(q[(cond, n)]) for n in DS})
    d = llm_all[llm_all.cond == "def"].set_index(["model", "dataset"])
    for m in LOCAL[1:] + [LARGE]:
        rows[f"{m}, names + definitions"] = pd.Series({n: f3(d.loc[(m, n), "macro"]) for n in DS})
    rows["CAt (reference)"] = pd.Series({n: f3(d.loc[("Qwen3-8B", n), "cat"]) for n in DS})
    gain = {}
    for n in DS:  # the local model with the smallest gain over CAt, with its 95% bootstrap interval
        r = d.loc[[(m, n) for m in LOCAL]].sort_values("llm_minus_cat").iloc[0]
        gain[n] = f"{r.llm_minus_cat:+.3f} [{r.lo:.3f}, {r.hi:.3f}]"
    rows["Smallest gain over CAt (7 local models)"] = pd.Series(gain)
    save(pd.DataFrame(rows).T[DS], "table4_llms")


def table5():
    r = pd.read_csv(RESULTS_DIR / "d2d" / "results.csv")
    g = r.groupby("dataset")

    def ms(col):
        return {n: f"{g[col].mean()[n]:.3f} ({g[col].std()[n]:.3f})" for n in DS}
    t = pd.DataFrame({"D2D, initial definitions": {n: f3(g.test_v0.mean()[n]) for n in DS},
                      "Revised, no check": ms("test_v1"), "D2D with check": ms("test_final"),
                      "Revisions kept": {n: f"{int(g.accepted.sum()[n])} of {g.size()[n]}" for n in DS},
                      "k-means + mapping": ms("kmeans_map"), "Nearest centroid": ms("centroid"),
                      "Logistic regression": ms("logreg")}).loc[DS]
    save(t, "table5_end_to_end")


def summary(disc, llm):
    lines = ["Kendall's tau between the ranking of the nine discovery methods on the restaurants and on each dataset (main k):"]
    for n in DS[1:]:
        tau, p = kendalltau(disc.loc[METHODS, "restaurants"], disc.loc[METHODS, n])
        lines.append(f"  {n:12s} tau = {tau:.2f} (p = {p:.3f})")
    zero = {m: [] for m in ("C1", "D1", "CAt", "LLM")}
    for n in DS:
        pc = pd.read_csv(DATASETS[n].out_dir / f"percat_k{DATASETS[n].k_main}.csv", index_col=0)
        f1 = json.loads(llm.loc[("Qwen3-8B", n), "percat"])
        for m in ("C1", "D1", "CAt"):
            zero[m] += list(pc[m] == 0)
        zero["LLM"] += [f1[c] == 0 for c in pc.index]
    lines.append(f"Categories with F1 = 0 (of {len(zero['LLM'])} test categories):")
    lines += [f"  {m:4s} {sum(v) / len(v):.0%}" for m, v in zero.items()]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    llm_all = pd.read_csv(RESULTS_DIR / "llm" / "results.csv")
    llm = llm_all[llm_all.cond == "def"].set_index(["model", "dataset"])
    disc = discovery_means()
    table3(disc, llm)
    table4(llm_all)
    if (RESULTS_DIR / "d2d" / "results.csv").exists():
        table5()
    summary(disc, llm)
    for f in sorted(OUT.glob("*.md")):
        print(f"\n## {f.stem}\n{f.read_text()}")


if __name__ == "__main__":
    main()
