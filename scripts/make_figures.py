"""Figures 2-4 of the paper (Figure 1, the framework diagram, is docs/framework.png).

    python scripts/make_figures.py

Reads results/discovery/*/ and results/llm/results.csv; writes PNG files (300 dpi) and the plotted values
as CSV to results/figures/:
  fig2_overview.png       macro-F1 of every method on the six datasets
  fig3_seeds.png          effect of the same seed phrases on k-means (e5) and NMF (TF-IDF)
  fig4_per_category.png   per-category F1 against category share of the test set
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# The paper uses Liberation Serif (metric-compatible with Times New Roman); register local copies if present.
for _f in Path.home().glob(".fonts/LiberationSerif-*.ttf"):
    font_manager.fontManager.addfont(str(_f))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d.config import DATASETS, LOCAL_MODELS, RESULTS_DIR, TITLES, short_name  # noqa: E402
from d2d.discovery import NAMES  # noqa: E402

FIG = RESULTS_DIR / "figures"
COLOR = {"unsup": "#2a78d6", "seeded": "#eb6834", "llm": "#1baf7a"}
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#d6336c"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8.5, "axes.edgecolor": AXIS, "text.color": "black", "axes.labelcolor": "black",
                     "axes.titlecolor": "black", "xtick.labelcolor": "black", "ytick.labelcolor": "black",
                     "legend.labelcolor": "black", "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
LABEL = dict(NAMES, LLM="Qwen3-8B (zero-shot)")
ORDER = ["LLM", "CAt", "D1", "A2S", "C1", "C3R", "C3", "A2", "E1top", "A1", "B1"]  # same order in every panel
SEEDED = {"CAt", "D1", "A2S"}
LOCAL = [short_name(m) for m in LOCAL_MODELS]


def track(m):
    return "llm" if m == "LLM" else "seeded" if m in SEEDED else "unsup"


def load_dataset(name, llm):
    spec = DATASETS[name]
    b = pd.read_csv(spec.out_dir / "baselines.csv", index_col=0).iloc[:, 0]
    return dict(key=name, title=TITLES[name], kmain=spec.k_main, runs=pd.read_csv(spec.out_dir / "runs.csv"),
                cat=b["CAt"], ceil=b["SUP_e5_centroid"], rnd=b["CHANCE_stratified"],
                percat=spec.out_dir / f"percat_k{spec.k_main}.csv")


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def legend(fig, handles, labels, y=-0.04, ncol=None, **kw):
    fig.legend(handles, labels, loc="lower center", ncol=ncol or len(labels), frameon=False, bbox_to_anchor=(0.5, y), **kw)


def fig_overview(DS, llm):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.6), sharey=True)
    rows, y = [], np.arange(len(ORDER))[::-1]
    for ax, d in zip(axes.flat, DS):
        R = d["runs"][d["runs"].k == d["kmain"]]
        mu, sd = [], []
        for m in ORDER:
            if m == "CAt":
                a, b = d["cat"], 0.0
            elif m == "LLM":
                a, b = llm.loc[("Qwen3-8B", d["key"]), "macro"], 0.0
            else:
                s = R[R.method == m].groupby("seed").macro.mean()
                a, b = s.mean(), s.std()
            mu.append(a)
            sd.append(b)
            rows.append(dict(dataset=d["title"], k=d["kmain"], method=m, macro_f1=round(a, 4), sd_seeds=round(b, 4)))
        ax.barh(y, mu, height=0.66, color=[COLOR[track(m)] for m in ORDER], edgecolor="white", lw=1)
        ax.errorbar(mu, y, xerr=sd, fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5)
        rng = [llm.loc[(m, d["key"]), "macro"] for m in LOCAL if (m, d["key"]) in llm.index]  # range of the 7 local LLMs
        ax.plot([min(rng), max(rng)], [y[0], y[0]], color=INK, lw=1.2, marker="|", ms=7)
        ax.axvline(d["ceil"], color=INK, lw=1, ls="--")
        ax.text(d["ceil"], -1.05, f" supervised {d['ceil']:.2f}", fontsize=6.3, va="center")
        ax.axvline(d["rnd"], color=MUTED, lw=1, ls=":")
        ax.set_ylim(-1.5, len(ORDER) - 0.3)
        ax.set_xlim(0, 0.9)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title(f"{d['title']}, k = {d['kmain']}", loc="left")
        grid(ax)
    for ax in axes[1]:
        ax.set_xlabel("Macro-F1")
    for ax in axes[:, 0]:
        ax.set_yticks(y, [LABEL[m] for m in ORDER])
    for ax in axes.flat:
        ax.tick_params(axis="y", length=0)
    legend(fig, [plt.Rectangle((0, 0), 1, 1, color=COLOR[t]) for t in ("unsup", "seeded", "llm")] +
           [plt.Line2D([], [], color=INK, marker="|", ms=7, lw=1.2)],
           ["Unsupervised discovery", "Seeded discovery", "Zero-shot LLM (Qwen3-8B)", "Range of 7 open-weight LLMs"], y=-0.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig2_overview.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIG / "fig2_overview.csv", index=False)


def fig_seeds(DS):
    pairs = [("k-means (e5)", "C1", "D1"), ("NMF (TF-IDF)", "A2", "A2S")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    rows, yt, yl = [], [], []
    for j, d in enumerate(DS):
        R = d["runs"][d["runs"].k == d["kmain"]]
        for i, (lab, a, b) in enumerate(pairs):
            yy = -(j * 2.6 + i)
            yt.append(yy)
            yl.append(f"{d['title']}: {lab}")
            va, vb = R[R.method == a].macro.mean(), R[R.method == b].macro.mean()
            if abs(vb - va) >= 0.01:
                ax.annotate("", xy=(vb, yy), xytext=(va, yy),
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=4, shrinkB=4, mutation_scale=7))
            ax.scatter([va], [yy], s=30, color=MUTED, zorder=3, edgecolor="white", lw=1)
            ax.scatter([vb], [yy], s=30, color=COLOR["seeded"], zorder=3, edgecolor="white", lw=1)
            ax.text(max(va, vb) + 0.008, yy, f"{vb - va:+.3f}", va="center", fontsize=6.5)
            rows.append(dict(dataset=d["title"], algorithm=lab, unseeded=round(va, 4), seeded=round(vb, 4), delta=round(vb - va, 4)))
    ax.set_yticks(yt, yl, fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Macro-F1")
    grid(ax)
    legend(fig, [plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=5), plt.Line2D([], [], color=COLOR["seeded"], marker="o", ls="", ms=5)],
           ["Without seeds", "With the same seed phrases"], y=-0.08)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig3_seeds.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIG / "fig3_seeds.csv", index=False)


def fig_per_category(DS, llm):
    rows = []
    for d in DS:
        t = pd.read_csv(d["percat"], index_col=0)
        f1 = json.loads(llm.loc[("Qwen3-8B", d["key"]), "percat"])
        t["LLM"] = [f1[c] for c in t.index]
        for c, r in t.iterrows():
            rows.append(dict(dataset=d["title"], category=c, share=r["share"], C1=r["C1"], D1=r["D1"], CAt=r["CAt"], LLM=r["LLM"]))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(FIG / "fig4_per_category.csv", index=False)
    panels = [("C1", "Unsupervised k-means (e5)"), ("D1", "Seeded k-means"), ("CAt", "Contrastive attention"), ("LLM", "Zero-shot Qwen3-8B")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        for col, d in zip(DATASET_COLORS, DS):
            x = df[df.dataset == d["title"]]
            ax.scatter(x.share * 100, x[m], s=12, color=col, alpha=0.85, edgecolor="white", lw=0.4)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_ylim(-0.03, 1)
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlabel("Category share of test set (%)", fontsize=7)
        grid(ax, "both")
        ax.text(0.12, 0.95, f"F1 = 0: {(df[m] == 0).mean():.0%} of categories", fontsize=6.3, va="top")
    axes[0].set_ylabel("Per-category F1")
    legend(fig, [plt.Line2D([], [], color=c, marker="o", ls="", ms=4) for c in DATASET_COLORS], [d["title"] for d in DS], y=-0.02, ncol=6, fontsize=6.5, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig4_per_category.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    llm = pd.read_csv(RESULTS_DIR / "llm" / "results.csv").query("cond == 'def'").set_index(["model", "dataset"])
    DS = [load_dataset(name, llm) for name in DATASETS]
    fig_overview(DS, llm)
    fig_seeds(DS)
    fig_per_category(DS, llm)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
