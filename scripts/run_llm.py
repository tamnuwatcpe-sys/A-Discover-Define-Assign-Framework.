"""Assign stage: zero-shot classification of every test set with local open-weight LLMs.

    python scripts/run_llm.py                                   # Qwen/Qwen3-8B, all datasets
    python scripts/run_llm.py Qwen/Qwen3-4B microsoft/Phi-4-mini-instruct --device cuda:1
    python scripts/run_llm.py Qwen/Qwen3-32B --max-memory "0:30GiB,1:30GiB"   # split across GPUs

The main model (Qwen3-8B) runs three prompt conditions (names, def, def_seeds); other models run "def".
Answers are cached in results/llm/<model>_<dataset>_<cond>.npy (raw text in *_raw.csv);
run scripts/score_llm.py afterwards to compute macro-F1 and the comparison with CAt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d.config import DATASETS, MAIN_MODEL, RESULTS_DIR, short_name  # noqa: E402
from d2d.data import load_categories, load_corpus  # noqa: E402
from d2d.llm import CONDITIONS, classify, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=[MAIN_MODEL])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--device", default=None, help="e.g. cuda:1 (default: D2D_LLM_DEVICE or cuda:0)")
    ap.add_argument("--max-memory", default=None, help='split one model across GPUs, e.g. "0:30GiB,1:30GiB"')
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    out = RESULTS_DIR / "llm"
    out.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        short = short_name(model_id)
        tok, model = load_model(model_id, args.device, args.max_memory)
        conds = CONDITIONS if model_id == MAIN_MODEL else ("def",)
        for name in args.datasets.split(","):
            texts, _, _ = load_corpus(name).test_view()
            cats = load_categories(name)
            for cond in conds:
                p = out / f"{short}_{name}_{cond}.npy"
                if p.exists():
                    continue
                pred, raw = classify(tok, model, DATASETS[name], cats, texts, cond, args.batch_size)
                np.save(p, pred)
                pd.Series(raw).to_csv(out / f"{short}_{name}_{cond}_raw.csv", index=False)
                print(short, name, cond, "unparsed", round((pred == "UNMAPPED").mean(), 4), flush=True)
        del model
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
