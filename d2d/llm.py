"""Define and Assign stages: zero-shot classification with a local open-weight LLM.

The model reads each text together with the category inventory and names the single category the text
mainly talks about. Decoding is greedy, answers are limited to 12 tokens, and Qwen3's reasoning mode is
switched off. No labeled data are used.
"""
import os

import numpy as np

CONDITIONS = ("names", "def", "def_seeds")  # names only / names + definitions (main) / definitions + seeds
UNMAPPED = "UNMAPPED"


def build_prompt(spec, categories, text, cond="def"):
    """The classification prompt for one text."""
    lines = []
    for name, v in categories.items():
        if cond == "names":
            lines.append(f"- {name}")
            continue
        examples = f" Examples: {'; '.join(v['seeds'])}." if cond == "def_seeds" else ""
        lines.append(f"- {name}: {v['definition']}{examples}")
    return (f"You classify {spec.unit}s from {spec.domain} by the aspect they talk about. The text is in {spec.language}.\n"
            f"Categories:\n" + "\n".join(lines) +
            f"\n\nText: {text}\n\nWhich single category does the text mainly talk about? Answer with the category name only.")


def parse_answer(output, names):
    """Map a free-text answer to the category name it mentions first (longest name wins ties)."""
    o = output.upper()
    hits = sorted((o.find(n.upper()), -len(n), n) for n in names if n.upper() in o)
    return hits[0][2] if hits else UNMAPPED


def load_model(model_id, device=None, max_memory=None):
    """Load a causal LM in half precision.

    max_memory: e.g. "6:30GiB,7:30GiB" to split a large model (Qwen3-32B) across GPUs;
    otherwise the model goes to `device` (default: D2D_LLM_DEVICE or cuda:0).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:  # Llama and Mistral tokenizers have no pad token
        tok.pad_token = tok.eos_token
    if max_memory:
        mm = {int(k): v for k, v in (x.split(":") for x in max_memory.split(","))}
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto", max_memory=mm)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(
            device or os.environ.get("D2D_LLM_DEVICE", "cuda:0"))
    return tok, model.eval()


def chat(tok, model, prompt):
    """Wrap a prompt in the model's chat template (Qwen3 reasoning mode off)."""
    kw = {"enable_thinking": False} if "Qwen3" in model.name_or_path else {}
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kw)


def generate(tok, model, prompts, max_new_tokens, batch_size):
    """Greedy generation for a list of chat-formatted prompts; returns the decoded continuations."""
    import torch
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out


def classify(tok, model, spec, categories, texts, cond="def", batch_size=24):
    """Assign every text to one category; returns (predicted names, raw answers)."""
    prompts = [chat(tok, model, build_prompt(spec, categories, t, cond)) for t in texts]
    raw = generate(tok, model, prompts, max_new_tokens=12, batch_size=batch_size)
    return np.array([parse_answer(o, list(categories)) for o in raw]), raw
