"""D2D: a Discover–Define–Assign framework for aspect category analysis with local LLMs.

The package is split by framework stage:
  data, text, embeddings      corpus loading, tokenization, and sentence embeddings
  discovery, cat              Discover stage: unsupervised and seeded clustering, CAt
  llm                         Assign stage: zero-shot classification with a local LLM
  revise                      Validate/Revise stage: rewriting weak definitions
  baselines, metrics          supervised baselines, macro-F1, and the cluster bootstrap
"""

__version__ = "1.0.0"
