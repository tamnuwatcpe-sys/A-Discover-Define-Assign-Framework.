"""Paths and per-dataset settings shared by all scripts."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("D2D_DATA", ROOT / "data"))          # raw datasets (see data/README.md)
RESULTS_DIR = Path(os.environ.get("D2D_RESULTS", ROOT / "results"))  # everything the scripts write
CATEGORY_DIR = ROOT / "categories"                                    # names, definitions, and seed phrases

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
SEEDS = [0, 1, 2, 3, 4]  # random seeds for every discovery method


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    lang: str          # "en", "ar", or "th"; selects the tokenizer
    ks: tuple          # numbers of clusters tried by every discovery method
    k_main: int        # main setting, about twice the number of categories
    language: str      # language name used in the LLM prompt
    unit: str          # text unit used in the LLM prompt
    domain: str        # domain description used in the LLM prompt

    @property
    def out_dir(self) -> Path:
        """Discovery results for this dataset."""
        return RESULTS_DIR / "discovery" / self.name

    @property
    def category_file(self) -> Path:
        return CATEGORY_DIR / f"{self.name}.json"


DATASETS = {d.name: d for d in [
    DatasetSpec("restaurants", "en", (6, 12, 18, 24), 12, "English", "sentence", "restaurant reviews"),
    DatasetSpec("laptops", "en", (11, 22, 44, 66), 44, "English", "sentence", "laptop reviews"),
    DatasetSpec("coursera", "en", (6, 12, 18, 24), 12, "English", "sentence", "reviews of online courses on Coursera"),
    DatasetSpec("arhotels", "ar", (4, 7, 14, 21), 14, "Arabic", "sentence", "hotel reviews"),
    DatasetSpec("thaifin", "th", (8, 16, 32, 48), 32, "Thai", "paragraph",
                "annual reports (Form 56-1) of Thai listed companies"),
    DatasetSpec("truevoice", "th", (12, 23, 46, 69), 46, "Thai", "customer message",
                "customer calls to the call center of a Thai mobile network operator"),
]}

# Display names used in tables and figures.
TITLES = {"restaurants": "Restaurants (English)", "laptops": "Laptops (English)", "coursera": "Coursera (English)",
          "arhotels": "Hotels (Arabic)", "thaifin": "Annual reports (Thai)", "truevoice": "Call center (Thai)"}

# Open-weight LLMs evaluated in the paper (Hugging Face repository IDs).
MAIN_MODEL = "Qwen/Qwen3-8B"
LOCAL_MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-7B-Instruct", "scb10x/llama3.1-typhoon2-8b-instruct",
                "ibm-granite/granite-3.3-8b-instruct", "mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-4-mini-instruct"]
LARGE_MODEL = "Qwen/Qwen3-32B"


def short_name(model: str) -> str:
    """File-name form of a model ID, e.g. 'Qwen/Qwen3-8B' -> 'Qwen3-8B'."""
    return model.split("/")[-1]
