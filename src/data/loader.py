"""
Dataset loading and splitting utilities.

Every other module imports data through here — nobody calls load_dataset() directly.
This gives us one place to fix bugs or change the loading logic.
"""

import pandas as pd
from datasets import load_dataset, Dataset
from src.config import DATASET_NAME, VARIETIES, TEXT_COL, VARIETY_COL

# Cache the dataset after first load so we don't re-download every call
_CACHE = {}


def load_besstie(force_reload=False):
    """
    Load the full BESSTIE-CW-26 dataset.

    Returns:
        DatasetDict with keys: train, validation, test
    """
    if "ds" not in _CACHE or force_reload:
        _CACHE["ds"] = load_dataset(DATASET_NAME)
    return _CACHE["ds"]


def get_split_as_df(split="train"):
    """Load a split and return it as a pandas DataFrame."""
    ds = load_besstie()
    return ds[split].to_pandas()


def filter_by_variety(dataset, variety):
    """Filter a HuggingFace Dataset to a single variety."""
    return dataset.filter(lambda x: x[VARIETY_COL] == variety)


def get_variety_splits(variety):
    """
    Get train/val/test for one variety.

    Args:
        variety: one of "en-AU", "en-IN", "en-UK"

    Returns:
        (train_ds, val_ds, test_ds) — each a HF Dataset filtered to that variety
    """
    ds = load_besstie()
    return (
        filter_by_variety(ds["train"], variety),
        filter_by_variety(ds["validation"], variety),
        filter_by_variety(ds["test"], variety),
    )


def get_task_labels(dataset, task):
    """
    Extract text and labels for a specific task from a dataset split.

    Args:
        dataset: HF Dataset or pandas DataFrame
        task: "Sentiment" or "Sarcasm"

    Returns:
        (texts, labels) — both as lists
    """
    if isinstance(dataset, pd.DataFrame):
        texts = dataset[TEXT_COL].tolist()
        labels = dataset[task].astype(int).tolist()
    else:
        texts = dataset[TEXT_COL]
        labels = [int(x) for x in dataset[task]]
    return texts, labels


def get_combined_train(task):
    """
    Get all training data combined across varieties for a task.
    Used for Q2.1 baselines.

    Returns:
        (texts, labels)
    """
    ds = load_besstie()
    return get_task_labels(ds["train"], task)


def get_split_stats():
    """
    Return a DataFrame with row counts per variety per split.
    Useful for the Q1 data analysis.
    """
    ds = load_besstie()
    rows = []
    for split_name in ["train", "validation", "test"]:
        df = ds[split_name].to_pandas()
        for var in VARIETIES:
            subset = df[df[VARIETY_COL] == var]
            rows.append({
                "split": split_name,
                "variety": var,
                "total": len(subset),
                "positive_sentiment": int(subset["Sentiment"].sum()),
                "negative_sentiment": int((subset["Sentiment"] == 0).sum()),
                "sarcastic": int(subset["Sarcasm"].sum()),
                "not_sarcastic": int((subset["Sarcasm"] == 0).sum()),
                "google": int((subset["source"] == "Google").sum()),
                "reddit": int((subset["source"] == "Reddit").sum()),
            })
    return pd.DataFrame(rows)
