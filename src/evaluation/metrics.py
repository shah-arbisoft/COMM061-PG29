"""
Shared evaluation utilities.

Every experiment module imports compute_metrics from here so we get
consistent metric computation and result formatting everywhere.
"""

import random
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, label_names=None):
    """
    Compute all required evaluation metrics.

    Returns:
        dict with macro_f1, precision_macro, recall_macro,
        per-class classification_report, and confusion_matrix
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_class_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_class_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_class_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_class_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "majority_class_only": len(set(y_pred)) == 1,
    }


def format_result_row(experiment, model, task, variety, seed, metrics, extra=None):
    """
    Create a standardised result dict for appending to the results DataFrame.
    Every experiment uses this so Q3 compilation is painless.
    """
    row = {
        "experiment": experiment,
        "model": model,
        "task": task,
        "variety": variety,
        "seed": seed,
        "macro_f1": round(metrics["macro_f1"], 4),
        "precision_macro": round(metrics["precision_macro"], 4),
        "recall_macro": round(metrics["recall_macro"], 4),
        "f1_class_0": round(metrics["f1_class_0"], 4),
        "f1_class_1": round(metrics["f1_class_1"], 4),
        "precision_class_0": round(metrics["precision_class_0"], 4),
        "precision_class_1": round(metrics["precision_class_1"], 4),
        "recall_class_0": round(metrics["recall_class_0"], 4),
        "recall_class_1": round(metrics["recall_class_1"], 4),
        "majority_class_only": metrics["majority_class_only"],
    }
    if extra:
        row.update(extra)
    return row


def print_results(metrics, title=""):
    """Pretty-print evaluation results to console."""
    if title:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")

    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")

    if metrics["majority_class_only"]:
        print("  ⚠ WARNING: Model only predicts one class!")

    report = metrics.get("classification_report", {})
    for label, m in report.items():
        if isinstance(m, dict) and "f1-score" in m:
            print(f"    {label:>15s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}")
    print()
