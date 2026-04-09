"""
Q3: Evaluation Visualisations
Owner: All (each member generates plots for their experiments)

Confusion matrices, results comparison charts, cross-variety heatmaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import VARIETIES, LABEL_MAPS, FIGURES_DIR


def plot_confusion_matrix(y_true, y_pred, task, title, save_name=None):
    """
    Plot a confusion matrix with proper class labels.

    Args:
        y_true, y_pred: label lists
        task: "Sentiment" or "Sarcasm" (for label names)
        title: plot title
        save_name: filename (without path) to save in figures/
    """
    labels = list(LABEL_MAPS[task].values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, ax=ax, cmap="Blues",
        colorbar=False
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cross_variety_heatmap(matrix_df, task="Sentiment", save_name=None):
    """
    Plot the 3×3 cross-variety evaluation matrix as a heatmap.

    Args:
        matrix_df: DataFrame with varieties as both index and columns, values = Macro-F1
        task: for the title
        save_name: filename to save
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix_df.astype(float), annot=True, fmt=".3f",
        cmap="YlOrRd", ax=ax, vmin=0.4, vmax=1.0,
        linewidths=0.5, square=True
    )
    ax.set_xlabel("Test Variety")
    ax.set_ylabel("Train Variety")
    ax.set_title(f"Cross-Variety Transferability — {task} (Macro-F1)")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_df, task, metric="macro_f1", save_name=None):
    """
    Bar chart comparing models side by side for a given task.

    Args:
        results_df: DataFrame with columns [model, variety, macro_f1, ...]
        task: filter to this task
        metric: which column to plot
    """
    df = results_df[results_df["task"] == task].copy()

    # Average across seeds
    avg = df.groupby(["model", "variety"])[metric].mean().reset_index()
    avg = avg[avg["variety"] != "all"]  # per-variety only

    fig, ax = plt.subplots(figsize=(10, 5))
    models = avg["model"].unique()
    x = np.arange(len(VARIETIES))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = avg[avg["model"] == model]
        vals = [
            model_data[model_data["variety"] == v][metric].values[0]
            if len(model_data[model_data["variety"] == v]) > 0 else 0
            for v in VARIETIES
        ]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(VARIETIES)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {task}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.show()


def compile_all_results():
    """
    Load all saved CSVs from results/ and merge into one master table.

    Returns:
        pd.DataFrame with all experiments
    """
    from src.config import RESULTS_DIR
    dfs = []
    for csv_file in RESULTS_DIR.glob("q2*.csv"):
        dfs.append(pd.read_csv(csv_file))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()
