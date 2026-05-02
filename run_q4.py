"""
Q4 Runner — Sarcasm Error Analysis & Few-Shot Prompting
=======================================================
Run this script after Q2.3 adapters are trained.

Best adapter: lora-en-AU seed 123 (macro_f1 = 0.731)

Usage:
    python run_q4.py              # Full pipeline
    python run_q4.py --step 1     # Extract errors only
    python run_q4.py --step 2     # Build prompt + test (after reviewing errors)
"""

import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.config import RESULTS_DIR, FIGURES_DIR, LABEL_MAPS
from src.models.prompting import (
    extract_errors, build_few_shot_prompt,
    test_with_few_shot, SARCASM_LABELS
)
from src.evaluation.metrics import set_seed

# ---------- Configuration ----------
ADAPTER_PATH = "adapters/en-AU_seed123"   # best performing adapter
VARIETY = "en-AU"
SEED = 123

# ---------- Manually authored explanations ----------
# After running Step 1, inspect q4_errors.csv and fill in the
# explanations below for the 4 chosen examples.
# The indices refer to positions in the errors list (0-9), NOT dataset indices.
CHOSEN_INDICES = [0, 1, 2, 3]  # pick 4 from the 10 extracted errors

# IMPORTANT: These explanations MUST be written by a human.
# They should contain linguistically sound reasoning about why
# the text is or is not sarcastic.
EXPLANATIONS = [
    # Explanation for error 0
    "",
    # Explanation for error 1
    "",
    # Explanation for error 2
    "",
    # Explanation for error 3
    "",
]


def step1_extract_errors():
    """Extract 10 misclassified sarcasm examples from the best adapter."""
    print("\n" + "=" * 60)
    print("  STEP 1: Extracting Misclassified Examples")
    print("=" * 60)

    set_seed(SEED)
    errors = extract_errors(ADAPTER_PATH, VARIETY, n=10, seed=SEED)

    # Save for human inspection
    errors_df = pd.DataFrame(errors)
    errors_path = RESULTS_DIR / "q4_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    # Print each error for review
    print("\n  10 Misclassified Examples:")
    print("-" * 60)
    for i, e in enumerate(errors):
        print(f"\n  [{i}] Dataset index: {e['index']}")
        print(f"      True: {e['true_name']}  |  Predicted: {e['pred_name']}  |  Type: {e['error_type']}")
        print(f"      Text: {e['text'][:150]}...")

    print(f"\n  Saved to {errors_path}")
    print("  >>> Review the errors and fill in EXPLANATIONS in run_q4.py <<<")
    print("  >>> Then run: python run_q4.py --step 2 <<<")

    return errors


def step2_fewshot_evaluation(errors=None):
    """Build few-shot prompt from explained examples and test on remaining 6."""
    print("\n" + "=" * 60)
    print("  STEP 2: Few-Shot Prompting Evaluation")
    print("=" * 60)

    # Load errors if not passed in
    if errors is None:
        errors_path = RESULTS_DIR / "q4_errors.csv"
        if not errors_path.exists():
            print("  ERROR: Run step 1 first to extract errors.")
            return
        errors_df = pd.read_csv(errors_path)
        errors = errors_df.to_dict("records")

    # Validate explanations
    for i, idx in enumerate(CHOSEN_INDICES):
        if not EXPLANATIONS[i]:
            print(f"  ERROR: Explanation {i} (error index {idx}) is empty.")
            print("  Fill in EXPLANATIONS in run_q4.py before running step 2.")
            return

    # Separate into explained (4) and test (6)
    explained = []
    test_indices = []

    for i, idx in enumerate(CHOSEN_INDICES):
        ex = errors[idx].copy()
        ex["explanation"] = EXPLANATIONS[i]
        explained.append(ex)

    for i in range(len(errors)):
        if i not in CHOSEN_INDICES:
            test_indices.append(i)
    test_examples = [errors[i] for i in test_indices]

    # Build the few-shot prompt
    print("\n[Building few-shot prompt from 4 explained examples...]")
    prompt = build_few_shot_prompt(explained)

    # Save prompt for the report
    prompt_path = RESULTS_DIR / "q4_fewshot_prompt.txt"
    prompt_path.write_text(prompt)
    print(f"  Prompt saved to {prompt_path}")
    print(f"\n  Prompt preview:\n{'-'*40}")
    print(prompt[:500] + "...")

    # Test remaining 6 examples
    print(f"\n[Testing {len(test_examples)} remaining examples with few-shot prompt...]")
    fewshot_results = test_with_few_shot(prompt, test_examples)

    # Save results
    results_df = pd.DataFrame(fewshot_results)
    results_path = RESULTS_DIR / "q4_fewshot_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")

    # Summary
    n_improved = sum(1 for r in fewshot_results if r["improved"])
    n_correct_fewshot = sum(1 for r in fewshot_results if r["fewshot_pred"] == r["true_label"])
    n_total = len(fewshot_results)

    print(f"\n  Before/After Summary:")
    print(f"  {'='*50}")
    print(f"  Adapter correct on these 6:    0 / {n_total}  (all were errors)")
    print(f"  Few-shot correct on these 6:   {n_correct_fewshot} / {n_total}")
    print(f"  Improved predictions:          {n_improved} / {n_total}")

    # Generate comparison figure
    _plot_before_after(fewshot_results, explained)

    # Save full analysis as JSON for report generation
    analysis = {
        "adapter": ADAPTER_PATH,
        "variety": VARIETY,
        "seed": SEED,
        "total_errors_extracted": len(errors),
        "explained_examples": explained,
        "fewshot_results": fewshot_results,
        "n_improved": n_improved,
        "n_correct_fewshot": n_correct_fewshot,
        "prompt": prompt,
    }
    analysis_path = RESULTS_DIR / "q4_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"  Full analysis saved to {analysis_path}")

    return analysis


def _plot_before_after(fewshot_results, explained_examples):
    """Generate comparison figures for Q4 report."""

    # Figure 1: Before/after comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))

    examples = [f"Ex {i+1}" for i in range(len(fewshot_results))]
    true_labels = [r["true_label"] for r in fewshot_results]
    adapter_preds = [r["adapter_pred"] for r in fewshot_results]
    fewshot_preds = [r["fewshot_pred"] for r in fewshot_results]

    x = np.arange(len(examples))
    width = 0.25

    colors_true = ["#4A4A4A" for _ in true_labels]
    colors_adapter = ["#B22222" for _ in adapter_preds]  # all wrong
    colors_fewshot = [
        "#2E7D32" if fp == tl else "#B22222"
        for fp, tl in zip(fewshot_preds, true_labels)
    ]

    bars1 = ax.bar(x - width, true_labels, width, label="Ground Truth",
                   color="#4A4A4A", edgecolor="white")
    bars2 = ax.bar(x, adapter_preds, width, label="Adapter (before)",
                   color="#B22222", edgecolor="white", alpha=0.8)
    bars3 = ax.bar(x + width, fewshot_preds, width, label="Few-Shot (after)",
                   color=colors_fewshot, edgecolor="white", alpha=0.8)

    ax.set_xlabel("Test Example", fontsize=10)
    ax.set_ylabel("Predicted Label (0=Not Sarcastic, 1=Sarcastic)", fontsize=9)
    ax.set_title("Before vs. After: Adapter vs. Few-Shot Predictions", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(examples)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not Sarcastic", "Sarcastic"], fontsize=9)
    ax.legend(fontsize=9, loc="upper right")

    # Mark improved predictions
    for i, r in enumerate(fewshot_results):
        if r["improved"]:
            ax.annotate("improved", (x[i] + width, fewshot_preds[i]),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=7, color="#2E7D32", fontweight="bold")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "q4_before_after.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # Figure 2: Error type breakdown
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Error type distribution in the 10 extracted errors
    error_types = ["False Negative\n(missed sarcasm)",
                   "False Positive\n(false alarm)"]
    fn_count = sum(1 for r in fewshot_results if r["adapter_pred"] == 0)
    fp_count = len(fewshot_results) - fn_count
    # Include the 4 explained ones too
    fn_explained = sum(1 for e in explained_examples if e["predicted_label"] == 0)
    fp_explained = len(explained_examples) - fn_explained

    all_fn = fn_count + fn_explained
    all_fp = fp_count + fp_explained

    bars = axes[0].bar(error_types, [all_fn, all_fp],
                       color=["#5C5C5C", "#8C8C8C"], edgecolor="white")
    axes[0].set_title("Error Type Distribution (10 examples)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=9)
    for bar, val in zip(bars, [all_fn, all_fp]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha="center", fontsize=10, fontweight="bold")

    # Right: Few-shot correction rate
    n_improved = sum(1 for r in fewshot_results if r["improved"])
    n_unchanged = len(fewshot_results) - n_improved
    axes[1].bar(["Corrected", "Still Wrong"], [n_improved, n_unchanged],
               color=["#2E7D32", "#B22222"], edgecolor="white", alpha=0.85)
    axes[1].set_title("Few-Shot Correction Rate (6 test examples)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=9)
    for i, (val, label) in enumerate(zip([n_improved, n_unchanged],
                                          ["Corrected", "Still Wrong"])):
        axes[1].text(i, val + 0.1, str(val), ha="center",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "q4_error_breakdown.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q4 Error Analysis")
    parser.add_argument("--step", type=int, default=0,
                       help="1=extract errors only, 2=few-shot eval, 0=both")
    args = parser.parse_args()

    if args.step == 0 or args.step == 1:
        errors = step1_extract_errors()

    if args.step == 0 or args.step == 2:
        step2_fewshot_evaluation(errors if args.step == 0 else None)
