"""
Q1: Data Analysis and Visualisation (15 marks)
Owner: M1

Covers:
  Q1.1 — Label distributions + class imbalance commentary
  Q1.2 — Vocabulary overlap (Jaccard + TF-IDF cosine) + linguistic distance
  Bonus — POS tag distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from src.config import VARIETIES, LABEL_MAPS, FIGURES_DIR, TASKS
from src.data.loader import get_split_stats, load_besstie, get_split_as_df
from src.data.preprocessing import (
    clean_text, tokenize_spacy, tokenize_spacy_with_pos, get_vocabulary
)


# ----------------------------------------------------------------
# Step 0 — Data Validation (run BEFORE any analysis)
# ----------------------------------------------------------------

def validate_dataset():
    """
    Comprehensive data quality checks on the raw BESSTIE dataset.
    Should be the FIRST thing called in any EDA pipeline.

    Checks:
      1. Missing / null values in text and label columns
      2. Empty or whitespace-only texts
      3. Duplicate texts (exact match)
      4. Label integrity (only 0/1, no unexpected values)
      5. Variety integrity (only expected varieties)
      6. Very short texts (< 10 characters — may be noise)
      7. Very long texts (> 1000 characters — potential outliers)
      8. Text length statistics per variety

    Returns:
        dict with validation results and flags
    """
    ds = load_besstie()
    report = {"splits": {}, "issues_found": False}

    print("=" * 60)
    print("  DATA VALIDATION REPORT")
    print("=" * 60)

    for split_name in ["train", "validation", "test"]:
        df = ds[split_name].to_pandas()
        split_report = {}
        print(f"\n--- {split_name.upper()} ({len(df)} rows) ---")

        # 1. Missing values
        null_text = df["text"].isna().sum()
        null_sentiment = df["Sentiment"].isna().sum()
        null_sarcasm = df["Sarcasm"].isna().sum()
        split_report["null_text"] = int(null_text)
        split_report["null_sentiment"] = int(null_sentiment)
        split_report["null_sarcasm"] = int(null_sarcasm)

        if null_text + null_sentiment + null_sarcasm > 0:
            print(f"  WARNING: Nulls found — text: {null_text}, "
                  f"Sentiment: {null_sentiment}, Sarcasm: {null_sarcasm}")
            report["issues_found"] = True
        else:
            print(f"  Nulls: None found")

        # 2. Empty or whitespace-only texts
        empty_texts = df["text"].apply(
            lambda t: not t or (isinstance(t, str) and t.strip() == "")
        ).sum()
        split_report["empty_texts"] = int(empty_texts)
        if empty_texts > 0:
            print(f"  WARNING: {empty_texts} empty/whitespace-only texts")
            report["issues_found"] = True
        else:
            print(f"  Empty texts: None")

        # 3. Exact duplicates
        dup_count = df["text"].duplicated().sum()
        split_report["duplicate_texts"] = int(dup_count)
        if dup_count > 0:
            print(f"  WARNING: {dup_count} duplicate texts found")
            report["issues_found"] = True
        else:
            print(f"  Duplicates: None")

        # 4. Label integrity
        sent_values = set(df["Sentiment"].unique())
        sarc_values = set(df["Sarcasm"].unique())
        sent_ok = sent_values.issubset({0, 1})
        sarc_ok = sarc_values.issubset({0, 1})
        split_report["label_integrity"] = sent_ok and sarc_ok

        if not sent_ok:
            print(f"  WARNING: Unexpected Sentiment values: {sent_values}")
            report["issues_found"] = True
        if not sarc_ok:
            print(f"  WARNING: Unexpected Sarcasm values: {sarc_values}")
            report["issues_found"] = True
        if sent_ok and sarc_ok:
            print(f"  Label integrity: OK (both binary 0/1)")

        # 5. Variety integrity
        var_values = set(df["variety"].unique())
        expected_vars = set(VARIETIES)
        var_ok = var_values == expected_vars
        split_report["variety_integrity"] = var_ok
        if not var_ok:
            print(f"  WARNING: Unexpected varieties: {var_values - expected_vars}")
            report["issues_found"] = True
        else:
            print(f"  Varieties: OK ({', '.join(sorted(var_values))})")

        # 6. Text length analysis
        df["text_len"] = df["text"].apply(lambda t: len(str(t)))
        df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))

        very_short = (df["text_len"] < 10).sum()
        very_long = (df["text_len"] > 1000).sum()
        split_report["very_short_texts"] = int(very_short)
        split_report["very_long_texts"] = int(very_long)

        if very_short > 0:
            print(f"  NOTE: {very_short} texts shorter than 10 characters")
            short_examples = df[df["text_len"] < 10]["text"].head(3).tolist()
            for ex in short_examples:
                print(f"    -> \"{ex}\"")
        if very_long > 0:
            print(f"  NOTE: {very_long} texts longer than 1000 characters")

        # 7. Text length statistics per variety
        print(f"\n  Text length stats (characters):")
        for var in VARIETIES:
            var_df = df[df["variety"] == var]
            print(f"    {var}: mean={var_df['text_len'].mean():.0f}, "
                  f"median={var_df['text_len'].median():.0f}, "
                  f"min={var_df['text_len'].min()}, "
                  f"max={var_df['text_len'].max()}")

        print(f"\n  Word count stats:")
        for var in VARIETIES:
            var_df = df[df["variety"] == var]
            print(f"    {var}: mean={var_df['word_count'].mean():.1f}, "
                  f"median={var_df['word_count'].median():.0f}, "
                  f"min={var_df['word_count'].min()}, "
                  f"max={var_df['word_count'].max()}")

        report["splits"][split_name] = split_report

    # Summary
    print("\n" + "=" * 60)
    if report["issues_found"]:
        print("  VALIDATION: Issues found — review warnings above")
    else:
        print("  VALIDATION: All checks passed")
    print("=" * 60)

    return report


def plot_text_length_distribution(save=True):
    """
    Plot text length distributions per variety (character count and word count).
    Important for understanding if varieties differ in verbosity and for
    setting max_length in transformer tokenisers.

    Saves: figures/q1_text_length_dist.png, figures/q1_word_count_dist.png
    """
    df = get_split_as_df("train")
    df["text_len"] = df["text"].apply(lambda t: len(str(t)))
    df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))

    # --- Chart 1: Character length by variety ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, var in enumerate(VARIETIES):
        var_df = df[df["variety"] == var]
        axes[i].hist(var_df["text_len"], bins=40, color="#5dade2",
                     edgecolor="white", alpha=0.8)
        axes[i].set_title(f"{var} (n={len(var_df)})")
        axes[i].set_xlabel("Character Length")
        axes[i].axvline(var_df["text_len"].median(), color="red",
                        linestyle="--", alpha=0.7, label="Median")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Text Length Distribution by Variety (Characters)", fontsize=12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "q1_text_length_dist.png", dpi=150)
    plt.show()

    # --- Chart 2: Word count by variety ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, var in enumerate(VARIETIES):
        var_df = df[df["variety"] == var]
        axes[i].hist(var_df["word_count"], bins=40, color="#58d68d",
                     edgecolor="white", alpha=0.8)
        axes[i].set_title(f"{var} (n={len(var_df)})")
        axes[i].set_xlabel("Word Count")
        axes[i].axvline(var_df["word_count"].median(), color="red",
                        linestyle="--", alpha=0.7, label="Median")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Word Count Distribution by Variety", fontsize=12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "q1_word_count_dist.png", dpi=150)
    plt.show()

    # Print summary
    print("\n  Text Length Summary (training set):")
    summary = df.groupby("variety").agg(
        mean_chars=("text_len", "mean"),
        median_chars=("text_len", "median"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
    ).round(1)
    print(summary.to_string())

    return summary


# ----------------------------------------------------------------
# Q1.1 — Label distributions
# ----------------------------------------------------------------

def plot_label_distributions(save=True):
    """
    Create bar charts showing:
      1. Sentiment (pos/neg) counts per variety across all splits
      2. Sarcasm (sarc/not) counts per variety across all splits
      3. Domain (Google vs Reddit) counts per variety

    Each chart is saved as a PNG in figures/ for the report.
    """
    stats = get_split_stats()

    # Aggregate across splits for the overview charts
    agg = stats.groupby("variety").sum(numeric_only=True).reset_index()

    # --- Chart 1: Sentiment distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(VARIETIES))
    w = 0.35
    ax.bar(x - w / 2, agg["positive_sentiment"], w, label="Positive", color="#5dade2")
    ax.bar(x + w / 2, agg["negative_sentiment"], w, label="Negative", color="#ec7063")
    ax.set_xticks(x)
    ax.set_xticklabels(VARIETIES)
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution by Variety")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "q1_sentiment_dist.png", dpi=150)
    plt.show()

    # --- Chart 2: Sarcasm distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, agg["sarcastic"], w, label="Sarcastic", color="#f5b041")
    ax.bar(x + w / 2, agg["not_sarcastic"], w, label="Not Sarcastic", color="#aeb6bf")
    ax.set_xticks(x)
    ax.set_xticklabels(VARIETIES)
    ax.set_ylabel("Count")
    ax.set_title("Sarcasm Distribution by Variety")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "q1_sarcasm_dist.png", dpi=150)
    plt.show()

    # --- Chart 3: Domain distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, agg["google"], w, label="Google Places", color="#58d68d")
    ax.bar(x + w / 2, agg["reddit"], w, label="Reddit", color="#bb8fce")
    ax.set_xticks(x)
    ax.set_xticklabels(VARIETIES)
    ax.set_ylabel("Count")
    ax.set_title("Source Domain Distribution by Variety")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "q1_domain_dist.png", dpi=150)
    plt.show()

    return stats


def plot_sarcasm_imbalance_per_split(save=True):
    """
    Show the sarcasm ratio per variety per split — this is what the rubric
    specifically asks you to comment on.
    """
    stats = get_split_stats()
    stats["sarcasm_ratio"] = stats["sarcastic"] / stats["total"]

    fig, ax = plt.subplots(figsize=(10, 5))
    splits = ["train", "validation", "test"]
    x_positions = []
    x_labels = []
    colours = {"train": "#3498db", "validation": "#2ecc71", "test": "#e74c3c"}

    bar_idx = 0
    for var in VARIETIES:
        for split in splits:
            row = stats[(stats["variety"] == var) & (stats["split"] == split)]
            if not row.empty:
                ratio = row["sarcasm_ratio"].values[0]
                ax.bar(bar_idx, ratio, color=colours[split], edgecolor="white")
                x_positions.append(bar_idx)
                x_labels.append(f"{var}\n({split[:3]})")
                bar_idx += 1
        bar_idx += 0.5  # gap between varieties

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Sarcasm Ratio")
    ax.set_title("Sarcasm Class Imbalance Across Varieties and Splits")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Balanced (0.5)")

    # Custom legend for splits
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colours[s], label=s.capitalize()) for s in splits]
    legend_elements.append(plt.Line2D([0], [0], color="gray", linestyle="--", label="Balanced"))
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "q1_sarcasm_imbalance.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------
# Q1.2 — Vocabulary overlap
# ----------------------------------------------------------------

def jaccard_similarity(set_a, set_b):
    """Jaccard similarity: |A ∩ B| / |A ∪ B|"""
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def tfidf_cosine_similarity(texts_a, texts_b):
    """
    Fit TF-IDF on both corpora combined, then compute cosine similarity
    between their mean TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
    all_texts = texts_a + texts_b
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Mean vector for each corpus
    vec_a = tfidf_matrix[:len(texts_a)].mean(axis=0)
    vec_b = tfidf_matrix[len(texts_a):].mean(axis=0)

    return float(cos_sim(np.asarray(vec_a), np.asarray(vec_b))[0, 0])


def compute_vocab_overlap(variety_a="en-UK", variety_b="en-IN"):
    """
    Q1.2 — Calculate vocabulary overlap between two varieties.

    Returns:
        dict with jaccard_similarity, tfidf_cosine, vocab_sizes, overlap_size
    """
    df = get_split_as_df("train")

    texts_a = df[df["variety"] == variety_a]["text"].apply(clean_text).tolist()
    texts_b = df[df["variety"] == variety_b]["text"].apply(clean_text).tolist()

    vocab_a = get_vocabulary(texts_a)
    vocab_b = get_vocabulary(texts_b)

    jaccard = jaccard_similarity(vocab_a, vocab_b)
    tfidf_cos = tfidf_cosine_similarity(texts_a, texts_b)

    results = {
        "variety_a": variety_a,
        "variety_b": variety_b,
        "vocab_size_a": len(vocab_a),
        "vocab_size_b": len(vocab_b),
        "overlap_size": len(vocab_a & vocab_b),
        "jaccard_similarity": round(jaccard, 4),
        "tfidf_cosine_similarity": round(tfidf_cos, 4),
    }

    print(f"\nVocabulary Overlap: {variety_a} vs {variety_b}")
    print(f"  {variety_a} vocab size: {results['vocab_size_a']:,}")
    print(f"  {variety_b} vocab size: {results['vocab_size_b']:,}")
    print(f"  Overlap:             {results['overlap_size']:,}")
    print(f"  Jaccard similarity:  {results['jaccard_similarity']}")
    print(f"  TF-IDF cosine:       {results['tfidf_cosine_similarity']}")

    return results


def compute_all_pairwise_overlaps():
    """
    Compute vocabulary overlap for all variety pairs.
    Useful for building a full picture of linguistic distance.

    Returns:
        list of result dicts (one per pair)
    """
    pairs = [
        ("en-UK", "en-AU"),
        ("en-UK", "en-IN"),
        ("en-AU", "en-IN"),
    ]
    all_results = []
    for var_a, var_b in pairs:
        result = compute_vocab_overlap(var_a, var_b)
        all_results.append(result)

    # Summary table
    summary_df = pd.DataFrame(all_results)
    print("\n--- Pairwise Vocabulary Overlap Summary ---")
    print(summary_df[["variety_a", "variety_b", "jaccard_similarity",
                       "tfidf_cosine_similarity"]].to_string(index=False))

    return all_results


# ----------------------------------------------------------------
# Bonus — POS tag analysis
# ----------------------------------------------------------------

def pos_distribution_by_label(task="Sarcasm", variety="en-UK", save=True):
    """
    Compare POS tag distributions between positive/negative examples.
    Helps explain whether sarcastic text uses more adjectives, adverbs, etc.
    """
    df = get_split_as_df("train")
    df = df[df["variety"] == variety]

    pos_counts = {0: Counter(), 1: Counter()}

    for _, row in df.iterrows():
        label = int(row[task])
        token_pos_pairs = tokenize_spacy_with_pos(clean_text(row["text"]))
        for _, pos in token_pos_pairs:
            pos_counts[label][pos] += 1

    # Normalise to proportions
    labels = LABEL_MAPS[task]
    all_pos_tags = sorted(set(pos_counts[0].keys()) | set(pos_counts[1].keys()))

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(all_pos_tags))
    w = 0.35

    for i, (label_id, label_name) in enumerate(labels.items()):
        total = sum(pos_counts[label_id].values())
        if total == 0:
            proportions = [0] * len(all_pos_tags)
        else:
            proportions = [pos_counts[label_id].get(tag, 0) / total for tag in all_pos_tags]
        ax.bar(x + (i - 0.5) * w, proportions, w, label=label_name)

    ax.set_xticks(x)
    ax.set_xticklabels(all_pos_tags, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(f"POS Distribution: {task} — {variety}")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"q1_pos_{task.lower()}_{variety}.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------
# Entry point — called from main.ipynb
# ----------------------------------------------------------------

def run_full_eda():
    """
    Run all Q1 analyses in proper NLP pipeline order:
      Step 0: Data validation (quality checks)
      Step 1: Data profiling (text length distributions)
      Step 2: Label distributions (Q1.1)
      Step 3: Vocabulary overlap (Q1.2)
      Step 4: POS tag analysis (Bonus)

    Usage in main.ipynb:
        from src.data.analysis import run_full_eda
        q1_results = run_full_eda()
    """
    print("=" * 60)
    print("  Q1: Data Analysis and Visualisation")
    print("=" * 60)

    # Step 0 — Validate the dataset before any analysis
    print("\nStep 0 — Data Validation")
    validation = validate_dataset()

    # Step 1 — Text length profiling
    print("\nStep 1 — Text Length Profiling")
    length_summary = plot_text_length_distribution()

    # Step 2 — Q1.1: Label distributions
    print("\nQ1.1 — Label distributions")
    stats = plot_label_distributions()
    plot_sarcasm_imbalance_per_split()

    # Step 3 — Q1.2: Vocabulary overlap (spec asks for inner-circle vs outer-circle)
    print("\nQ1.2 — Vocabulary overlap")
    all_overlaps = compute_all_pairwise_overlaps()

    # Step 4 — Bonus: POS tag analysis
    print("\nBonus — POS tag analysis")
    pos_distribution_by_label("Sarcasm", "en-UK")
    pos_distribution_by_label("Sarcasm", "en-AU")
    pos_distribution_by_label("Sarcasm", "en-IN")

    return {
        "validation": validation,
        "length_summary": length_summary,
        "stats": stats,
        "vocab_overlaps": all_overlaps,
    }
