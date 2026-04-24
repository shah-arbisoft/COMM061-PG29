"""
Q2.1 (partial): Classical Baselines
Owner: M2

TF-IDF + Logistic Regression and TF-IDF + Linear SVM.
Trained on combined data, evaluated per-variety.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from src.config import (
    SEEDS, VARIETIES, TASKS, RESULTS_DIR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    TEXT_COL, VARIETY_COL
)
from src.data.loader import load_besstie, get_task_labels
from src.data.preprocessing import clean_text
from src.evaluation.metrics import compute_metrics, format_result_row

# Map short names to sklearn classes
CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs"
    ),
    "svm": lambda: LinearSVC(
        class_weight="balanced", max_iter=5000, C=1.0
    ),
}


def _build_pipeline(classifier_name):
    """Build a TF-IDF + classifier pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
            strip_accents="unicode",
            min_df=2,
        )),
        ("clf", CLASSIFIERS[classifier_name]()),
    ])


def train_and_evaluate(task, classifier_name, seed):
    """
    Train a classical baseline on combined training data,
    evaluate on the full test set and per-variety.

    Returns:
        list of result dicts (one per variety + one overall)
    """
    from src.evaluation.metrics import set_seed
    set_seed(seed)

    ds = load_besstie()

    # Train on combined training data
    train_texts, train_labels = get_task_labels(ds["train"], task)
    train_texts = [clean_text(t) for t in train_texts]

    pipe = _build_pipeline(classifier_name)
    pipe.fit(train_texts, train_labels)

    results = []

    # Evaluate on full test set
    test_texts, test_labels = get_task_labels(ds["test"], task)
    test_texts_clean = [clean_text(t) for t in test_texts]
    preds = pipe.predict(test_texts_clean)
    metrics = compute_metrics(test_labels, preds)

    results.append(format_result_row(
        experiment="Q2.1", model=classifier_name, task=task,
        variety="all", seed=seed, metrics=metrics
    ))

    # Evaluate per variety
    test_df = ds["test"].to_pandas()
    for var in VARIETIES:
        var_df = test_df[test_df[VARIETY_COL] == var]
        var_texts = [clean_text(t) for t in var_df[TEXT_COL]]
        var_labels = var_df[task].astype(int).tolist()
        var_preds = pipe.predict(var_texts)
        var_metrics = compute_metrics(var_labels, var_preds)

        results.append(format_result_row(
            experiment="Q2.1", model=classifier_name, task=task,
            variety=var, seed=seed, metrics=var_metrics
        ))

    return results


def run_classical_baselines():
    """
    Run all classical baselines for Q2.1.

    Usage in main.ipynb:
        from src.models.classical import run_classical_baselines
        df = run_classical_baselines()
    """
    all_results = []

    for seed in SEEDS:
        for task in TASKS:
            for clf_name in CLASSIFIERS:
                print(f"  Training {clf_name} on {task} (seed={seed})...")
                results = train_and_evaluate(task, clf_name, seed)
                all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "q21_classical_baselines.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'q21_classical_baselines.csv'}")
    return df
