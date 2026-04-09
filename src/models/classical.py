"""
Q2.1 (partial): Classical Baselines
Owner: M2

TF-IDF + Logistic Regression and TF-IDF + Linear SVM.
Trained on combined data, evaluated per-variety.

Dependencies:
  - src.config (SEEDS, VARIETIES, TASKS, RESULTS_DIR, TFIDF_*, TEXT_COL, VARIETY_COL)
  - src.data.loader (load_besstie, get_task_labels)
  - src.data.preprocessing (clean_text)
  - src.evaluation.metrics (compute_metrics, format_result_row, set_seed)
"""


def train_and_evaluate(task, classifier_name, seed):
    """
    Train a classical baseline on combined training data,
    evaluate on the full test set and per-variety.

    Args:
        task: "Sentiment" or "Sarcasm"
        classifier_name: "logreg" or "svm"
        seed: random seed for reproducibility

    Steps:
      1. Set seed
      2. Load BESSTIE dataset
      3. Get train texts/labels, apply clean_text()
      4. Build TF-IDF + classifier pipeline (use config values, class_weight='balanced')
      5. Fit on combined training data
      6. Evaluate on full test set → format_result_row()
      7. Evaluate per variety → format_result_row() for each

    Returns:
        list of result dicts (one per variety + one overall)
    """
    raise NotImplementedError("M2: Implement train_and_evaluate")


def run_classical_baselines():
    """
    Run all classical baselines for Q2.1.

    Loops over: seeds × tasks × classifiers (logreg, svm)
    Saves results to: results/q21_classical_baselines.csv

    Usage in main.ipynb:
        from src.models.classical import run_classical_baselines
        df = run_classical_baselines()

    Returns:
        pd.DataFrame with all results
    """
    raise NotImplementedError("M2: Implement run_classical_baselines")
