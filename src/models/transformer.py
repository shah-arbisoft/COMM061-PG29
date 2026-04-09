"""
Q2.1 (partial) + Q2.2: Transformer Experiments
Owner: M2

Q2.1: Fine-tune RoBERTa on combined data for both tasks.
Q2.2: Fine-tune RoBERTa per variety for sentiment, build 3x3 matrix.

Dependencies:
  - src.config (SEEDS, VARIETIES, TASKS, RESULTS_DIR, ENCODER_*)
  - src.data.loader (load_besstie, get_variety_splits, get_task_labels)
  - src.data.preprocessing (clean_text)
  - src.evaluation.metrics (compute_metrics, format_result_row, set_seed)
"""


def fine_tune_roberta(train_texts, train_labels, val_texts, val_labels, seed, task):
    """
    Fine-tune RoBERTa-base and return the trained model + tokenizer.

    Args:
        train_texts: list of cleaned training texts
        train_labels: list of int labels
        val_texts: list of cleaned validation texts
        val_labels: list of int labels
        seed: random seed
        task: "Sentiment" or "Sarcasm"

    Key requirements:
      - Use WeightedTrainer (custom Trainer subclass with class-weighted CrossEntropyLoss)
      - Use config values for all hyperparameters (epochs, lr, batch_size, etc.)
      - eval_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1"

    Returns:
        (model, tokenizer) tuple
    """
    raise NotImplementedError("M2: Implement fine_tune_roberta")


def predict_with_model(model, tokenizer, texts):
    """
    Run inference on a list of texts.

    Returns:
        list of predicted label ints
    """
    raise NotImplementedError("M2: Implement predict_with_model")


# ----------------------------------------------------------------
# Q2.1: Combined training
# ----------------------------------------------------------------

def run_transformer_experiment():
    """
    Q2.1 — Fine-tune RoBERTa on combined data for both tasks.
    Evaluate on full test set + per variety.

    Loops over: seeds × tasks
    Saves results to: results/q21_roberta.csv

    Usage:
        from src.models.transformer import run_transformer_experiment
        df = run_transformer_experiment()

    Returns:
        pd.DataFrame with all results
    """
    raise NotImplementedError("M2: Implement run_transformer_experiment")


# ----------------------------------------------------------------
# Q2.2: Cross-variety evaluation
# ----------------------------------------------------------------

def run_cross_variety_experiment(task="Sentiment"):
    """
    Q2.2 — Train per variety, test on all. Build 3x3 transfer matrix.

    For each seed:
      For each train_variety:
        1. Fine-tune RoBERTa on that variety's train/val data
        2. Evaluate on ALL varieties' test sets

    Build a 3x3 matrix (rows=train_variety, cols=test_variety) of Macro-F1.
    Saves: results/q22_cross_variety.csv and results/q22_matrix.csv

    Returns:
        (matrix_df, full_results_df)
    """
    raise NotImplementedError("M2: Implement run_cross_variety_experiment")
