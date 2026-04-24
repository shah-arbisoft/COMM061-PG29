"""
Q2.1 (partial) + Q2.2: Transformer Experiments
Owner: M2

Q2.1: Fine-tune RoBERTa on combined data for both tasks.
Q2.2: Fine-tune RoBERTa per variety for sentiment, build 3x3 matrix.
"""

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

from src.config import (
    SEEDS, VARIETIES, TASKS, RESULTS_DIR,
    ENCODER_MODEL, ENCODER_MAX_LENGTH, ENCODER_EPOCHS,
    ENCODER_LR, ENCODER_BATCH_SIZE, ENCODER_WARMUP_RATIO,
    ENCODER_WEIGHT_DECAY, TEXT_COL, VARIETY_COL
)
from src.data.loader import load_besstie, get_variety_splits, get_task_labels
from src.data.preprocessing import clean_text
from src.evaluation.metrics import compute_metrics, format_result_row, set_seed


def _get_class_weights(labels):
    """
    Compute class weights inversely proportional to frequency.
    Used to handle sarcasm imbalance.
    """
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _tokenize_dataset(texts, labels, tokenizer):
    """Convert texts + labels into a HuggingFace Dataset with tokenised inputs."""
    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=ENCODER_MAX_LENGTH, return_tensors="pt"
    )
    ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    ds.set_format("torch")
    return ds


def _compute_trainer_metrics(eval_pred):
    """Callback for HuggingFace Trainer."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    metrics = compute_metrics(labels, preds)
    return {"macro_f1": metrics["macro_f1"]}


def fine_tune_roberta(train_texts, train_labels, val_texts, val_labels, seed, task):
    """
    Fine-tune RoBERTa-base and return the trained model + tokenizer.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER_MODEL, num_labels=2
    )

    train_ds = _tokenize_dataset(train_texts, train_labels, tokenizer)
    val_ds = _tokenize_dataset(val_texts, val_labels, tokenizer)

    class_weights = _get_class_weights(train_labels)

    args = TrainingArguments(
        output_dir=f"/tmp/roberta_{task}_{seed}",
        num_train_epochs=ENCODER_EPOCHS,
        per_device_train_batch_size=ENCODER_BATCH_SIZE,
        per_device_eval_batch_size=ENCODER_BATCH_SIZE * 2,
        learning_rate=ENCODER_LR,
        warmup_ratio=ENCODER_WARMUP_RATIO,
        weight_decay=ENCODER_WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_trainer_metrics,
    )

    trainer.train()
    return model, tokenizer


def predict_with_model(model, tokenizer, texts):
    """Run inference on a list of texts. Returns predicted labels."""
    model.eval()
    device = next(model.parameters()).device
    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=ENCODER_MAX_LENGTH, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)

    return torch.argmax(outputs.logits, dim=1).cpu().tolist()


# ----------------------------------------------------------------
# Q2.1: Combined training
# ----------------------------------------------------------------

def run_transformer_experiment():
    """
    Q2.1: Fine-tune RoBERTa on combined data for both tasks.

    Usage:
        from src.models.transformer import run_transformer_experiment
        df = run_transformer_experiment()
    """
    ds = load_besstie()
    all_results = []

    for seed in SEEDS:
        for task in TASKS:
            print(f"\n  Fine-tuning RoBERTa on {task} (seed={seed})...")

            train_texts, train_labels = get_task_labels(ds["train"], task)
            val_texts, val_labels = get_task_labels(ds["validation"], task)
            train_texts = [clean_text(t) for t in train_texts]
            val_texts = [clean_text(t) for t in val_texts]

            model, tokenizer = fine_tune_roberta(
                train_texts, train_labels, val_texts, val_labels, seed, task
            )

            # Evaluate on full test + per variety
            test_df = ds["test"].to_pandas()
            for var in ["all"] + VARIETIES:
                if var == "all":
                    subset = test_df
                else:
                    subset = test_df[test_df[VARIETY_COL] == var]

                t_texts = [clean_text(t) for t in subset[TEXT_COL]]
                t_labels = subset[task].astype(int).tolist()
                preds = predict_with_model(model, tokenizer, t_texts)
                metrics = compute_metrics(t_labels, preds)

                all_results.append(format_result_row(
                    experiment="Q2.1", model="roberta-base", task=task,
                    variety=var, seed=seed, metrics=metrics
                ))

            # Cleanup GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "q21_roberta.csv", index=False)
    return df


# ----------------------------------------------------------------
# Q2.2: Cross-variety evaluation
# ----------------------------------------------------------------

def run_cross_variety_experiment(task="Sentiment"):
    """
    Q2.2: Train per variety, test on all others.
    Builds a 3x3 transfer matrix (Macro-F1).

    Returns:
        (matrix_df, full_results_df)
    """
    all_results = []

    for seed in SEEDS:
        for train_var in VARIETIES:
            print(f"\n  Training on {train_var} for {task} (seed={seed})...")

            train_ds, val_ds, _ = get_variety_splits(train_var)
            train_texts, train_labels = get_task_labels(train_ds, task)
            val_texts, val_labels = get_task_labels(val_ds, task)
            train_texts = [clean_text(t) for t in train_texts]
            val_texts = [clean_text(t) for t in val_texts]

            model, tokenizer = fine_tune_roberta(
                train_texts, train_labels, val_texts, val_labels, seed, task
            )

            # Evaluate on every variety's test set
            for test_var in VARIETIES:
                _, _, test_ds = get_variety_splits(test_var)
                t_texts, t_labels = get_task_labels(test_ds, task)
                t_texts = [clean_text(t) for t in t_texts]
                preds = predict_with_model(model, tokenizer, t_texts)
                metrics = compute_metrics(t_labels, preds)

                all_results.append(format_result_row(
                    experiment="Q2.2", model=f"roberta-{train_var}",
                    task=task, variety=test_var, seed=seed, metrics=metrics,
                    extra={"train_variety": train_var}
                ))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    full_df = pd.DataFrame(all_results)
    full_df.to_csv(RESULTS_DIR / "q22_cross_variety.csv", index=False)

    # Build the 3x3 matrix (averaged across seeds)
    matrix = pd.DataFrame(index=VARIETIES, columns=VARIETIES, dtype=float)
    for train_var in VARIETIES:
        for test_var in VARIETIES:
            mask = (
                (full_df["train_variety"] == train_var) &
                (full_df["variety"] == test_var)
            )
            matrix.loc[train_var, test_var] = full_df.loc[mask, "macro_f1"].mean()

    matrix.to_csv(RESULTS_DIR / "q22_matrix.csv")
    print("\nCross-variety matrix (Macro-F1):")
    print(matrix.round(4))

    return matrix, full_df
