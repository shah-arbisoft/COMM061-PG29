"""
Q2.3: LoRA Adapter Training (15 marks)
Owner: M3

Train lightweight LoRA adapters on Qwen2.5-1.5B-Instruct per variety
for Sarcasm detection. Upload to HuggingFace for deployment.

Dependencies:
  - src.config (SEEDS, VARIETIES, RESULTS_DIR, LLM_*, LORA_*, HF_ADAPTER_PATHS)
  - src.data.loader (get_variety_splits, get_task_labels)
  - src.data.preprocessing (clean_text)
  - src.evaluation.metrics (compute_metrics, format_result_row, set_seed)
  - peft (LoraConfig, get_peft_model, TaskType, PeftModel)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.config import (
    SEEDS, VARIETIES, RESULTS_DIR,
    LLM_MODEL, LLM_MAX_LENGTH, LLM_EPOCHS, LLM_LR,
    LLM_BATCH_SIZE, LLM_GRADIENT_ACCUMULATION,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    VARIETY_COL
)
from src.data.loader import get_variety_splits, get_task_labels
from src.data.preprocessing import clean_text
from src.evaluation.metrics import compute_metrics, format_result_row, set_seed


# Adapter save directory (not committed to git)
ADAPTER_DIR = RESULTS_DIR.parent / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)


def _get_class_weights(labels):
    """Compute class weights inversely proportional to frequency."""
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted loss for sarcasm imbalance."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_lora_config():
    """
    Return the shared LoRA configuration using values from config.py.

    Returns:
        peft.LoraConfig
    """
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )


def _tokenize_dataset(texts, labels, tokenizer):
    """Convert texts + labels into a HuggingFace Dataset with tokenised inputs."""
    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=LLM_MAX_LENGTH, return_tensors="pt"
    )
    ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    ds.set_format("torch")
    return ds


def _compute_trainer_metrics(eval_pred):
    """Callback for HuggingFace Trainer evaluation."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    metrics = compute_metrics(labels, preds)
    return {"macro_f1": metrics["macro_f1"]}


def train_adapter(variety, seed=42, max_samples=None):
    """
    Train a single LoRA adapter for one variety on Sarcasm.

    Args:
        variety: "en-AU", "en-IN", or "en-UK"
        seed: random seed
        max_samples: cap training samples (None = use all)

    Returns:
        save_path: path to saved adapter weights
    """
    set_seed(seed)

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        LLM_MODEL, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Attach LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    # Upcast ALL trainable parameters to float32 for stable gradients
    # (LoRA layers + classification head which PEFT marks as modules_to_save)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # The base model outputs float16 hidden_states, but the classification
    # head (score) is now float32 — add a hook to cast input on the fly
    def _cast_to_float32(module, args):
        return tuple(
            a.float() if isinstance(a, torch.Tensor) else a for a in args
        )

    for name, module in model.named_modules():
        if "score" in name and isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(_cast_to_float32)

    # Enable gradient checkpointing to reduce activation memory on 6GB GPU
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # Load variety-specific data for Sarcasm
    train_ds, val_ds, _ = get_variety_splits(variety)
    train_texts, train_labels = get_task_labels(train_ds, "Sarcasm")
    val_texts, val_labels = get_task_labels(val_ds, "Sarcasm")

    # Clean text
    train_texts = [clean_text(t) for t in train_texts]
    val_texts = [clean_text(t) for t in val_texts]

    # Cap samples if requested (useful for debugging / low VRAM)
    if max_samples is not None:
        train_texts = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        val_texts = val_texts[:max_samples]
        val_labels = val_labels[:max_samples]

    # Tokenize
    train_dataset = _tokenize_dataset(train_texts, train_labels, tokenizer)
    val_dataset = _tokenize_dataset(val_texts, val_labels, tokenizer)

    # Class weights for sarcasm imbalance
    class_weights = _get_class_weights(train_labels)

    # Training arguments from config
    output_dir = str(ADAPTER_DIR / f"checkpoints_{variety}_{seed}")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=LLM_EPOCHS,
        per_device_train_batch_size=LLM_BATCH_SIZE,
        per_device_eval_batch_size=LLM_BATCH_SIZE,
        gradient_accumulation_steps=LLM_GRADIENT_ACCUMULATION,
        learning_rate=LLM_LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=False,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_trainer_metrics,
    )

    trainer.train()

    # Save adapter weights only (not full model)
    save_path = ADAPTER_DIR / f"{variety}_seed{seed}"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Adapter saved to {save_path}")

    # Free GPU memory so evaluation can load a fresh model
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(save_path)


def load_adapter(adapter_path):
    """
    Load a saved adapter on top of the base model.

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        LLM_MODEL, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.float16
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def predict_with_adapter(model, tokenizer, texts, batch_size=4):
    """Run inference on a list of texts in batches. Returns predicted labels."""
    device = next(model.parameters()).device
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts, truncation=True, padding="max_length",
            max_length=LLM_MAX_LENGTH, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)

        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        all_preds.extend(preds)

    return all_preds


def evaluate_adapter(adapter_path, test_variety):
    """
    Load an adapter and evaluate on a variety's test set.

    Returns:
        (metrics_dict, predictions_list, labels_list)
    """
    model, tokenizer = load_adapter(adapter_path)

    _, _, test_ds = get_variety_splits(test_variety)
    test_texts, test_labels = get_task_labels(test_ds, "Sarcasm")
    test_texts = [clean_text(t) for t in test_texts]

    preds = predict_with_adapter(model, tokenizer, test_texts)
    metrics = compute_metrics(test_labels, preds)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, preds, test_labels


def upload_adapter(adapter_path, hf_repo):
    """Push adapter to HuggingFace Hub for deployment."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"  Uploaded {adapter_path} to {hf_repo}")


def run_lora_experiment(max_samples=None):
    """
    Full Q2.3 pipeline.

    For each seed:
      For each variety:
        1. Train LoRA adapter on Sarcasm
        2. Evaluate on ALL varieties (cross-variety)

    Saves: results/q23_lora.csv

    Usage:
        from src.models.lora_adapter import run_lora_experiment
        df, adapters = run_lora_experiment(max_samples=500)

    Returns:
        (results_df, adapter_paths_dict)
    """
    all_results = []
    adapter_paths = {}

    for seed in SEEDS:
        for train_var in VARIETIES:
            print(f"\n  Training LoRA on {train_var} for Sarcasm (seed={seed})...")

            # Train adapter
            adapter_path = train_adapter(train_var, seed=seed, max_samples=max_samples)
            adapter_paths[f"{train_var}_seed{seed}"] = adapter_path

            # Evaluate on every variety (cross-variety)
            for test_var in VARIETIES:
                print(f"    Evaluating on {test_var}...")
                metrics, preds, labels = evaluate_adapter(adapter_path, test_var)

                all_results.append(format_result_row(
                    experiment="Q2.3", model=f"lora-{train_var}",
                    task="Sarcasm", variety=test_var, seed=seed,
                    metrics=metrics,
                    extra={"train_variety": train_var}
                ))

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "q23_lora.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'q23_lora.csv'}")

    return df, adapter_paths
