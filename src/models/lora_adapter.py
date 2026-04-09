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


def get_lora_config():
    """
    Return the shared LoRA configuration using values from config.py.

    Returns:
        peft.LoraConfig
    """
    raise NotImplementedError("M3: Implement get_lora_config")


def train_adapter(variety, seed=42, max_samples=None):
    """
    Train a single LoRA adapter for one variety on Sarcasm.

    Args:
        variety: "en-AU", "en-IN", or "en-UK"
        seed: random seed
        max_samples: cap training samples (None = use all)

    Steps:
      1. Load base model (Qwen2.5-1.5B-Instruct) for sequence classification
      2. Attach LoRA adapter using get_lora_config()
      3. Load variety-specific train/val data for Sarcasm
      4. Train with HuggingFace Trainer
      5. Save adapter weights (not full model)

    Returns:
        save_path: path to saved adapter weights
    """
    raise NotImplementedError("M3: Implement train_adapter")


def load_adapter(adapter_path):
    """
    Load a saved adapter on top of the base model.

    Returns:
        (model, tokenizer) tuple
    """
    raise NotImplementedError("M3: Implement load_adapter")


def evaluate_adapter(adapter_path, test_variety):
    """
    Load an adapter and evaluate on a variety's test set.

    Returns:
        (metrics_dict, predictions_list, labels_list)
    """
    raise NotImplementedError("M3: Implement evaluate_adapter")


def upload_adapter(adapter_path, hf_repo):
    """Push adapter to HuggingFace Hub for deployment."""
    raise NotImplementedError("M3: Implement upload_adapter")


def run_lora_experiment(max_samples=None):
    """
    Full Q2.3 pipeline.

    For each seed:
      For each variety:
        1. Train LoRA adapter
        2. Evaluate on ALL varieties (cross-variety)

    Saves: results/q23_lora.csv

    Usage:
        from src.models.lora_adapter import run_lora_experiment
        df, adapters = run_lora_experiment(max_samples=500)

    Returns:
        (results_df, adapter_paths_dict)
    """
    raise NotImplementedError("M3: Implement run_lora_experiment")
