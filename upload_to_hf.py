"""
Upload trained models to HuggingFace Hub for Q5.1 deployment.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # enter your HF token

Usage:
    python upload_to_hf.py --username YOUR_HF_USERNAME

This uploads:
    1. LoRA adapters for each variety (best seed)
    2. Fine-tuned RoBERTa sentiment model

After uploading, update HF_ADAPTER_PATHS and HF_ROBERTA_SENTIMENT_PATH
in src/config.py with your actual username.
"""

import argparse
import os
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    VARIETIES, BEST_ADAPTER_SEEDS, ENCODER_MODEL,
    SEEDS, TASKS
)
from src.data.loader import load_besstie, get_task_labels
from src.data.preprocessing import clean_text
from src.models.transformer import fine_tune_roberta
from src.evaluation.metrics import set_seed


ADAPTER_DIR = Path("adapters")


def upload_adapters(username):
    """Upload LoRA adapters to HuggingFace."""
    api = HfApi()

    for variety in VARIETIES:
        seed = BEST_ADAPTER_SEEDS[variety]
        adapter_path = ADAPTER_DIR / f"{variety}_seed{seed}"

        if not adapter_path.exists():
            print(f"  WARNING: {adapter_path} not found, skipping {variety}")
            continue

        repo_id = f"{username}/besstie-lora-{variety}"
        print(f"\n  Uploading {adapter_path} -> {repo_id}")

        # Create repo if it does not exist
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"  Repo creation note: {e}")

        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded {variety} adapter (seed {seed})")


def upload_roberta_sentiment(username):
    """
    Train RoBERTa on combined sentiment data (best seed) and upload to HF.

    If you already have a saved RoBERTa checkpoint, modify this to load
    from disk instead of retraining.
    """
    repo_id = f"{username}/besstie-roberta-sentiment"
    save_dir = Path("models/roberta_sentiment")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if already saved locally
    if (save_dir / "config.json").exists():
        print(f"\n  Loading saved RoBERTa sentiment model from {save_dir}")
    else:
        print(f"\n  Training RoBERTa sentiment model (seed 42)...")
        seed = 42
        set_seed(seed)

        ds = load_besstie()
        train_texts, train_labels = get_task_labels(ds["train"], "Sentiment")
        val_texts, val_labels = get_task_labels(ds["validation"], "Sentiment")
        train_texts = [clean_text(t) for t in train_texts]
        val_texts = [clean_text(t) for t in val_texts]

        model, tokenizer = fine_tune_roberta(
            train_texts, train_labels, val_texts, val_labels, seed, "Sentiment"
        )

        # Save locally
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"  Saved to {save_dir}")

    # Upload
    print(f"  Uploading to {repo_id}")
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  Repo creation note: {e}")

    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Uploaded RoBERTa sentiment model")


def update_config(username):
    """Print the config values to update."""
    print("\n" + "=" * 60)
    print("  UPDATE src/config.py with these values:")
    print("=" * 60)
    print(f"""
HF_ADAPTER_PATHS = {{
    "en-AU": "{username}/besstie-lora-en-AU",
    "en-IN": "{username}/besstie-lora-en-IN",
    "en-UK": "{username}/besstie-lora-en-UK",
}}
HF_ROBERTA_SENTIMENT_PATH = "{username}/besstie-roberta-sentiment"
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload models to HuggingFace")
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--skip-roberta", action="store_true",
                       help="Skip RoBERTa upload (if already uploaded)")
    parser.add_argument("--skip-adapters", action="store_true",
                       help="Skip adapter upload (if already uploaded)")
    args = parser.parse_args()

    if not args.skip_adapters:
        upload_adapters(args.username)

    if not args.skip_roberta:
        upload_roberta_sentiment(args.username)

    update_config(args.username)
