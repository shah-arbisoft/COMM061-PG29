"""
Central configuration — every tuneable value lives here.
Import from this file instead of hardcoding numbers anywhere else.
"""

from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- Reproducibility ----------
SEEDS = [42, 123]

# ---------- Dataset ----------
DATASET_NAME = "surrey-nlp/BESSTIE-CW-26"
VARIETIES = ["en-AU", "en-IN", "en-UK"]
VARIETY_LABELS = {"en-AU": "Australian", "en-IN": "Indian", "en-UK": "British"}
TASKS = ["Sentiment", "Sarcasm"]
TEXT_COL = "text"
VARIETY_COL = "variety"
SOURCE_COL = "source"

# Label mappings for readability in reports
LABEL_MAPS = {
    "Sentiment": {0: "Negative", 1: "Positive"},
    "Sarcasm": {0: "Not Sarcastic", 1: "Sarcastic"},
}

# ---------- Classical baselines (Q2.1) ----------
TFIDF_MAX_FEATURES = 30_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True  # log-normalise TF — standard NLP practice

# ---------- Transformer (Q2.1, Q2.2) ----------
ENCODER_MODEL = "roberta-base"
ENCODER_MAX_LENGTH = 128
ENCODER_EPOCHS = 4
ENCODER_LR = 2e-5
ENCODER_BATCH_SIZE = 16
ENCODER_WARMUP_RATIO = 0.1
ENCODER_WEIGHT_DECAY = 0.01

# ---------- LLM + LoRA (Q2.3) ----------
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LLM_MAX_LENGTH = 128
LLM_EPOCHS = 3
LLM_LR = 2e-4
LLM_BATCH_SIZE = 4
LLM_GRADIENT_ACCUMULATION = 2  # effective batch = 16

# ---------- Evaluation ----------
PRIMARY_METRIC = "macro_f1"

# ---------- Deployment (Q5) ----------
GRADIO_TITLE = "BESSTIE — Sarcasm & Sentiment Detector"
GRADIO_DESCRIPTION = (
    "Select an English variety and enter text. "
    "The system loads the appropriate LoRA adapter for that dialect."
)
# Replace with your actual HuggingFace paths after uploading adapters
# Use best seed per variety: en-AU=123, en-IN=42, en-UK=42
HF_ADAPTER_PATHS = {
    "en-AU": "Shah9212/besstie-lora-en-AU",
    "en-IN": "Shah9212/besstie-lora-en-IN",
    "en-UK": "Shah9212/besstie-lora-en-UK",
}
HF_ROBERTA_SENTIMENT_PATH = "Shah9212/besstie-roberta-sentiment"

# Best adapter seeds per variety (highest in-variety macro_f1)
# en-AU: seed 123 (0.731) > seed 42 (0.703)
# en-IN: seed 42 (0.568) > seed 123 (0.563)
# en-UK: seed 42 (0.640)
BEST_ADAPTER_SEEDS = {
    "en-AU": 123,
    "en-IN": 42,
    "en-UK": 42,
}