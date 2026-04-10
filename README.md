Sequence classification for **Sentiment** and **Sarcasm** detection across three English varieties (en-AU, en-IN, en-UK) using the [BESSTIE](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26) dataset.

## Quick Start

```bash
# Clone
git clone https://github.com/shah-arbisoft/COMM061-PG29.git
cd COMM061-PG29

# Virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run
jupyter notebook main.ipynb
```

## Project Structure

```
COMM061-PG29/
├── main.ipynb                  # Orchestrator notebook — run this
├── requirements.txt            # Pinned dependencies
├── figures/                    # Saved charts (auto-created)
├── results/                    # Saved CSV results (auto-created)
└── src/
    ├── config.py               # All hyperparameters — single source of truth
    ├── data/
    │   ├── loader.py           # Dataset loading, filtering, caching
    │   ├── preprocessing.py    # Text cleaning + spaCy tokenisation
    │   └── analysis.py         # EDA, distributions, vocab overlap
    ├── models/
    │   ├── classical.py        # TF-IDF + LogReg/SVM baselines
    │   ├── transformer.py      # RoBERTa fine-tuning
    │   ├── lora_adapter.py     # LoRA adapters per variety
    │   └── prompting.py        # Error analysis + few-shot prompting
    ├── evaluation/
    │   ├── metrics.py          # Shared metric computation + result formatting
    │   └── visualisation.py    # Confusion matrices, heatmaps, comparison charts
    └── deployment/
        ├── app.py              # Gradio web app
        └── benchmark.py        # Inference timing
```

## How It Works

`main.ipynb` is a thin orchestrator, it imports functions from `src/` and calls them. Each team member works in their own module to avoid merge conflicts.

All experiment results are saved to `results/` as CSV using a shared `format_result_row()` function, so Q3 evaluation is just reading those files. All figures are saved to `figures/` at 150 DPI.

## Dataset

Loaded from HuggingFace at runtime: [`surrey-nlp/BESSTIE-CW-26`](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26)

| Variety | Train | Val | Test | Total | Sarcasm % |
|---------|-------|-----|------|-------|-----------|
| en-AU   | 1,145 | 95  | 667  | 1,907 | 29.4%     |
| en-IN   | 1,399 | 117 | 816  | 2,332 | 6.8%      |
| en-UK   | 1,203 | 101 | 700  | 2,004 | 7.6%      |

Do NOT commit the dataset or any model checkpoints.

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Protected | Stable code — PRs only |
| `m1/data-analysis` | EDA and visualisation |
| `m2/baselines` | Classical and transformer experiments |
| `m3/lora` | LoRA adapters and error analysis |


## Key Conventions

- **Config:** All hyperparameters live in `src/config.py` — never hardcode values
- **Data:** All loading goes through `src/data/loader.py` — never call `load_dataset()` directly
- **Metrics:** All evaluation goes through `src/evaluation/metrics.py` — never compute F1 manually
- **Seeds:** Minimum 2 runs per experiment (42, 123)
- **Imbalance:** `class_weight='balanced'` (sklearn), weighted CrossEntropyLoss (torch)
- **Figures:** PNG at 150 DPI in `figures/`
- **Results:** CSV in `results/` via `format_result_row()`
