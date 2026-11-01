# COMM061 NLP Coursework : PG29

Sequence classification for **Sentiment** and **Sarcasm** detection across three English varieties (en-AU, en-IN, en-UK) using the [BESSTIE](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26) dataset.

University of Surrey — MSc Data Science, Semester 2 (2025–26)

## Team

Four-person group coursework (group PG29): Shah Hussain, Janani Veerakumar, Syed Shahbaz and [@wallihabib-cpu](https://github.com/wallihabib-cpu). I led deployment: the Gradio demo in `src/deployment/` and the Hugging Face model upload (`upload_to_hf.py`).

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
    │   └── analysis.py         # Q1: EDA, distributions, vocab overlap
    ├── models/
    │   ├── classical.py        # Q2.1: TF-IDF + LogReg/SVM baselines
    │   ├── transformer.py      # Q2.1 + Q2.2: RoBERTa fine-tuning
    │   ├── lora_adapter.py     # Q2.3: LoRA adapters per variety
    │   └── prompting.py        # Q4: Error analysis + few-shot prompting
    ├── evaluation/
    │   ├── metrics.py          # Shared metric computation + result formatting
    │   └── visualisation.py    # Confusion matrices, heatmaps, comparison charts
    └── deployment/
        ├── app.py              # Q5.1: Gradio web app
        └── benchmark.py        # Q5.2: Inference timing
```

## How It Works

`main.ipynb` is a thin orchestrator — it imports functions from `src/` and calls them. Each team member works in their own module to avoid merge conflicts.

All experiment results are saved to `results/` as CSV using a shared `format_result_row()` function, so Q3 evaluation is just reading those files. All figures are saved to `figures/` at 150 DPI.

## Dataset

Loaded from HuggingFace at runtime: [`surrey-nlp/BESSTIE-CW-26`](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26)

| Variety | Train | Val | Test | Total | Sarcasm % |
|---------|-------|-----|------|-------|-----------|
| en-AU   | 1,145 | 95  | 667  | 1,907 | 29.4%     |
| en-IN   | 1,399 | 117 | 816  | 2,332 | 6.8%      |
| en-UK   | 1,203 | 101 | 700  | 2,004 | 7.6%      |


## Deployment

### Running the Gradio App

```bash
python -m src.deployment.app
```

This launches a local Gradio interface for sarcasm and sentiment detection. Select an English variety (en-AU, en-IN, en-UK) and enter text to analyse. The app loads the appropriate LoRA adapter from HuggingFace automatically.

To generate a public shareable link, open the app from a notebook:

```python
from src.deployment.app import build_app
app = build_app()
app.launch(share=True)
```

### Running Benchmarks

```bash
python -m src.deployment.benchmark
```

Benchmarks SVM, RoBERTa and LoRA inference latency across short, medium and long inputs. Results are saved to `results/q52_benchmarks.csv` and the radar chart to `figures/q52_benchmark_chart.png`.


## Key Conventions

- **Config:** All hyperparameters live in `src/config.py` — never hardcode values
- **Data:** All loading goes through `src/data/loader.py` — never call `load_dataset()` directly
- **Metrics:** All evaluation goes through `src/evaluation/metrics.py` — never compute F1 manually
- **Seeds:** Minimum 2 runs per experiment (42, 123)
- **Imbalance:** `class_weight='balanced'` (sklearn), weighted CrossEntropyLoss (torch)
- **Figures:** PNG at 150 DPI in `figures/`
- **Results:** CSV in `results/` via `format_result_row()`
