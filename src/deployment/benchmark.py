"""
Q5.2: Efficiency Analysis (5 marks)
Owner: M4

Compare inference time across model types (SVM, RoBERTa, LoRA-LLM)
on varying input sizes.

Dependencies:
  - src.config (RESULTS_DIR)

NOTE: This is a Week 4 task. Requires all three model types to be trained first.
"""


def time_function(fn, *args, n_runs=20, warmup=3, **kwargs):
    """
    Time a function call, returning average milliseconds.
    Includes warmup runs that aren't counted.

    Returns:
        dict with mean_ms, std_ms, min_ms, max_ms
    """
    raise NotImplementedError("M4: Implement in Week 4")


def run_benchmarks(svm_predict_fn, roberta_predict_fn, lora_predict_fn):
    """
    Benchmark all three model types on standard and large inputs.

    Args:
        svm_predict_fn: callable(text) -> prediction
        roberta_predict_fn: callable(text) -> prediction
        lora_predict_fn: callable(text) -> prediction

    Saves: results/q52_benchmarks.csv

    Returns:
        pd.DataFrame with timing results
    """
    raise NotImplementedError("M4: Implement in Week 4")
