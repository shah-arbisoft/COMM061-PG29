"""
Q4: Sarcasm Explanation and Error Analysis (10 marks)
Owner: M3

Pipeline:
  1. Run best adapter on test set, collect misclassifications
  2. Human-write explanations for 4 of them
  3. Build a few-shot prompt with those 4 examples
  4. Re-test remaining 6 with the prompt
  5. Compare before/after

Dependencies:
  - src.config (LLM_MODEL, LLM_MAX_LENGTH, RESULTS_DIR)
  - src.models.lora_adapter (load_adapter, evaluate_adapter)
  - src.data.loader (get_variety_splits, get_task_labels)
  - src.data.preprocessing (clean_text)
  - src.evaluation.metrics (compute_metrics)
"""


def extract_errors(adapter_path, variety, n=10):
    """
    Run the adapter on a test set and extract n misclassified examples.

    Returns:
        list of dicts: [{text, true_label, predicted_label, index,
                         true_name, pred_name}, ...]
    """
    raise NotImplementedError("M3: Implement extract_errors")


def build_few_shot_prompt(explained_examples):
    """
    Build a few-shot prompt from 4 examples with human-written explanations.

    Args:
        explained_examples: list of 4 dicts, each with keys:
            text, true_label, true_name, explanation

    Returns:
        prompt_template: string with {input_text} placeholder
    """
    raise NotImplementedError("M3: Implement build_few_shot_prompt")


def test_with_few_shot(prompt_template, test_examples):
    """
    Run the remaining examples through the LLM with the few-shot prompt.

    Returns:
        list of dicts with original + new prediction + improved flag
    """
    raise NotImplementedError("M3: Implement test_with_few_shot")


def run_error_analysis(adapter_path, variety):
    """
    Full Q4 pipeline.

    Usage:
        from src.models.prompting import run_error_analysis
        results = run_error_analysis("path/to/adapter", "en-UK")

    NOTE: M3 must manually fill in the 'explanation' field for the first 4 errors.
    This function provides the scaffold — the linguistic reasoning is human work.

    Returns:
        dict with errors, explained_examples, prompt_template, fewshot_results
    """
    raise NotImplementedError("M3: Implement run_error_analysis")
