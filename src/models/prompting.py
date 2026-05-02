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
  - src.models.lora_adapter (load_adapter, predict_with_adapter)
  - src.data.loader (get_variety_splits, get_task_labels)
  - src.data.preprocessing (clean_text)
  - src.evaluation.metrics (compute_metrics)
"""

import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import (
    LLM_MODEL, RESULTS_DIR, LABEL_MAPS, SEEDS, VARIETIES
)
from src.models.lora_adapter import load_adapter, predict_with_adapter
from src.data.loader import get_variety_splits, get_task_labels
from src.data.preprocessing import clean_text
from src.evaluation.metrics import compute_metrics, set_seed


SARCASM_LABELS = LABEL_MAPS["Sarcasm"]   # {0: "Not Sarcastic", 1: "Sarcastic"}


def extract_errors(adapter_path, variety, n=10, seed=42):
    """
    Run the adapter on the test set and extract n misclassified examples.

    We select a mix of false positives and false negatives to give
    a representative sample for error analysis.

    Args:
        adapter_path: path to saved LoRA adapter
        variety: English variety to evaluate on
        n: number of errors to return
        seed: for reproducibility

    Returns:
        list of dicts: [{text, true_label, predicted_label, index,
                         true_name, pred_name}, ...]
    """
    set_seed(seed)

    # Load adapter and get predictions
    model, tokenizer = load_adapter(adapter_path)
    _, _, test_ds = get_variety_splits(variety)
    test_texts, test_labels = get_task_labels(test_ds, "Sarcasm")
    test_texts_clean = [clean_text(t) for t in test_texts]

    preds = predict_with_adapter(model, tokenizer, test_texts_clean)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Collect all misclassifications
    errors = []
    for i, (text, true, pred) in enumerate(zip(test_texts, test_labels, preds)):
        if true != pred:
            errors.append({
                "index": i,
                "text": text,
                "true_label": int(true),
                "predicted_label": int(pred),
                "true_name": SARCASM_LABELS[int(true)],
                "pred_name": SARCASM_LABELS[int(pred)],
                "error_type": "false_positive" if pred == 1 else "false_negative",
            })

    # Select a balanced mix: try to get both FP and FN
    fps = [e for e in errors if e["error_type"] == "false_positive"]
    fns = [e for e in errors if e["error_type"] == "false_negative"]

    selected = []
    # Alternate between FP and FN for balanced selection
    fp_idx, fn_idx = 0, 0
    while len(selected) < n and (fp_idx < len(fps) or fn_idx < len(fns)):
        if fn_idx < len(fns) and (fp_idx >= len(fps) or len(selected) % 2 == 0):
            selected.append(fns[fn_idx])
            fn_idx += 1
        elif fp_idx < len(fps):
            selected.append(fps[fp_idx])
            fp_idx += 1

    total_errors = len(errors)
    total_fp = len(fps)
    total_fn = len(fns)
    print(f"  Total test samples: {len(test_labels)}")
    print(f"  Total errors: {total_errors} ({total_fp} FP, {total_fn} FN)")
    print(f"  Selected {len(selected)} errors for analysis")

    return selected


def build_few_shot_prompt(explained_examples):
    """
    Build a few-shot prompt from 4 examples with human-written explanations.

    The prompt instructs the model to classify text as Sarcastic or Not Sarcastic,
    using the explained examples as demonstrations.

    Args:
        explained_examples: list of 4 dicts, each with keys:
            text, true_label, true_name, explanation

    Returns:
        prompt_template: string with {input_text} placeholder
    """
    prompt_parts = [
        "You are a sarcasm detection system for English text. "
        "Classify each input as either 'Sarcastic' or 'Not Sarcastic'. "
        "Consider context, tone indicators, and whether the literal meaning "
        "contradicts the intended meaning.\n\n"
        "Here are some examples with explanations:\n"
    ]

    for i, ex in enumerate(explained_examples, 1):
        prompt_parts.append(
            f"Example {i}:\n"
            f"Text: \"{ex['text']}\"\n"
            f"Label: {ex['true_name']}\n"
            f"Explanation: {ex['explanation']}\n"
        )

    prompt_parts.append(
        "Now classify the following text. "
        "Respond with ONLY 'Sarcastic' or 'Not Sarcastic' on the first line, "
        "followed by a brief explanation.\n\n"
        "Text: \"{input_text}\"\n"
        "Label:"
    )

    return "\n".join(prompt_parts)


def _parse_llm_response(response_text):
    """
    Parse the LLM's response to extract the predicted label.

    Returns:
        (predicted_label: int, raw_response: str)
        predicted_label is 1 for Sarcastic, 0 for Not Sarcastic, -1 if ambiguous
    """
    response_text = response_text.strip()
    first_line = response_text.split("\n")[0].strip().lower()

    if "not sarcastic" in first_line:
        return 0, response_text
    elif "sarcastic" in first_line:
        return 1, response_text
    else:
        # Try broader matching in the full response
        lower_resp = response_text.lower()
        if "not sarcastic" in lower_resp:
            return 0, response_text
        elif "sarcastic" in lower_resp:
            return 1, response_text
        return -1, response_text


def test_with_few_shot(prompt_template, test_examples, max_new_tokens=100):
    """
    Run the remaining 6 examples through the base LLM with the few-shot prompt.

    Uses Qwen2.5-1.5B-Instruct in generative mode (not sequence classification).

    Args:
        prompt_template: string from build_few_shot_prompt with {input_text}
        test_examples: list of dicts from extract_errors (the 6 not used in prompt)
        max_new_tokens: maximum generation length

    Returns:
        list of dicts with: text, true_label, true_name, adapter_pred,
        adapter_pred_name, fewshot_pred, fewshot_pred_name, improved, llm_response
    """
    # Load the base model in generative mode
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    results = []
    for ex in test_examples:
        # Format the prompt with the test text
        filled_prompt = prompt_template.format(input_text=ex["text"])

        # Use chat template for instruct models
        messages = [{"role": "user", "content": filled_prompt}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        )

        # apply_chat_template may return a BatchEncoding or a plain tensor
        if hasattr(prompt_ids, "input_ids"):
            input_ids = prompt_ids["input_ids"].to(model.device)
            attention_mask = prompt_ids["attention_mask"].to(model.device)
        else:
            input_ids = prompt_ids.to(model.device)
            attention_mask = None

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

        # Decode only the generated tokens (exclude the prompt)
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[0][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        fewshot_pred, raw_response = _parse_llm_response(response)

        improved = (
            fewshot_pred == ex["true_label"]
            and ex["predicted_label"] != ex["true_label"]
        )

        results.append({
            "index": ex["index"],
            "text": ex["text"],
            "true_label": ex["true_label"],
            "true_name": ex["true_name"],
            "adapter_pred": ex["predicted_label"],
            "adapter_pred_name": ex["pred_name"],
            "fewshot_pred": fewshot_pred,
            "fewshot_pred_name": SARCASM_LABELS.get(fewshot_pred, "Ambiguous"),
            "improved": improved,
            "llm_response": raw_response,
        })

        print(f"  [{ex['index']}] True={ex['true_name']}, "
              f"Adapter={ex['pred_name']}, "
              f"FewShot={SARCASM_LABELS.get(fewshot_pred, '?')} "
              f"{'IMPROVED' if improved else ''}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_error_analysis(adapter_path, variety, seed=42):
    """
    Full Q4 pipeline.

    This function extracts errors, but the 4 explanations must be
    manually authored — they require linguistic reasoning that
    constitutes the core academic contribution of Q4.

    Usage:
        from src.models.prompting import run_error_analysis
        results = run_error_analysis("adapters/en-AU_seed123", "en-AU", seed=123)

    Returns:
        dict with errors, explained_examples, prompt_template, fewshot_results
    """
    print("=" * 60)
    print("  Q4: Sarcasm Error Analysis")
    print("=" * 60)

    # Step 1: Extract 10 misclassified examples
    print("\n[Step 1] Extracting misclassified examples...")
    errors = extract_errors(adapter_path, variety, n=10, seed=seed)

    # Save errors for inspection
    errors_df = pd.DataFrame(errors)
    errors_path = RESULTS_DIR / "q4_errors.csv"
    errors_df.to_csv(errors_path, index=False)
    print(f"  Saved {len(errors)} errors to {errors_path}")
    print("\n  Review these errors and add explanations for 4 of them.")
    print("  Then call build_few_shot_prompt() and test_with_few_shot().\n")

    return {
        "errors": errors,
        "errors_path": str(errors_path),
    }
