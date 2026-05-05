"""
Q5.1  Deployment Endpoint
Owner: M4

Gradio web app for sarcasm detection and sentiment analysis across
three English varieties (Australian, Indian, British).

Architecture
    Sarcasm  -- per-variety LoRA adapters (Qwen2.5-1.5B-Instruct)
                hosted on HuggingFace, loaded via from_pretrained()
    Sentiment -- single fine-tuned RoBERTa-base, also on HuggingFace

Switching varieties only reloads the small LoRA adapter weights
(~35 MB each) rather than the full 1.5B-param base, which keeps
latency low when a user changes dialect mid-session.

Usage
    python -m src.deployment.app

Dependencies
    pip install gradio transformers peft torch safetensors huggingface_hub
"""

import time
import torch
import gradio as gr
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from peft import PeftModel

from src.config import (
    VARIETIES, VARIETY_LABELS, LLM_MODEL, LLM_MAX_LENGTH,
    ENCODER_MODEL, ENCODER_MAX_LENGTH,
    HF_ADAPTER_PATHS, HF_ROBERTA_SENTIMENT_PATH,
    GRADIO_TITLE, GRADIO_DESCRIPTION, LABEL_MAPS
)
from src.data.preprocessing import clean_text


# ---- Model cache ----
# We keep loaded models in memory so repeated requests for the same
# variety are instant.  Switching variety only needs to load a new
# LoRA adapter (~35 MB) instead of the full base model again.
_sarcasm_cache = {}      # variety code -> (model, tokenizer)
_sentiment_cache = {}    # "model" -> model, "tokenizer" -> tokenizer


def _get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sarcasm_model(variety):
    """
    Load a variety-specific LoRA sarcasm adapter from HuggingFace.

    Loads the base Qwen model, attaches the LoRA adapter WITHOUT
    merging (to preserve the classification head that PEFT manages
    via modules_to_save), and falls back to manual weight restoration
    if needed.

    Results are cached after the first call per variety.
    """
    if variety in _sarcasm_cache:
        return _sarcasm_cache[variety]

    device = _get_device()
    adapter_repo = HF_ADAPTER_PATHS[variety]
    print(f"[load] Downloading sarcasm adapter for {variety} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_repo, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load frozen base in fp16 to save memory
    base_model = AutoModelForSequenceClassification.from_pretrained(
        LLM_MODEL, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.float16
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Attach LoRA weights - do NOT merge yet, let PEFT handle
    # the classification head through its modules_to_save mechanism
    model = PeftModel.from_pretrained(base_model, adapter_repo)

    # Check if PEFT loaded the classification head correctly
    # by looking at whether score.weight has non-trivial values
    adapter_safetensors = hf_hub_download(
        adapter_repo, "adapter_model.safetensors"
    )
    saved_weights = load_file(adapter_safetensors)

    # Print available keys to diagnose loading
    score_keys = [k for k in saved_weights.keys() if "score" in k]
    print(f"[load] Score keys in safetensors: {score_keys}")

    # Try multiple key patterns for the classification head
    possible_keys = [
        "base_model.model.score.modules_to_save.default.weight",
        "base_model.model.score.original_module.weight",
        "base_model.model.score.weight",
        "score.weight",
    ]

    head_loaded = False
    for key in possible_keys:
        if key in saved_weights:
            print(f"[load] Restoring score.weight from key: {key}")
            # Access the actual linear layer inside PEFT wrapper
            try:
                model.base_model.model.score.modules_to_save.default.weight.data = (
                    saved_weights[key].float()
                )
                head_loaded = True
            except (AttributeError, KeyError):
                pass

            if not head_loaded:
                try:
                    model.base_model.model.score.original_module.weight.data = (
                        saved_weights[key].float()
                    )
                    head_loaded = True
                except (AttributeError, KeyError):
                    pass

            if not head_loaded:
                try:
                    model.score.weight.data = saved_weights[key].float()
                    head_loaded = True
                except (AttributeError, KeyError):
                    pass

            if head_loaded:
                break

    if not head_loaded:
        print("[load] WARNING: Could not restore score.weight!")
        print(f"[load] All keys: {list(saved_weights.keys())[:10]}")

    # Now merge LoRA into base for faster inference
    model = model.merge_and_unload()

    # If we couldn't load via PEFT structure, try direct assignment post-merge
    if not head_loaded:
        for key in possible_keys:
            if key in saved_weights:
                print(f"[load] Post-merge restore from: {key}")
                model.score.weight.data = saved_weights[key].float()
                head_loaded = True
                break

    # Verify the head has meaningful weights (not random init)
    w = model.score.weight.data
    print(f"[load] score.weight stats: mean={w.mean():.4f}, std={w.std():.4f}, "
          f"min={w.min():.4f}, max={w.max():.4f}")

    model = model.float()
    model.eval()
    model.to(device)

    _sarcasm_cache[variety] = (model, tokenizer)
    print(f"[load] {variety} sarcasm model ready on {device}")
    return model, tokenizer


def load_sentiment_model():
    """
    Load the shared RoBERTa sentiment model from HuggingFace.

    A single RoBERTa-base was fine-tuned on the combined training
    data (all varieties) for Sentiment, since sentiment patterns
    generalise well across dialects.  Cached after first load.
    """
    if "model" in _sentiment_cache:
        return _sentiment_cache["model"], _sentiment_cache["tokenizer"]

    device = _get_device()
    print("[load] Downloading RoBERTa sentiment model ...")

    tokenizer = AutoTokenizer.from_pretrained(HF_ROBERTA_SENTIMENT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_ROBERTA_SENTIMENT_PATH, num_labels=2
    )
    model.eval()
    model.to(device)

    _sentiment_cache["model"] = model
    _sentiment_cache["tokenizer"] = tokenizer
    print(f"[load] Sentiment model ready on {device}")
    return model, tokenizer


# ---- Inference helpers ----

def predict_sarcasm(text, variety):
    """
    Run sarcasm inference for one text using the variety adapter.

    The classification head maps index 0 -> Not Sarcastic, index 1 -> Sarcastic
    (matching dataset label convention). Temperature scaling calibrates the
    output so the UI shows reasonable confidence values rather than
    near-100% extremes from raw logits.

    Returns dict  {"Sarcastic": float, "Not Sarcastic": float}
    """
    # Temperature softens extreme logit gaps into usable probabilities
    TEMPERATURE = 3.0

    model, tokenizer = load_sarcasm_model(variety)
    device = next(model.parameters()).device

    cleaned = clean_text(text)
    encodings = tokenizer(
        cleaned, truncation=True, padding="max_length",
        max_length=LLM_MAX_LENGTH, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits.float()

    # Log raw logits for debugging (visible in terminal only)
    raw = logits[0].tolist()
    print(f"  [debug] logits: [{raw[0]:.3f}, {raw[1]:.3f}]")

    # Temperature scale then softmax
    scaled = logits / TEMPERATURE
    probs = torch.nn.functional.softmax(scaled, dim=1)[0]

    # Index 0 = Not Sarcastic, Index 1 = Sarcastic (dataset convention)
    not_sarc = float(probs[0])
    sarc = float(probs[1])

    return {
        "Not Sarcastic": not_sarc,
        "Sarcastic": sarc,
    }


def predict_sentiment(text):
    """
    Run sentiment inference using the shared RoBERTa model.

    Returns dict  {"Negative": float, "Positive": float}
    """
    model, tokenizer = load_sentiment_model()
    device = next(model.parameters()).device

    cleaned = clean_text(text)
    encodings = tokenizer(
        cleaned, truncation=True, padding="max_length",
        max_length=ENCODER_MAX_LENGTH, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.nn.functional.softmax(logits.float(), dim=1)[0]

    return {
        LABEL_MAPS["Sentiment"][i]: float(probs[i])
        for i in range(len(probs))
    }


def predict(text, variety):
    """
    Main handler called by the Gradio UI.

    Takes raw user text + variety display name, runs both sarcasm
    and sentiment models, returns two dicts of {label: confidence}.
    """
    if not text or not text.strip():
        return (
            {"Not Sarcastic": 0.5, "Sarcastic": 0.5},
            {"Negative": 0.5, "Positive": 0.5},
        )

    # Map the human-readable dropdown value back to a variety code
    variety_code = None
    for code, label in VARIETY_LABELS.items():
        if label in variety or code in variety:
            variety_code = code
            break
    if variety_code is None:
        variety_code = "en-AU"  # sensible default

    sarcasm_result = predict_sarcasm(text, variety_code)
    sentiment_result = predict_sentiment(text)

    return sarcasm_result, sentiment_result


# ---- Example inputs covering all three varieties ----
# Picked to show a range of sarcastic / non-sarcastic + positive / negative
EXAMPLES = [
    # Australian
    ["When I saw that Hungry Jack had a generous rating I nearly "
     "choked on my diet lemonade.", "Australian English"],
    ["Great coffee and the staff were really friendly. "
     "Would definitely come back.", "Australian English"],
    # British
    ["Great service, waited only 45 minutes for a glass of water.",
     "British English"],
    ["Lovely little cafe with fresh scones and perfect tea.",
     "British English"],
    # Indian
    ["Very good restaurant, the waiter brought cold soup and "
     "smiled about it.", "Indian English"],
    ["The food was very tasty and the ambiance was nice. "
     "Will visit again for sure.", "Indian English"],
]


def _build_theme():
    """
    Custom dark theme: charcoal background, slate greys, ice blue accents.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e8f4fd", c100="#d1e9fb", c200="#a3d3f7",
            c300="#75bdf3", c400="#47a7ef", c500="#1a91eb",
            c600="#1574bc", c700="#10578d", c800="#0a3a5e",
            c900="#051d2f", c950="#021019",
        ),
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=("Inter", "system-ui", "sans-serif"),
    ).set(
        body_background_fill="#0f1115",
        body_background_fill_dark="#0f1115",
        block_background_fill="#1a1d23",
        block_background_fill_dark="#1a1d23",
        block_border_color="#2a2d35",
        block_border_color_dark="#2a2d35",
        block_label_text_color="#b8bcc5",
        block_label_text_color_dark="#b8bcc5",
        block_title_text_color="#e2e5ea",
        block_title_text_color_dark="#e2e5ea",
        body_text_color="#d1d5db",
        body_text_color_dark="#d1d5db",
        button_primary_background_fill="#1a91eb",
        button_primary_background_fill_dark="#1a91eb",
        button_primary_background_fill_hover="#47a7ef",
        button_primary_background_fill_hover_dark="#47a7ef",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        input_background_fill="#12141a",
        input_background_fill_dark="#12141a",
        input_border_color="#2a2d35",
        input_border_color_dark="#2a2d35",
        border_color_primary="#1a91eb",
        border_color_primary_dark="#1a91eb",
        shadow_drop="0 4px 14px rgba(0,0,0,0.4)",
        shadow_drop_lg="0 8px 24px rgba(0,0,0,0.5)",
        block_radius="12px",
        button_large_radius="10px",
        input_radius="8px",
    )


# Extra CSS for transitions and polish that Gradio themes cant cover
_CUSTOM_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: auto;
}
.gr-button {
    transition: all 0.25s ease;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.gr-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(26,145,235,0.35);
}
.gr-box, .gr-panel {
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.gr-box:hover, .gr-panel:hover {
    border-color: #1a91eb !important;
    box-shadow: 0 0 0 1px rgba(26,145,235,0.15);
}
textarea, input[type="text"] {
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
textarea:focus, input[type="text"]:focus {
    border-color: #1a91eb !important;
    box-shadow: 0 0 0 3px rgba(26,145,235,0.12) !important;
}
.label-class-Sarcastic { color: #ef4444 !important; }
.label-class-Not { color: #22c55e !important; }
footer { opacity: 0.6; transition: opacity 0.3s; }
footer:hover { opacity: 1; }
#header-text {
    background: linear-gradient(135deg, #1a91eb 0%, #75bdf3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.example-row { transition: background 0.2s ease; border-radius: 8px; }
.example-row:hover { background: rgba(26,145,235,0.06); }
"""


def build_app():
    """
    Build and return the Gradio Blocks application.

    Layout:
        Header    -- gradient title + subtitle
        Main row  -- input panel (left), results panel (right)
        Examples  -- clickable demo inputs
        Footer    -- model info
    """
    variety_choices = [VARIETY_LABELS[v] + " English" for v in VARIETIES]
    theme = _build_theme()

    with gr.Blocks(
        title=GRADIO_TITLE,
        theme=theme,
        css=_CUSTOM_CSS,
    ) as app:

        # Header
        gr.HTML(
            '<div style="text-align:center; padding: 28px 0 8px;">'
            '<h1 id="header-text" style="font-size:2.2rem; margin:0;">'
            'BESSTIE</h1>'
            '<p style="color:#8b919a; font-size:0.95rem; margin-top:6px;">'
            'Sarcasm &amp; Sentiment Detection across English Varieties'
            '</p></div>'
        )

        with gr.Row(equal_height=True):
            # Left: input controls
            with gr.Column(scale=3, min_width=380):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Paste a review or sentence here...",
                    lines=5,
                    elem_id="main-input",
                )
                variety_input = gr.Dropdown(
                    choices=variety_choices,
                    value=variety_choices[0],
                    label="Variety",
                    info="Loads the matching LoRA adapter for sarcasm",
                )
                submit_btn = gr.Button(
                    "Analyse",
                    variant="primary",
                    size="lg",
                )

            # Right: output labels
            with gr.Column(scale=2, min_width=280):
                sarcasm_output = gr.Label(
                    label="Sarcasm (LoRA Adapter)",
                    num_top_classes=2,
                    elem_id="sarcasm-out",
                )
                sentiment_output = gr.Label(
                    label="Sentiment (RoBERTa)",
                    num_top_classes=2,
                    elem_id="sentiment-out",
                )

        # Event bindings
        submit_btn.click(
            fn=predict,
            inputs=[text_input, variety_input],
            outputs=[sarcasm_output, sentiment_output],
        )
        text_input.submit(
            fn=predict,
            inputs=[text_input, variety_input],
            outputs=[sarcasm_output, sentiment_output],
        )

        # Example inputs
        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_input, variety_input],
            label="Quick examples",
            examples_per_page=6,
        )

        # Footer
        gr.HTML(
            '<div style="text-align:center; padding:20px 0 10px; '
            'border-top:1px solid #2a2d35; margin-top:24px;">'
            '<span style="color:#6b7280; font-size:0.82rem;">'
            'COMM061 Group PG29 &middot; '
            'Sarcasm: per-variety LoRA on Qwen2.5-1.5B-Instruct &middot; '
            'Sentiment: fine-tuned RoBERTa-base'
            '</span></div>'
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
