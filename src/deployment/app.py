"""
Q5.1: Deployment Endpoint (15 marks)
Owner: M4

Gradio app with variety selector. Backend loads the correct LoRA adapter
per user selection. Adapters hosted on HuggingFace.

Dependencies:
  - src.config (VARIETIES, VARIETY_LABELS, LLM_MODEL, LLM_MAX_LENGTH,
                HF_ADAPTER_PATHS, GRADIO_TITLE, GRADIO_DESCRIPTION)
  - peft (PeftModel)

NOTE: This is a Week 3-4 task. Do not implement until LoRA adapters are trained.
"""


def predict(text, variety):
    """
    Run inference on user input.

    Args:
        text: user input text
        variety: selected English variety (e.g. "en-UK")

    Returns:
        dict mapping label names to confidence scores
    """
    raise NotImplementedError("M4: Implement after LoRA adapters are ready")


def build_app():
    """
    Build the Gradio interface.

    Usage:
        from src.deployment.app import build_app
        app = build_app()
        app.launch()
    """
    raise NotImplementedError("M4: Implement Gradio app in Week 3-4")
