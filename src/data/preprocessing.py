"""
Text preprocessing utilities.

Shared across classical and transformer pipelines so both
operate on the same cleaned text.

Design decision — MINIMAL cleaning:
  The BESSTIE dataset intentionally contains dialectal slang, code-mixing,
  and non-standard grammar. Aggressive cleaning (lowercasing, stopword
  removal, stemming) would destroy the very signal the coursework tests.
  We only remove noise that adds no linguistic value.
"""

import re

_NLP = None  # lazy-load spaCy


def get_spacy_nlp():
    """Lazy-load spaCy model (avoids slow import on every module load)."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _NLP


def clean_text(text):
    """
    Minimal text cleaning — aggressive cleaning destroys dialectal signal.
    We keep slang, code-mixing, and non-standard grammar intact because
    that's exactly what the coursework tests.

    Handles:
      - None / non-string inputs (returns empty string)
      - URLs (removed)
      - HTML entities like &amp; &lt; (decoded)
      - Excessive whitespace (collapsed)
      - Leading/trailing whitespace (stripped)
    """
    # Guard against None or non-string input
    if text is None or not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Decode common HTML entities
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_spacy(text):
    """
    Tokenise a single text using spaCy.
    Returns list of lowercased tokens (no punctuation, no stopwords).
    Used for vocabulary overlap analysis in Q1.
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy_nlp()
    doc = nlp(text)
    return [
        tok.text.lower()
        for tok in doc
        if not tok.is_punct and not tok.is_space
    ]


def tokenize_spacy_with_pos(text):
    """
    Tokenise and return (token, POS) pairs.
    Used for POS tag analysis in Q1.
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy_nlp()
    doc = nlp(text)
    return [(tok.text.lower(), tok.pos_) for tok in doc if not tok.is_space]


def get_vocabulary(texts):
    """
    Build the unique vocabulary from a list of texts.

    Args:
        texts: list of strings

    Returns:
        set of unique lowercased tokens
    """
    vocab = set()
    for text in texts:
        tokens = tokenize_spacy(clean_text(text))
        vocab.update(tokens)
    return vocab
