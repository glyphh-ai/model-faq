"""
Custom encoder for the FAQ helpdesk model.

Exports:
  ENCODER_CONFIG — EncoderConfig with semantic + context layers
  encode_query(query) — converts NL question to a Concept for similarity search
  entry_to_record(entry) — converts a JSONL entry to an encodable record

The model is intentionally simple: questions go in, best matching
answer comes out. No verb/object decomposition — just semantic matching.
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
    TemporalConfig,
)

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    temporal_source="auto",
    temporal_config=TemporalConfig(signal_type="auto"),
    layers=[
        Layer(
            name="semantic",
            similarity_weight=0.7,
            segments=[
                Segment(
                    name="identity",
                    roles=[
                        Role(
                            name="question_id",
                            similarity_weight=0.1,
                            key_part=True,
                            lexicons=["id", "question id", "faq id"],
                        ),
                        Role(
                            name="category",
                            similarity_weight=0.6,
                            lexicons=["category", "topic", "department", "type"],
                        ),
                    ],
                ),
                Segment(
                    name="content",
                    roles=[
                        Role(
                            name="question",
                            similarity_weight=1.0,
                            lexicons=["question", "query", "ask", "help"],
                        ),
                        Role(
                            name="answer",
                            similarity_weight=0.4,
                            lexicons=["answer", "response", "solution", "reply"],
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.3,
            segments=[
                Segment(
                    name="tags",
                    roles=[
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            lexicons=["keywords", "tags", "terms", "related"],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

_CATEGORY_SIGNALS = {
    "account": ["account", "login", "password", "sign in", "register", "profile",
                "username", "email", "verify", "two-factor", "2fa", "mfa"],
    "billing": ["billing", "invoice", "charge", "payment", "subscription", "plan",
                "upgrade", "downgrade", "refund", "credit card", "receipt", "price"],
    "product": ["feature", "how to", "how do", "use", "setup", "configure",
                "integration", "api", "dashboard", "settings", "tutorial"],
    "shipping": ["shipping", "delivery", "track", "tracking", "ship", "arrive",
                 "package", "courier", "estimated", "transit"],
    "returns": ["return", "exchange", "refund", "warranty", "damaged", "wrong item",
                "cancel order", "return policy"],
    "technical": ["error", "bug", "crash", "not working", "broken", "fix", "issue",
                  "troubleshoot", "debug", "timeout", "500", "404"],
    "general": ["hello", "hi", "thanks", "help", "contact", "hours", "phone"],
}

_STOP_WORDS = {
    "how", "do", "i", "a", "the", "to", "is", "what", "my", "an",
    "can", "does", "it", "in", "on", "for", "with", "me", "about",
    "are", "which", "who", "will", "their", "this", "that", "of",
    "please", "need", "want", "would", "like", "could", "should",
}


def _infer_category(text):
    """Infer FAQ category from question text."""
    lower = text.lower()
    best_cat = "general"
    best_score = 0
    for cat, signals in _CATEGORY_SIGNALS.items():
        score = sum(1 for s in signals if s in lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


# ---------------------------------------------------------------------------
# encode_query — NL question → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL question into a Concept-compatible dict."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    words = cleaned.split()

    category = _infer_category(query)
    keywords = " ".join(w for w in words if w not in _STOP_WORDS)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "question_id": "",
            "category": category,
            "question": query,
            "answer": "",
            "keywords": keywords,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL entry → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL entry to an encodable record with metadata."""
    question = entry.get("question", "")
    question_id = entry.get("question_id", "")
    category = entry.get("category", "")
    answer = entry.get("answer", "")
    kw_list = entry.get("keywords", [])
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    # Auto-generate question_id if not provided
    if not question_id:
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:40]
        question_id = f"faq_{slug}"

    # Auto-infer category if not provided
    if not category:
        category = _infer_category(question)

    return {
        "concept_text": question,
        "attributes": {
            "question_id": question_id,
            "category": category,
            "question": question,
            "answer": answer,
            "keywords": kw_str,
        },
        "metadata": {
            "answer": answer,
            "category": category,
            "question_id": question_id,
            "original_question": question,
        },
    }
