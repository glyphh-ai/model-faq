"""
Custom encoder for the FAQ helpdesk model.

Exports:
  ENCODER_CONFIG — EncoderConfig with semantic + context layers
  encode_query(query) — converts NL question to a Concept for similarity search
  entry_to_record(entry) — converts a JSONL entry to an encodable record

Primary matching signal: bag-of-words on the question field. Shared words
between a user question ("forgot my password") and an FAQ entry ("how do I
reset my password") drive similarity — no verb/object decomposition needed.

IntentExtractor handles keyword extraction and provides domain/action signals
to boost category inference accuracy.
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
)
from glyphh.intent import get_extractor


# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    include_temporal=False,
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
                        ),
                        Role(
                            name="category",
                            similarity_weight=0.6,
                            lexicons=[
                                "account", "billing", "product",
                                "shipping", "returns", "technical", "general",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="content",
                    roles=[
                        Role(
                            name="question",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="answer",
                            similarity_weight=0.4,
                            text_encoding="bag_of_words",
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
                            text_encoding="bag_of_words",
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
    "account":   ["account", "login", "password", "sign in", "register", "profile",
                  "verify", "2fa", "mfa", "username", "email address", "two-factor"],
    "billing":   ["billing", "invoice", "charge", "payment", "subscription", "plan",
                  "upgrade", "downgrade", "refund", "credit card", "receipt", "price",
                  "cost", "fee", "charged", "billed", "money back"],
    "product":   ["feature", "setup", "configure", "integration", "api", "dashboard",
                  "settings", "tutorial", "get started", "how to use", "install",
                  "connect", "enable", "available"],
    "shipping":  ["shipping", "delivery", "track", "tracking", "ship", "arrive",
                  "package", "courier", "transit", "estimated", "international",
                  "deliver", "order", "package"],
    "returns":   ["return", "exchange", "warranty", "damaged", "broken", "wrong item",
                  "cancel order", "return policy", "send back", "send it back",
                  "wrong", "defective"],
    "technical": ["error", "bug", "crash", "not working", "broken", "fix", "issue",
                  "troubleshoot", "debug", "timeout", "500", "404", "slow",
                  "loading", "problem", "fails", "failing"],
    "general":   ["contact", "support", "business hours", "phone", "privacy policy",
                  "data", "gdpr", "hours", "talk to", "speak to", "human"],
}

# IntentExtractor domain → FAQ category shortcut
_DOMAIN_CATEGORY: dict[str, str] = {
    "payments": "billing",
    "tickets":  "technical",
}

# IntentExtractor action → FAQ category shortcut
_ACTION_CATEGORY: dict[str, str] = {
    "charge":      "billing",
    "refund":      "billing",
    "subscribe":   "billing",
    "cancel":      "billing",
    "track":       "shipping",
}


def _infer_category(text: str, domain: str = "", action: str = "") -> str:
    """Infer FAQ category from text + optional IntentExtractor signals.

    Text signals take priority. Domain/action shortcuts are used only as
    a fallback when the text contains no recognisable category signals.
    """
    lower = text.lower()
    best_cat, best_score = "general", 0
    for cat, signals in _CATEGORY_SIGNALS.items():
        score = sum(1 for s in signals if s in lower)
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_score > 0:
        return best_cat

    # No text signal found — use IntentExtractor shortcuts as fallback
    if domain in _DOMAIN_CATEGORY:
        return _DOMAIN_CATEGORY[domain]
    if action in _ACTION_CATEGORY:
        return _ACTION_CATEGORY[action]
    return best_cat


def _preprocess(text: str) -> str:
    """Lowercase and normalise punctuation for consistent BoW encoding."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# encode_query — NL question → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL question into a Concept-compatible dict."""
    cleaned = _preprocess(query)

    extractor = get_extractor()
    extracted = extractor.extract(cleaned)

    category = _infer_category(
        cleaned,
        domain=extracted.get("domain") or "",
        action=extracted.get("action") or "",
    )
    keywords = extracted.get("keywords") or cleaned

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "question_id": "",
            "category": category,
            "question": cleaned,
            "answer": "",
            "keywords": keywords,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL entry → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL entry to an encodable record with metadata."""
    question   = entry.get("question", "")
    question_id = entry.get("question_id", "")
    category   = entry.get("category", "")
    answer     = entry.get("answer", "")
    kw_list    = entry.get("keywords", [])
    kw_str     = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    # Auto-generate question_id if not provided
    if not question_id:
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:40]
        question_id = f"faq_{slug}"

    # Auto-infer category if not provided
    if not category:
        category = _infer_category(question)

    # Preprocess text for consistent BoW encoding
    question_clean = _preprocess(question)
    answer_clean   = _preprocess(answer)

    return {
        "concept_text": question,
        "attributes": {
            "question_id": question_id,
            "category":    category,
            "question":    question_clean,
            "answer":      answer_clean,
            "keywords":    kw_str,
        },
        "metadata": {
            "answer":            answer,
            "category":          category,
            "question_id":       question_id,
            "original_question": question,
        },
    }
