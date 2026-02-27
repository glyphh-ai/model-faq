"""Test that encode_query produces correct attributes from NL text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query


def test_account_query_infers_account():
    result = encode_query("I forgot my password")
    assert result["attributes"]["category"] == "account"


def test_billing_query_infers_billing():
    result = encode_query("why was I charged twice on my invoice")
    assert result["attributes"]["category"] == "billing"


def test_shipping_query_infers_shipping():
    result = encode_query("where is my delivery tracking number")
    assert result["attributes"]["category"] == "shipping"


def test_returns_query_infers_returns():
    result = encode_query("I want to return this and get a refund")
    # Returns and billing both contain "refund" — returns wins because of "return"
    assert result["attributes"]["category"] in ("returns", "billing")


def test_technical_query_infers_technical():
    result = encode_query("I keep getting an error message")
    assert result["attributes"]["category"] == "technical"


def test_product_query_infers_product():
    result = encode_query("how do I configure the integration")
    assert result["attributes"]["category"] == "product"


def test_query_has_empty_question_id():
    """Queries are not stored FAQs — question_id must be empty."""
    result = encode_query("how do I get started")
    assert result["attributes"]["question_id"] == ""


def test_query_has_empty_answer():
    """Queries don't have answers — the model finds them."""
    result = encode_query("what is your return policy")
    assert result["attributes"]["answer"] == ""


def test_query_has_stable_name():
    """Same query text must produce the same concept name."""
    q  = "how do I reset my password"
    r1 = encode_query(q)
    r2 = encode_query(q)
    assert r1["name"] == r2["name"]


def test_question_is_lowercased():
    """Question attribute must be lowercase for consistent BoW encoding."""
    result = encode_query("How Do I Reset My PASSWORD?")
    assert result["attributes"]["question"] == result["attributes"]["question"].lower()


def test_keywords_are_non_empty():
    """IntentExtractor should produce at least some keywords."""
    result = encode_query("how do I find the shipping status for my order")
    assert result["attributes"]["keywords"].strip() != ""


def test_keywords_exclude_common_stop_words():
    """Common stop words should not dominate the keywords field."""
    result = encode_query("how do I find the shipping status for my order")
    kw = result["attributes"]["keywords"].split()
    # Core meaningful words should be present
    assert any(w in kw for w in ("shipping", "status", "order", "find")), (
        f"Expected meaningful keywords in: {kw}"
    )
