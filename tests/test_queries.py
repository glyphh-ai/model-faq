"""Test that encode_query produces correct attributes from NL text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query


def test_account_query_infers_account():
    """Questions about passwords/login should infer category=account."""
    result = encode_query("I forgot my password")
    assert result["attributes"]["category"] == "account"


def test_billing_query_infers_billing():
    """Questions about charges/invoices should infer category=billing."""
    result = encode_query("why was I charged twice on my invoice")
    assert result["attributes"]["category"] == "billing"


def test_shipping_query_infers_shipping():
    """Questions about delivery/tracking should infer category=shipping."""
    result = encode_query("where is my delivery tracking number")
    assert result["attributes"]["category"] == "shipping"


def test_returns_query_infers_returns():
    """Questions about returns/refunds should infer category=returns."""
    result = encode_query("I want to return this and get a refund")
    assert result["attributes"]["category"] == "returns"


def test_technical_query_infers_technical():
    """Questions about errors/bugs should infer category=technical."""
    result = encode_query("I keep getting an error message")
    assert result["attributes"]["category"] == "technical"


def test_product_query_infers_product():
    """Questions about features/setup should infer category=product."""
    result = encode_query("how do I configure the integration")
    assert result["attributes"]["category"] == "product"


def test_query_has_empty_question_id():
    """Queries are not stored FAQs, so question_id should be empty."""
    result = encode_query("how do I get started")
    assert result["attributes"]["question_id"] == ""


def test_query_has_empty_answer():
    """Queries don't have answers — the model finds them."""
    result = encode_query("what is your return policy")
    assert result["attributes"]["answer"] == ""


def test_query_has_stable_name():
    """Same query text should produce the same concept name."""
    q = "how do I reset my password"
    r1 = encode_query(q)
    r2 = encode_query(q)
    assert r1["name"] == r2["name"]


def test_keywords_exclude_stop_words():
    """Keywords should filter out common stop words."""
    result = encode_query("how do I find the shipping status for my order")
    kw = result["attributes"]["keywords"]
    assert "how" not in kw.split()
    assert "do" not in kw.split()
    assert "the" not in kw.split()
    assert "for" not in kw.split()
