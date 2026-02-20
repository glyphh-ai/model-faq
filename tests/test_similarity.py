"""Test that raw user questions match the correct FAQ entries.

Encodes raw questions (no category, no answer) and compares against
the training FAQ data to verify the model returns the right answer.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, encode_query, entry_to_record

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept, SimilarityCalculator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="module")
def similarity():
    return SimilarityCalculator()


@pytest.fixture(scope="module")
def faq_glyphs(encoder):
    """Encode all FAQ entries into glyphs with their metadata."""
    faq_path = DATA_DIR / "faq.jsonl"
    glyphs = []
    with open(faq_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            record = entry_to_record(entry)
            concept = Concept(
                name=record["attributes"]["question_id"],
                attributes=record["attributes"],
            )
            glyph = encoder.encode(concept)
            glyphs.append((glyph, record["metadata"]))
    return glyphs


def _find_best_match(query_text, faq_glyphs, encoder, similarity):
    """Encode a raw question and find the best matching FAQ entry."""
    record = encode_query(query_text)
    query_glyph = encoder.encode(Concept(
        name=record["name"],
        attributes=record["attributes"],
    ))

    best_score = -1
    best_meta = None
    for faq_glyph, meta in faq_glyphs:
        result = similarity.compute(query_glyph, faq_glyph)
        if result.score > best_score:
            best_score = result.score
            best_meta = meta
    return best_meta, best_score


def test_password_question_matches_account(encoder, similarity, faq_glyphs, test_queries):
    """'I forgot my password' should match the password reset FAQ."""
    q = next(q for q in test_queries if "password" in q["question"])
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_billing_question_matches_billing(encoder, similarity, faq_glyphs, test_queries):
    """'I want to stop being charged' should match billing."""
    q = next(q for q in test_queries if "charged" in q["question"])
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_shipping_question_matches_shipping(encoder, similarity, faq_glyphs, test_queries):
    """'where is my package' should match shipping."""
    q = next(q for q in test_queries if "package" in q["question"])
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_damaged_item_matches_returns(encoder, similarity, faq_glyphs, test_queries):
    """'the thing I ordered arrived broken' should match returns."""
    q = next(q for q in test_queries if "broken" in q["question"])
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_slow_app_matches_technical(encoder, similarity, faq_glyphs, test_queries):
    """'everything is super slow today' should match technical."""
    q = next(q for q in test_queries if "slow" in q["question"])
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_contact_question_matches_general(encoder, similarity, faq_glyphs, test_queries):
    """'I need to talk to a real person' should match general/contact."""
    q = next(q for q in test_queries if "real person" in q["question"])
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_all_queries_have_positive_scores(encoder, similarity, faq_glyphs, test_queries):
    """Every test query should have a positive similarity score."""
    for q in test_queries:
        meta, score = _find_best_match(q["question"], faq_glyphs, encoder, similarity)
        assert score > 0, f"Query '{q['question']}' got zero similarity"
