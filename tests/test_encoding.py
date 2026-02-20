"""Test that the encoder config is valid and roles encode correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, entry_to_record


def test_config_has_required_fields(encoder_config):
    """EncoderConfig must have dimension, seed, and layers."""
    assert encoder_config.dimension > 0
    assert encoder_config.seed >= 0
    assert len(encoder_config.layers) >= 1


def test_semantic_layer_exists(encoder_config):
    """Must have a semantic layer with identity and content segments."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "semantic" in layer_names

    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    seg_names = [s.name for s in semantic.segments]
    assert "identity" in seg_names
    assert "content" in seg_names


def test_context_layer_exists(encoder_config):
    """Must have a context layer with keywords."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "context" in layer_names

    context = next(l for l in encoder_config.layers if l.name == "context")
    roles = context.segments[0].roles
    role_names = [r.name for r in roles]
    assert "keywords" in role_names


def test_question_id_is_key_part(encoder_config):
    """question_id must be the key_part role."""
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    identity = next(s for s in semantic.segments if s.name == "identity")
    qid_role = next(r for r in identity.roles if r.name == "question_id")
    assert qid_role.key_part is True


def test_question_role_has_highest_weight(encoder_config):
    """The question role should have the highest similarity weight."""
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    content = next(s for s in semantic.segments if s.name == "content")
    q_role = next(r for r in content.roles if r.name == "question")
    a_role = next(r for r in content.roles if r.name == "answer")
    assert q_role.similarity_weight > a_role.similarity_weight


def test_entry_to_record_produces_all_attributes():
    """entry_to_record must produce all required attributes."""
    entry = {
        "question_id": "test_q",
        "category": "billing",
        "question": "how much does it cost",
        "answer": "Check the pricing page.",
        "keywords": ["pricing", "cost"],
    }
    record = entry_to_record(entry)
    attrs = record["attributes"]

    for key in ["question_id", "category", "question", "answer", "keywords"]:
        assert key in attrs, f"Missing attribute: {key}"


def test_entry_to_record_auto_generates_id():
    """question_id should be auto-generated if not provided."""
    entry = {
        "question": "what is your refund policy",
        "answer": "30 day refund window.",
        "category": "returns",
    }
    record = entry_to_record(entry)
    assert record["attributes"]["question_id"].startswith("faq_")


def test_entry_to_record_auto_infers_category():
    """Category should be inferred if not provided."""
    entry = {
        "question": "how do I reset my password",
        "answer": "Go to settings.",
    }
    record = entry_to_record(entry)
    assert record["attributes"]["category"] == "account"


def test_test_queries_have_no_answers(test_queries):
    """Test data should only contain raw questions — no answers."""
    for q in test_queries:
        assert "answer" not in q, f"Query has answer — test data should be raw"
        assert "question" in q
