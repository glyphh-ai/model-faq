"""Shared fixtures for FAQ model tests."""

import json
import sys
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

TESTS_DIR = Path(__file__).resolve().parent
CONCEPTS_PATH = TESTS_DIR / "test-concepts.json"


@pytest.fixture(scope="session")
def test_queries():
    """Load raw test queries from test-concepts.json."""
    with open(CONCEPTS_PATH) as f:
        return json.load(f)["queries"]


@pytest.fixture(scope="session")
def encoder_config():
    """Import and return the model's ENCODER_CONFIG."""
    from encoder import ENCODER_CONFIG
    return ENCODER_CONFIG
