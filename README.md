# FAQ Helpdesk

Domain-agnostic FAQ model for helpdesk and knowledge base agents. Matches user questions to answers via HDC semantic similarity. Replace the sample data with your own Q&A pairs — no retraining, no embeddings API.

## Getting Started

```bash
# Install glyphh
pip install glyphh

# Clone and run locally
git clone https://github.com/glyphh-ai/model-faq.git
cd model-faq

# Start the local dev server (no account needed)
glyphh dev . -d

# Query it
glyphh chat "I forgot my password"
glyphh chat "where is my order"
glyphh chat "can I get a refund"
```

The server runs at `http://localhost:8002`. Open the Chat URL shown in the startup output to use the browser UI.

## How It Works

User question → IntentExtractor (keywords + category) → HDC encode (10,000d) → cosine similarity against stored FAQ entries → best matching answer + category + confidence.

Primary matching signal: **bag-of-words on the question text**. Shared words between a user question ("forgot my password") and an FAQ entry ("how do I reset my password") drive similarity — no LLM, no embeddings API call.

## Model Structure

```
faq/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── build.py               # package model into .glyphh file
├── data/
│   └── faq.jsonl          # FAQ entries (training data)
├── tests/
│   ├── test-concepts.json # 10 sample questions for testing (no answers)
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, BoW encoding, entry_to_record
│   ├── test_similarity.py # end-to-end match accuracy tests
│   └── test_queries.py    # encode_query unit tests
└── benchmark/
    ├── run.py             # benchmark runner (accuracy, latency, category breakdown)
    └── queries.json       # 32 queries across 4 difficulty categories
```

## Encoder

Two-layer HDC with 10,000 dimensions:

| Layer | Weight | Segment | Role | Encoding | Weight |
|-------|--------|---------|------|----------|--------|
| semantic | 0.7 | identity | question_id | symbolic (key_part) | 0.1 |
| | | | category | lexicon (7 values) | 0.6 |
| | | content | question | bag-of-words | 1.0 |
| | | | answer | bag-of-words | 0.4 |
| context | 0.3 | tags | keywords | bag-of-words | 0.8 |

Category lexicons: `account`, `billing`, `product`, `shipping`, `returns`, `technical`, `general`

## FAQ Data Format

Replace `data/faq.jsonl` with your own Q&A pairs. Each line:

```json
{
  "question_id": "acct_reset_password",
  "category": "account",
  "question": "how do I reset my password",
  "answer": "Go to Settings > Security > Reset Password. Enter your email to receive a reset link.",
  "keywords": ["password", "reset", "forgot", "locked out"]
}
```

`question_id` and `category` are auto-generated if omitted — the encoder infers category from the question text via IntentExtractor + keyword signals.

**keywords** should include synonyms, abbreviations, and related terms that users might phrase the question with but that aren't in the question itself.

## Testing

```bash
# Via CLI
glyphh model test ./faq
glyphh model test ./faq -v
glyphh model test ./faq -k similarity

# Directly with pytest
cd faq/
pytest tests/ -v
```

The test suite encodes 10 raw user questions (no answers) and verifies the model returns the correct category and `question_id` for each.

## Benchmark

```bash
# Run accuracy benchmark (no API key needed)
python benchmark/run.py

# Adjust confidence threshold
python benchmark/run.py --threshold 0.45

# Save raw results
python benchmark/run.py --output benchmark/results/
```

32 queries across 4 difficulty categories:

| Category | n | Description |
|----------|---|-------------|
| `clear` | 10 | Unambiguous single-category questions |
| `near_collision` | 10 | Questions that overlap multiple categories |
| `adversarial` | 7 | Informal phrasing, abbreviations, unusual wording |
| `open_set` | 5 | Out-of-scope questions (should abstain) |
