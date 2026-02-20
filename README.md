# FAQ Helpdesk

Domain-agnostic FAQ model for helpdesk and knowledge base agents. Matches user questions to answers via HDC similarity. Replace the sample data with your own Q&A pairs.

## How It Works

You load your FAQ entries (question + answer + category). Users ask questions in natural language. The model encodes the question, compares it against all stored FAQ entries by meaning (not keywords), and returns the best matching answer with its category and confidence score.

## Model Structure

```
faq/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, auto_load_concepts, test config
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── build.py               # package model into .glyphh file
├── tests.py               # test runner entry point
├── data/
│   └── faq.jsonl          # FAQ entries (training data)
├── tests/
│   ├── test-concepts.json # sample questions for testing (no answers)
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, role encoding
│   ├── test_similarity.py # answer matching correctness
│   └── test_queries.py    # category inference from NL
└── README.md
```

## Roles

| Role | Type | Description |
|------|------|-------------|
| question_id | text (key_part) | Stable FAQ entry identifier |
| category | categorical | account, billing, product, shipping, returns, technical, general |
| question | text | The FAQ question (highest similarity weight) |
| answer | text | The response to return |
| keywords | text | Boost terms for matching |

## Customizing

Replace `data/faq.jsonl` with your own Q&A pairs. Each line:

```json
{
  "question_id": "acct_reset_password",
  "category": "account",
  "question": "how do I reset my password",
  "answer": "Go to Settings > Security > Reset Password.",
  "keywords": ["password", "reset", "forgot"]
}
```

`question_id` and `category` are auto-generated if omitted — the encoder infers category from the question text.

## Testing

Run the test suite before deploying:

```bash
# Via CLI
glyphh model test ./faq
glyphh model test ./faq -v

# Or directly
cd faq/
python tests.py
```

The test suite uses `tests/test-concepts.json` — 10 raw user questions with no answers. Tests encode these questions, compare against the FAQ data, and verify the model returns the correct category and answer.

## Query Examples

```bash
glyphh query "I forgot my password and can't get in"
glyphh query "where is my package"
glyphh query "I want to return this item"
glyphh query "everything is really slow today"
```
