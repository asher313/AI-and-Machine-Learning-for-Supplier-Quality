# Build 4: reproducible assistant walkthrough

From the installed repository root, generate the fictional corpus and run the
software replay (if Chapter 17 already generated these files, reuse them):

```bash
python scripts/generate_retrieval_data.py
python scripts/index_corpus.py
python -m sqm_ai.assistant.run_build4
```

The replay reads `artifacts/retrieval/index.json` and writes
`artifacts/build4/answer.json`. Output files are protected against overwrite.
The generator, indexer, and replay make no paid API calls. They use 13 fictional
documents, 25 chunks, and labelled lexical-hash vectors, with no AS9100 text.
Program visibility, dates, and embedding space are filtered before retrieval.
The draft is a scripted extraction from one source; the support check compares
text, and question relevance is scripted true. A `verified` replay result
therefore demonstrates software composition only. It does not establish
neural retrieval, answer relevance, citation entailment, or production quality.

The real answer pipeline accepts already-authorized chunks, requests structured
claims, requires a check for every claim, and withholds failed drafts. It
separates absent-source refusals from verification failures. The direct API
adapter requires trusted `synthetic` or `approved_uncontrolled` classification.
Production identity, UI, source links, protected review storage, and feedback
persistence remain application integrations.

## Optional live evaluation

Install `uv sync --extra evaluation`. Ragas 0.4.3 requires the compatible
`langchain-community==0.3.31`; its unbounded dependency otherwise selects a
version missing a legacy import. `AnthropicScorer` implements Ragas' structured
LLM interface using the native Anthropic SDK, avoiding the default adapter's
obsolete sampling parameters. The shared adapter owns bounded retries and
logs token usage. The current scoring model is the registered standard tier;
no provider is selected implicitly. BGE embeddings load at the pinned revision
in `sqm_ai.retrieval.embed`.

Prepare a JSON list of saved, approved evaluation rows. Each contains `id`,
`question`, `answer`, `contexts` (a list of strings), `reference`, and
`data_classification`. References must be reviewed against the frozen corpus.
Then explicitly enable live requests:

```bash
python scripts/eval_generation.py --input data/eval/answers.json \
  --output artifacts/build4/ragas.json --run-live
```

This mode requires credentials, paid scoring requests, and local BGE weights.
Each row can require several generation and token-count requests; answer
relevancy uses generated questions and embedding comparisons. Missing/non-finite
metric values remain null. Preserve input/corpus/model/prompt/dependency
versions with the run. Do not use fictional replay responses as a quality gate.

Offline validation covers four answer-policy tests, two metric-contract tests,
and one real SDK/Ragas integration test using mocked HTTP only. A separate
PostgreSQL/pgvector test verifies atomic revision replacement, rollback on an
invalid insert, and exact as-of boundaries. The mock's perfect metric values
are scripted expectations, not model-quality measurements. Live scoring and
production assistant performance were not tested.

Primary references:

- [Ragas structured LLM interface](https://docs.ragas.io/en/stable/references/llms/)
- [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [Context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
