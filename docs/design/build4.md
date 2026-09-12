# DESIGN RECORD — Evidence-based standards assistant

Owner: teaching implementation; fictional operational owners are Asher, Lena and Ravi.
Date: 2026-09-11. Status: synthetic teaching prototype; no company deployment approval.

## 1. Clarify

Retrieve authorized, date-appropriate source evidence and produce reviewable cited answers.

## 2. Scope

The companion implements the executable learning workflow described below. No guarantee that a valid citation or model checker prevents every hallucination; no distribution of unlicensed standards text.

## 3. Architecture

Versioned authorized corpus → scoped hybrid retrieval → optional neural reranking → structured claims/citations → coverage/support/relevance checks → answer or withheld/refusal result.

## 4. Deep dive

Retrieval and parent expansion must share access/date/embedding-space constraints. Citation checks cover each claim, not only each marker. Valid time alone does not reconstruct historical knowledge or recorded-time snapshots.

## 5. Tradeoffs, risks and validation

Synthetic corpus/retrieval/answer replay and PostgreSQL authorization/version transactions verified. Native Ragas judge adapter tested with HTTP mocks; no measured live answer quality is claimed. See RETRIEVAL_RUN.md and BUILD4_RUN.md.

Synthetic fixtures are designed to reproduce documented descriptive contracts. They do not establish real-world performance, causal effectiveness or legal compliance. API/cloud integrations require explicit operational approval and evaluation.

## Ramp-up

Provide licensed/authorized documents and independently labeled evaluation cases; prespecify answerable/refusal criteria, monitor human review and freeze the evaluation corpus/model configuration.

Assign a real accountable owner, dates and acceptance evidence before operational adoption. The fictional names in the narrative are not signatures.

## Changed since

2026-09-11: Recorded the reviewed implementation and verification limits. Preserve earlier decisions in version control and update the current design when the implementation changes.
