# scripts/eval_generation.py
# TODO(book): condensed in Chapter 18 — questions,
# answers, contexts and expected are built by the caller
# from the 200-question golden set (18.6). Complete
# before production use.
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, context_precision, context_recall,
    faithfulness,
)

data = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,        # list of lists of chunk text
    "ground_truth": expected,
})

report = evaluate(
    data,
    metrics=[
        faithfulness,        # is the answer only from context?
        answer_relevancy,    # does it answer the question?
        context_precision,   # were the chunks relevant?
        context_recall,      # was everything needed retrieved?
    ],
)
print(report)
