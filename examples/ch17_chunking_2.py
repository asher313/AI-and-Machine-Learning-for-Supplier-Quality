# Chapter 17 teaching listing. Supply the inputs described in the text.
from sqm_ai.retrieval.chunking import semantic_chunk

# Explicitly supply the embedding model from Section 17.2.
chunks = semantic_chunk(document, embedder, threshold=0.5)
