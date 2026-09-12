"""Bounded answer cache; semantic reuse requires an application verifier."""

import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CacheScope:
    # Trusted application constructs this AFTER authorizing the request.
    authorization_scope: str
    corpus_revision: str
    model_prompt_revision: str
    embedding_revision: str


class SemanticCache:
    def __init__(
        self,
        embedder,
        threshold=0.97,
        ttl=300,
        capacity=1000,
        *,
        verify=None,
        clock=time.monotonic,
    ):
        if not -1 <= threshold <= 1 or ttl <= 0 or capacity < 1:
            raise ValueError("invalid cache settings")
        self.embedder, self.threshold = embedder, threshold
        self.ttl, self.capacity, self.verify, self.clock = (
            ttl,
            capacity,
            verify,
            clock,
        )
        self.entries = []
        self.width = None

    def _embed(self, text):
        vector = np.asarray(
            self.embedder.encode(
                [text], normalize_embeddings=True
            )[0],
            dtype=np.float32,
        )
        if (
            vector.ndim != 1
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0
        ):
            raise ValueError("invalid embedding")
        if self.width is None:
            self.width = len(vector)
        if len(vector) != self.width:
            raise ValueError(
                "embedding dimension changed; use a new cache"
            )
        return vector / np.linalg.norm(vector)

    def _prune(self):
        now = self.clock()
        self.entries = [e for e in self.entries if e[0] > now]

    def get(self, query, *, scope):
        self._prune()
        eligible = [e for e in self.entries if e[1] == scope]
        for _, _, stored, _, answer in reversed(eligible):
            if stored == query:
                return deepcopy(answer)
        # Similarity alone never authorizes reuse of a factual answer.
        if self.verify is None or not eligible:
            return None
        vector = self._embed(query)
        ranked = sorted(
            eligible,
            key=lambda e: float(e[3] @ vector),
            reverse=True,
        )
        for _, _, stored, key, answer in ranked:
            if float(key @ vector) < self.threshold:
                break
            if self.verify(
                query, stored, deepcopy(answer), scope
            ):
                return deepcopy(answer)
        return None

    def put(self, query, value, *, scope):
        if not isinstance(scope, CacheScope) or not all(
            vars(scope).values()
        ):
            raise ValueError(
                "complete trusted cache scope required"
            )
        key = self._embed(query)
        self._prune()
        self.entries = [
            e
            for e in self.entries
            if not (e[1] == scope and e[2] == query)
        ]
        self.entries.append(
            (
                self.clock() + self.ttl,
                scope,
                query,
                key,
                deepcopy(value),
            )
        )
        self.entries = self.entries[-self.capacity :]
