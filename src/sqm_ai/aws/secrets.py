"""Bounded TTL caching, explicit refresh and no secret retrieval on import."""

from collections import OrderedDict
from copy import deepcopy
import json
import threading
import time


class SecretCache:
    def __init__(
        self,
        client,
        *,
        ttl_seconds=300,
        max_entries=8,
        clock=time.monotonic,
    ):
        if not 0 < ttl_seconds <= 3600 or max_entries < 1:
            raise ValueError(
                "bounded positive TTL and capacity required"
            )
        self.client, self.ttl, self.capacity, self.clock = (
            client,
            ttl_seconds,
            max_entries,
            clock,
        )
        self.values = OrderedDict()
        self.lock = threading.Lock()

    def get_secret(self, name, *, force_refresh=False):
        with self.lock:
            cached = self.values.get(name)
            if (
                cached
                and not force_refresh
                and self.clock() < cached[0]
            ):
                self.values.move_to_end(name)
                return deepcopy(cached[1])
            # On refresh failure, do not fall back to an expired credential.
            self.values.pop(name, None)
            response = self.client.get_secret_value(
                SecretId=name, VersionStage="AWSCURRENT"
            )
            value = json.loads(response["SecretString"])
            if not isinstance(value, dict):
                raise ValueError("JSON object secret required")
            self.values[name] = (self.clock() + self.ttl, value)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
            return deepcopy(value)
