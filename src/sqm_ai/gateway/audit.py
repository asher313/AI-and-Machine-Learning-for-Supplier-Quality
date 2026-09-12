"""Append-only lifecycle events and optional authenticated encrypted payloads."""

import hashlib
import json
import os
from pathlib import Path
import uuid

import psycopg
from psycopg.types.json import Jsonb


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256(text):
    return (
        hashlib.sha256(text.encode()).hexdigest()
        if text is not None
        else None
    )


class AuditWriter:
    def __init__(self, dsn):
        self.dsn = dsn

    def write(self, row):
        # A separate connection/transaction cannot be rolled back by the caller's business transaction.
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO sqm.llm_audit_events(trace_id,event,record) VALUES (%s,%s,%s)",
                (row["trace_id"], row["event"], Jsonb(row)),
            )


class EncryptedContentStore:
    """Local teaching store. Production requires approved key/storage/access management."""

    def __init__(self, directory, key, *, key_id, allowed_levels):
        from cryptography.hazmat.primitives.ciphers.aead import (
            AESGCM,
        )

        if len(key) != 32 or not key_id:
            raise ValueError(
                "32-byte key and nonempty key ID required"
            )
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.aead, self.key_id = AESGCM(key), key_id
        self.allowed_levels = frozenset(allowed_levels)

    def put(self, trace_id, kind, text, *, classification):
        if classification not in self.allowed_levels:
            raise ValueError(
                "archive destination not approved for this classification"
            )
        blob_id = uuid.uuid4().hex
        aad = canonical(
            {
                "trace_id": trace_id,
                "kind": kind,
                "key_id": self.key_id,
                "classification": classification,
            }
        ).encode()
        nonce = os.urandom(12)
        ciphertext = self.aead.encrypt(nonce, text.encode(), aad)
        value = {
            "aad": aad.decode(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
        }
        path = self.path / (blob_id + ".json")
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        return blob_id

    def get(self, blob_id, *, trace_id, kind, classification):
        if len(blob_id) != 32 or any(
            c not in "0123456789abcdef" for c in blob_id
        ):
            raise ValueError("invalid blob ID")
        if classification not in self.allowed_levels:
            raise ValueError("archive classification not allowed")
        value = json.loads(
            (self.path / (blob_id + ".json")).read_text()
        )
        aad = canonical(
            {
                "trace_id": trace_id,
                "kind": kind,
                "key_id": self.key_id,
                "classification": classification,
            }
        ).encode()
        if value["aad"].encode() != aad:
            raise ValueError("archive context mismatch")
        return self.aead.decrypt(
            bytes.fromhex(value["nonce"]),
            bytes.fromhex(value["ciphertext"]),
            aad,
        ).decode()
