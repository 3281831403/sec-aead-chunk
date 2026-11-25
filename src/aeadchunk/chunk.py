"""Chunked authenticated encryption with message-level structure checks.

A message is split into fixed-size chunks.  Every chunk carries the metadata
needed to place it in the message, and the metadata is meant to be covered by
the authentication tag so a chunk cannot be moved between messages or
positions unnoticed.
"""

from __future__ import annotations

import base64
import json
import os
import struct
from collections.abc import Mapping
from typing import Any

import hashlib
import hmac

CHUNK_SIZE = 32
_NONCE_SIZE = 12
_KEY_SIZE = 32



class _InvalidTag(Exception):
    """Authentication tag mismatch."""


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(b"ks" + key + nonce + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(out[:length])


def _aead_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    stream = _keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(key, b"tag" + aad + nonce + ciphertext, hashlib.sha256).digest()[:16]
    return ciphertext + tag


def _aead_decrypt(key: bytes, nonce: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < 16:
        raise _InvalidTag("short ciphertext")
    ciphertext, tag = blob[:-16], blob[-16:]
    expect = hmac.new(key, b"tag" + aad + nonce + ciphertext, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(tag, expect):
        raise _InvalidTag("tag mismatch")
    stream = _keystream(key, nonce, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


def _hkdf(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


class AEADChunkError(Exception):
    """Raised when a chunk or a whole message cannot be authenticated."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"unsupported context value: {type(value).__name__}")


def _context_bytes(context: Any) -> bytes:
    if isinstance(context, bytes):
        return b"b:" + context
    if isinstance(context, bytearray):
        return b"b:" + bytes(context)
    if isinstance(context, str):
        return b"s:" + context.encode("utf-8")
    if isinstance(context, Mapping):
        try:
            payload = json.dumps(
                context,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=_json_default,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("context mapping must be canonicalizable") from exc
        return b"j:" + payload.encode("utf-8")
    raise TypeError("context must be bytes, str, or a mapping")


def _validate_position(index: int, total: int, final: bool) -> None:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if not isinstance(final, bool):
        raise TypeError("final must be a boolean")
    if total < 1:
        raise ValueError("total must be at least one")
    if index < 0 or index >= total:
        raise ValueError("index must be inside the message")


def derive_key(master: bytes, context: Any) -> bytes:
    """Derive the 32-byte chunk key for a message context."""

    if not isinstance(master, (bytes, bytearray)):
        raise TypeError("master must be bytes")
    material = b"aeadchunk/key/v1"
    return _hkdf(bytes(master), b"aeadchunk/kdf/salt/v1", material)


def build_aad(index: int, total: int, final: bool, context: Any) -> bytes:
    """Build the associated data for one chunk of a message."""

    _validate_position(index, total, final)
    return b"aeadchunk/aad/v1" + _context_bytes(context)


def seal_chunk(
    key: bytes,
    index: int,
    total: int,
    final: bool,
    context: Any,
    plaintext: bytes,
) -> dict[str, Any]:
    """Encrypt one chunk and attach its placement metadata."""

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("key must be 32 bytes")
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("plaintext must be bytes")
    _validate_position(index, total, final)
    nonce = os.urandom(_NONCE_SIZE)
    aad = build_aad(index, total, final, context)
    ciphertext = _aead_encrypt(bytes(key), nonce, bytes(plaintext), aad)
    return {
        "index": index,
        "total": total,
        "final": final,
        "nonce": nonce,
        "ciphertext": ciphertext,
    }


def open_chunk(
    key: bytes,
    chunk: Mapping[str, Any],
    expected_index: int,
    total: int,
    context: Any,
) -> bytes:
    """Decrypt one chunk and return its plaintext."""

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("key must be 32 bytes")
    if not isinstance(chunk, Mapping):
        raise TypeError("chunk must be a mapping")
    nonce = chunk["nonce"]
    ciphertext = chunk["ciphertext"]
    index = chunk["index"]
    stored_total = chunk["total"]
    final = chunk["final"]
    aad = build_aad(index, stored_total, final, context)
    return _aead_decrypt(bytes(key), nonce, ciphertext, aad)


def _seal_message_with_key(
    key: bytes, context: Any, data: bytes
) -> list[dict[str, Any]]:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    payload = bytes(data)
    total = max(1, (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE)
    chunks: list[dict[str, Any]] = []
    for index in range(total):
        start = index * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunks.append(
            seal_chunk(
                key,
                index,
                total,
                index == total - 1,
                context,
                payload[start:end],
            )
        )
    return chunks


def _open_message_with_key(
    key: bytes, context: Any, chunks: list[Mapping[str, Any]]
) -> bytes:
    if not isinstance(chunks, (list, tuple)):
        raise TypeError("chunks must be a sequence")
    if not chunks:
        return b""
    first = chunks[0]
    declared_total = first.get("total") if isinstance(first, Mapping) else None
    parts: list[bytes] = []
    saw_final = False
    for position, chunk in enumerate(chunks):
        parts.append(open_chunk(key, chunk, position, declared_total, context))
        if isinstance(chunk, Mapping) and bool(chunk.get("final")):
            saw_final = True
    if not saw_final:
        return b""
    return b"".join(parts)


def seal_message(master: bytes, context: Any, data: bytes) -> list[dict[str, Any]]:
    """Encrypt a whole message and return its chunks in order."""

    key = derive_key(master, context)
    return _seal_message_with_key(key, context, data)


def open_message(
    master: bytes, context: Any, chunks: list[Mapping[str, Any]]
) -> bytes:
    """Authenticate and decrypt all chunks of one message."""

    key = derive_key(master, context)
    return _open_message_with_key(key, context, chunks)


def validate_chunks(chunks: list[Mapping[str, Any]]) -> list[str]:
    """Return human-readable structural problems found in a chunk list."""

    issues: list[str] = []
    if not isinstance(chunks, (list, tuple)):
        return ["chunks must be a sequence"]
    if not chunks:
        return ["message has no chunks"]
    first = chunks[0]
    declared_total = first.get("total") if isinstance(first, Mapping) else None
    last_position = len(chunks) - 1
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            issues.append(f"chunk {position}: not a mapping")
            continue
        if "index" not in chunk:
            issues.append(f"chunk {position}: missing index")
            index = None
        else:
            index = chunk["index"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index != position
            ):
                issues.append(f"chunk {position}: index does not match position")
        if "total" not in chunk:
            issues.append(f"chunk {position}: missing total")
        if "final" not in chunk:
            issues.append(f"chunk {position}: missing final marker")
        else:
            final = bool(chunk["final"])
            if isinstance(declared_total, int) and not isinstance(
                declared_total, bool
            ):
                expected_final = (
                    position == last_position and index == declared_total - 1
                )
            else:
                expected_final = position == last_position
            if final is not expected_final:
                issues.append(f"chunk {position}: final marker misplaced")
    return issues


def aead_batch(messages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Seal and immediately open several independent messages."""

    if not isinstance(messages, Mapping):
        raise TypeError("messages must be a mapping")
    results: dict[str, Any] = {}
    active_key: bytes | None = None
    for name, spec in messages.items():
        if not isinstance(spec, Mapping):
            raise TypeError("each message must be a mapping")
        master = spec["master"]
        context = spec["context"]
        data = spec["data"]
        if active_key is None:
            active_key = derive_key(master, context)
        chunks = _seal_message_with_key(active_key, context, data)
        plaintext = _open_message_with_key(active_key, context, chunks)
        results[name] = {
            "context": context,
            "chunks": chunks,
            "plaintext": plaintext,
        }
    return results