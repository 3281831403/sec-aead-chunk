"""Chunked authenticated encryption with message-level structure checks.

A message is split into fixed-size chunks.  Every chunk is bound, through the
AEAD associated data, to:

* its own chunk index and the total number of chunks in the message;
* an explicit final-chunk marker;
* a per-message random identifier shared by all chunks of the message;
* the full canonicalized message context.

As a result a chunk cannot be moved to another position, duplicated, spliced
from another message or truncated without invalidating its authentication
tag.  Every failure while opening untrusted chunk data is reported as the
single :class:`AEADChunkError` exception type.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
from collections.abc import Mapping
from typing import Any

CHUNK_SIZE = 32
_NONCE_SIZE = 12
_KEY_SIZE = 32
_TAG_SIZE = 16

_AAD_DOMAIN = b"aeadchunk/aad/v2"
_AAD_DOMAIN_MID = b"aeadchunk/aad/v2/mid"
_KEY_DOMAIN = b"aeadchunk/key/v2"
_KDF_SALT = b"aeadchunk/kdf/salt/v1"
_MESSAGE_ID_SIZE = 16


class _InvalidTag(Exception):
    """Internal signal for an authentication tag mismatch."""


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
    tag = hmac.new(key, b"tag" + aad + nonce + ciphertext, hashlib.sha256).digest()[:_TAG_SIZE]
    return ciphertext + tag


def _aead_decrypt(key: bytes, nonce: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < _TAG_SIZE:
        raise _InvalidTag("short ciphertext")
    ciphertext, tag = blob[:-_TAG_SIZE], blob[-_TAG_SIZE:]
    expect = hmac.new(key, b"tag" + aad + nonce + ciphertext, hashlib.sha256).digest()[:_TAG_SIZE]
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
    """Derive the 32-byte chunk key for a message context.

    The full canonicalized context is mixed into the HKDF info so different
    contexts always derive independent keys.
    """

    if not isinstance(master, (bytes, bytearray)):
        raise TypeError("master must be bytes")
    info = _KEY_DOMAIN + _context_bytes(context)
    return _hkdf(bytes(master), _KDF_SALT, info)


def _validate_message_id(message_id: Any) -> None:
    if message_id is not None and (
        not isinstance(message_id, bytes) or len(message_id) != _MESSAGE_ID_SIZE
    ):
        raise TypeError(f"message_id must be {_MESSAGE_ID_SIZE} bytes or None")


def build_aad(
    index: int,
    total: int,
    final: bool,
    context: Any,
    message_id: bytes | None = None,
) -> bytes:
    """Build the associated data for one chunk of a message.

    The AAD binds the chunk index, the total chunk count, the final marker,
    the per-message unique identifier (when the chunk belongs to a sealed
    message) and every field of the context.  Fixed-width struct headers and
    distinct domain prefixes make the encoding unambiguous and prevent
    field-confusion and cross-form collisions.
    """

    _validate_position(index, total, final)
    _validate_message_id(message_id)
    header = struct.pack(">QQB", index, total, 1 if final else 0)
    if message_id is None:
        return _AAD_DOMAIN + header + _context_bytes(context)
    return _AAD_DOMAIN_MID + header + message_id + _context_bytes(context)


def seal_chunk(
    key: bytes,
    index: int,
    total: int,
    final: bool,
    context: Any,
    plaintext: bytes,
    message_id: bytes | None = None,
) -> dict[str, Any]:
    """Encrypt one chunk and attach its placement metadata.

    When *message_id* is given it identifies the whole message and is bound
    into the tag, so chunks cannot be moved between messages even when they
    share a key, context and position.
    """

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("key must be 32 bytes")
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("plaintext must be bytes")
    _validate_message_id(message_id)
    _validate_position(index, total, final)
    # A fresh random nonce per chunk: encrypting the same plaintext twice
    # yields different ciphertexts while both remain decryptable.
    nonce = os.urandom(_NONCE_SIZE)
    aad = build_aad(index, total, final, context, message_id)
    ciphertext = _aead_encrypt(bytes(key), nonce, bytes(plaintext), aad)
    chunk = {
        "index": index,
        "total": total,
        "final": final,
        "nonce": nonce,
        "ciphertext": ciphertext,
    }
    if message_id is not None:
        chunk["message_id"] = message_id
    return chunk


def open_chunk(
    key: bytes,
    chunk: Mapping[str, Any],
    expected_index: int,
    total: int,
    context: Any,
    message_id: bytes | None = None,
) -> bytes:
    """Decrypt and authenticate one chunk at its expected position.

    The chunk's stored index/total/final must exactly match the position at
    which it is presented and, when present, its message identifier must
    match *message_id*.  Any malformed structure or authentication failure
    raises :class:`AEADChunkError` and nothing else.
    """

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("key must be 32 bytes")
    if isinstance(expected_index, bool) or not isinstance(expected_index, int):
        raise TypeError("expected_index must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if total < 1 or expected_index < 0 or expected_index >= total:
        raise ValueError("expected_index must be inside the message")
    _validate_message_id(message_id)

    expected_final = expected_index == total - 1
    try:
        if not isinstance(chunk, Mapping):
            raise AEADChunkError("malformed chunk")
        nonce = chunk["nonce"]
        ciphertext = chunk["ciphertext"]
        index = chunk["index"]
        stored_total = chunk["total"]
        final = chunk["final"]
        stored_message_id = chunk.get("message_id")

        if isinstance(index, bool) or not isinstance(index, int):
            raise AEADChunkError("malformed chunk index")
        if isinstance(stored_total, bool) or not isinstance(stored_total, int):
            raise AEADChunkError("malformed chunk total")
        if not isinstance(final, bool):
            raise AEADChunkError("malformed final marker")
        if not isinstance(nonce, bytes) or len(nonce) != _NONCE_SIZE:
            raise AEADChunkError("malformed nonce")
        if not isinstance(ciphertext, bytes) or len(ciphertext) < _TAG_SIZE:
            raise AEADChunkError("malformed ciphertext")
        if stored_message_id is not None and (
            not isinstance(stored_message_id, bytes)
            or len(stored_message_id) != _MESSAGE_ID_SIZE
        ):
            raise AEADChunkError("malformed message identifier")

        # Bind the chunk to its exact place in the message.
        if index != expected_index:
            raise AEADChunkError("chunk index does not match position")
        if stored_total != total:
            raise AEADChunkError("chunk total does not match message")
        if final is not expected_final:
            raise AEADChunkError("final marker misplaced")
        # Bind the chunk to the exact message it was sealed in.
        if stored_message_id != message_id:
            raise AEADChunkError("message identifier mismatch")

        aad = build_aad(expected_index, total, expected_final, context, message_id)
        return _aead_decrypt(bytes(key), nonce, ciphertext, aad)
    except _InvalidTag as exc:
        raise AEADChunkError("chunk authentication failed") from exc
    except AEADChunkError:
        raise
    except Exception as exc:  # pragma: no cover - defensive normalization
        raise AEADChunkError("malformed chunk") from exc


def _seal_message_with_key(
    key: bytes, context: Any, data: bytes
) -> list[dict[str, Any]]:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    payload = bytes(data)
    total = max(1, (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE)
    # One fresh random identifier per sealed message: every chunk is bound to
    # it, so chunks from two otherwise identical messages cannot be spliced.
    message_id = os.urandom(_MESSAGE_ID_SIZE)
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
                message_id,
            )
        )
    return chunks


def _open_message_with_key(
    key: bytes, context: Any, chunks: list[Mapping[str, Any]]
) -> bytes:
    if not isinstance(chunks, (list, tuple)):
        raise AEADChunkError("chunks must be a sequence")
    if not chunks:
        # An empty plaintext still seals to one (empty) chunk, so zero chunks
        # is always a truncated/forged message.
        raise AEADChunkError("message has no chunks")

    first = chunks[0]
    if not isinstance(first, Mapping):
        raise AEADChunkError("malformed chunk")
    declared_total = first.get("total")
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total < 1
    ):
        raise AEADChunkError("malformed message total")

    # The number of presented chunks must equal the authenticated total:
    # this rejects both truncation and extension/splicing.
    if len(chunks) != declared_total:
        raise AEADChunkError("chunk count does not match declared total")

    # The message identifier must be present, well-formed and identical on
    # every chunk before any tag is checked.
    message_id = first.get("message_id")
    if not isinstance(message_id, bytes) or len(message_id) != _MESSAGE_ID_SIZE:
        raise AEADChunkError("malformed message identifier")

    parts: list[bytes] = []
    for position, chunk in enumerate(chunks):
        # open_chunk enforces index == position, total consistency, the
        # placement of the final marker and the shared message identifier,
        # so reordering, duplication and cross-message splicing fail here
        # before any plaintext is released.
        if not isinstance(chunk, Mapping) or chunk.get("message_id") != message_id:
            raise AEADChunkError("message identifier mismatch")
        parts.append(
            open_chunk(key, chunk, position, declared_total, context, message_id)
        )
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

    total_count = len(chunks)
    message_ids: set[bytes] = set()
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            issues.append(f"chunk {position}: not a mapping")
            continue

        if "index" not in chunk:
            issues.append(f"chunk {position}: missing index")
        else:
            index = chunk["index"]
            if isinstance(index, bool) or not isinstance(index, int):
                issues.append(f"chunk {position}: index must be an integer")
            elif index != position:
                issues.append(f"chunk {position}: index does not match position")

        if "total" not in chunk:
            issues.append(f"chunk {position}: missing total")
        else:
            declared_total = chunk["total"]
            if isinstance(declared_total, bool) or not isinstance(
                declared_total, int
            ):
                issues.append(f"chunk {position}: total must be an integer")
            elif declared_total != total_count:
                issues.append(f"chunk {position}: total does not match chunk count")

        if "final" not in chunk:
            issues.append(f"chunk {position}: missing final marker")
        elif not isinstance(chunk["final"], bool):
            issues.append(f"chunk {position}: final marker must be a boolean")
        elif chunk["final"] is not (position == total_count - 1):
            issues.append(f"chunk {position}: final marker misplaced")

        message_id = chunk.get("message_id")
        if message_id is not None:
            if not isinstance(message_id, bytes) or len(message_id) != _MESSAGE_ID_SIZE:
                issues.append(f"chunk {position}: message_id must be {_MESSAGE_ID_SIZE} bytes")
            else:
                message_ids.add(message_id)
    if len(message_ids) > 1:
        issues.append("chunks carry inconsistent message identifiers")
    return issues


def aead_batch(messages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Seal and immediately open several independent messages.

    Every message derives its own key from its own master key and context;
    no key material is shared across entries.
    """

    if not isinstance(messages, Mapping):
        raise TypeError("messages must be a mapping")
    results: dict[str, Any] = {}
    for name, spec in messages.items():
        if not isinstance(spec, Mapping):
            raise TypeError("each message must be a mapping")
        master = spec["master"]
        context = spec["context"]
        data = spec["data"]
        key = derive_key(master, context)
        chunks = _seal_message_with_key(key, context, data)
        plaintext = _open_message_with_key(key, context, chunks)
        results[name] = {
            "context": context,
            "chunks": chunks,
            "plaintext": plaintext,
        }
    return results
