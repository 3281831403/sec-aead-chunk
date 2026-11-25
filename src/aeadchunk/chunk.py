"""Chunked authenticated encryption with message-level structure checks.

A message is split into fixed-size chunks.  Every chunk is encrypted with an
encrypt-then-MAC construction whose associated data binds the chunk to:

* its position in the message (``index``),
* the total number of chunks (``total``),
* the final-chunk marker (``final``),
* and the full, canonically encoded message ``context``.

The message-level decryption path additionally enforces the structural
invariant that the presented chunks form exactly one complete, ordered
message: every declared position is present, nothing is duplicated or
missing, and the final marker sits on exactly the last chunk.  Any failure
during decryption -- authentication, structure, or malformed framing -- is
reported as a single :class:`AEADChunkError` so callers cannot distinguish
failure modes and act as an oracle.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from collections.abc import Mapping
from typing import Any

CHUNK_SIZE = 32
_NONCE_SIZE = 12
_KEY_SIZE = 32
_TAG_SIZE = 16

# Single, constant message for every decryption failure: revealing whether a
# failure came from a bad tag, bad structure or malformed fields would hand an
# attacker an oracle.
_AUTH_FAIL = "message authentication failed"

_AAD_DOMAIN = b"aeadchunk/aad/v2"
_KEY_DOMAIN = b"aeadchunk/key/v1"
_KDF_SALT = b"aeadchunk/kdf/salt/v1"
_CONTEXT_DOMAIN = b"aeadchunk/ctx/v2"


class _InvalidTag(Exception):
    """Internal: authentication tag mismatch. Never escapes the module."""


class AEADChunkError(Exception):
    """Raised when a chunk or a whole message cannot be authenticated."""


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hashlib.sha256(
                b"ks" + key + nonce + counter.to_bytes(8, "big")
            ).digest()
        )
        counter += 1
    return bytes(out[:length])


def _aead_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    stream = _keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(
        key, b"tag" + aad + nonce + ciphertext, hashlib.sha256
    ).digest()[:_TAG_SIZE]
    return ciphertext + tag


def _aead_decrypt(key: bytes, nonce: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < _TAG_SIZE:
        raise _InvalidTag("short ciphertext")
    ciphertext, tag = blob[:-_TAG_SIZE], blob[-_TAG_SIZE:]
    expect = hmac.new(
        key, b"tag" + aad + nonce + ciphertext, hashlib.sha256
    ).digest()[:_TAG_SIZE]
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
        block = hmac.new(
            prk, block + info + bytes([counter]), hashlib.sha256
        ).digest()
        okm += block
        counter += 1
    return okm[:length]


# ---------------------------------------------------------------------------
# Canonical context encoding
# ---------------------------------------------------------------------------
#
# JSON has no type tags, so distinct Python values can collide (True vs 1,
# strings that look like numbers, ...).  Every value below is encoded with an
# explicit type tag and every variable-length field is length-framed, making
# the encoding unambiguous: two different values never produce the same bytes.


def _frame(tag: bytes, raw: bytes) -> bytes:
    return tag + str(len(raw)).encode("ascii") + b":" + raw


def _encode_value(value: Any) -> bytes:
    if value is None:
        return b"n:"
    if isinstance(value, bool):
        return b"t:" if value else b"f:"
    if isinstance(value, int):
        return _frame(b"i:", str(value).encode("ascii"))
    if isinstance(value, float):
        return _frame(b"d:", repr(value).encode("ascii"))
    if isinstance(value, str):
        return _frame(b"s:", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return _frame(b"b:", bytes(value))
    if isinstance(value, (list, tuple)):
        body = b"".join(_encode_value(item) for item in value)
        return _frame(b"l:", body)
    if isinstance(value, (set, frozenset)):
        body = b"".join(_encode_value(item) for item in sorted(value))
        return _frame(b"l:", body)
    if isinstance(value, Mapping):
        encoded_items = []
        for key, item in value.items():
            encoded_items.append(_encode_value(key) + _encode_value(item))
        encoded_items.sort()
        return _frame(b"m:", b"".join(encoded_items))
    raise TypeError(f"unsupported context value: {type(value).__name__}")


def _context_bytes(context: Any) -> bytes:
    if not isinstance(context, (bytes, bytearray, str, Mapping)):
        raise TypeError("context must be bytes, str, or a mapping")
    blob = _encode_value(context)
    return _CONTEXT_DOMAIN + str(len(blob)).encode("ascii") + b":" + blob


# ---------------------------------------------------------------------------
# Position metadata
# ---------------------------------------------------------------------------


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

    The full context is bound into the HKDF ``info`` field, so distinct
    contexts under the same master key yield independent chunk keys.
    """

    if not isinstance(master, (bytes, bytearray)):
        raise TypeError("master must be bytes")
    ctx = _context_bytes(context)
    info = _KEY_DOMAIN + str(len(ctx)).encode("ascii") + b":" + ctx
    return _hkdf(bytes(master), _KDF_SALT, info)


def build_aad(index: int, total: int, final: bool, context: Any) -> bytes:
    """Build the associated data for one chunk of a message.

    The AAD covers every piece of placement metadata (index, total, final
    marker) plus the full context, so a chunk cannot be moved to another
    position, message or context without invalidating its tag.
    """

    _validate_position(index, total, final)
    header = struct.pack(">QQB", index, total, 1 if final else 0)
    return _AAD_DOMAIN + header + _context_bytes(context)


# ---------------------------------------------------------------------------
# Single-chunk API
# ---------------------------------------------------------------------------


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
    """Decrypt one chunk and return its plaintext.

    The chunk's stored ``index``/``total`` must match the position at which it
    is presented, and every decryption failure is raised uniformly as
    :class:`AEADChunkError`.
    """

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("key must be 32 bytes")
    try:
        if not isinstance(chunk, Mapping):
            raise AEADChunkError(_AUTH_FAIL)
        nonce = chunk.get("nonce")
        ciphertext = chunk.get("ciphertext")
        index = chunk.get("index")
        stored_total = chunk.get("total")
        final = chunk.get("final")
        if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != _NONCE_SIZE:
            raise AEADChunkError(_AUTH_FAIL)
        if (
            not isinstance(ciphertext, (bytes, bytearray))
            or len(ciphertext) < _TAG_SIZE
        ):
            raise AEADChunkError(_AUTH_FAIL)
        if isinstance(index, bool) or not isinstance(index, int):
            raise AEADChunkError(_AUTH_FAIL)
        if isinstance(stored_total, bool) or not isinstance(stored_total, int):
            raise AEADChunkError(_AUTH_FAIL)
        if not isinstance(final, bool):
            raise AEADChunkError(_AUTH_FAIL)
        if index != expected_index or stored_total != total:
            raise AEADChunkError(_AUTH_FAIL)
        # Re-validates bounds (0 <= index < stored_total, stored_total >= 1)
        # and that the final marker can only sit on the last chunk.
        _validate_position(index, stored_total, final)
        if final is not (index == stored_total - 1):
            raise AEADChunkError(_AUTH_FAIL)
        aad = build_aad(index, stored_total, final, context)
        return _aead_decrypt(
            bytes(key), bytes(nonce), bytes(ciphertext), aad
        )
    except AEADChunkError:
        raise
    except Exception as exc:  # noqa: BLE001 - uniform external error type
        raise AEADChunkError(_AUTH_FAIL) from exc


# ---------------------------------------------------------------------------
# Whole-message API
# ---------------------------------------------------------------------------


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
        raise AEADChunkError(_AUTH_FAIL)
    if not chunks:
        # An empty sequence cannot carry an authenticated final marker.
        raise AEADChunkError(_AUTH_FAIL)
    try:
        first = chunks[0]
        if not isinstance(first, Mapping):
            raise AEADChunkError(_AUTH_FAIL)
        declared_total = first.get("total")
        if (
            isinstance(declared_total, bool)
            or not isinstance(declared_total, int)
            or declared_total < 1
        ):
            raise AEADChunkError(_AUTH_FAIL)
        # The presented sequence must be exactly one whole message: no
        # truncation, no appended/duplicated chunks.
        if len(chunks) != declared_total:
            raise AEADChunkError(_AUTH_FAIL)

        parts: list[bytes] = []
        final_position = declared_total - 1
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise AEADChunkError(_AUTH_FAIL)
            stored_total = chunk.get("total")
            if stored_total != declared_total:
                raise AEADChunkError(_AUTH_FAIL)
            index = chunk.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index != position
            ):
                raise AEADChunkError(_AUTH_FAIL)
            final = chunk.get("final")
            if not isinstance(final, bool):
                raise AEADChunkError(_AUTH_FAIL)
            if final is not (position == final_position):
                raise AEADChunkError(_AUTH_FAIL)
            parts.append(
                open_chunk(key, chunk, position, declared_total, context)
            )
        return b"".join(parts)
    except AEADChunkError:
        raise
    except Exception as exc:  # noqa: BLE001 - uniform external error type
        raise AEADChunkError(_AUTH_FAIL) from exc


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
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total < 1
    ):
        issues.append("chunk 0: missing or invalid total")
        declared_total = None
    elif len(chunks) != declared_total:
        issues.append(
            f"chunk count {len(chunks)} does not match declared total "
            f"{declared_total}"
        )

    last_position = len(chunks) - 1
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
        elif declared_total is not None and chunk["total"] != declared_total:
            issues.append(f"chunk {position}: total disagrees with message total")
        if "final" not in chunk:
            issues.append(f"chunk {position}: missing final marker")
        elif not isinstance(chunk["final"], bool):
            issues.append(f"chunk {position}: final marker must be a boolean")
        else:
            expected_final = (
                declared_total is not None and position == last_position
            )
            if chunk["final"] is not expected_final:
                issues.append(f"chunk {position}: final marker misplaced")
        if "nonce" not in chunk or not isinstance(
            chunk.get("nonce"), (bytes, bytearray)
        ):
            issues.append(f"chunk {position}: missing or invalid nonce")
        if "ciphertext" not in chunk or not isinstance(
            chunk.get("ciphertext"), (bytes, bytearray)
        ):
            issues.append(f"chunk {position}: missing or invalid ciphertext")
    return issues


def aead_batch(messages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Seal and immediately open several independent messages.

    Each message derives its own key from its own master key and context; no
    key or intermediate state is shared between entries, so one message can
    neither influence nor be confused with another.
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
