"""Five public checks for the chunked AEAD exercise."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aeadchunk import (
    AEADChunkError,
    aead_batch,
    derive_key,
    open_chunk,
    open_message,
    seal_chunk,
    seal_message,
)

MASTER_A = b"A" * 32
MASTER_B = b"B" * 32
CONTEXT_A = {"tenant": "alpha", "stream": "archive"}
CONTEXT_B = {"tenant": "beta", "stream": "archive"}


def _rejected(call: Callable[[], Any]) -> bool:
    try:
        call()
    except AEADChunkError:
        return True
    except Exception:
        return False
    return False


def _attempt(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception:
        return None


def _flip_first_byte(value: bytes) -> bytes:
    return bytes([value[0] ^ 0x01]) + value[1:]


def test_single_chunk_round_trip() -> None:
    key = derive_key(MASTER_A, CONTEXT_A)
    payload = b"single chunk payload"
    chunk = seal_chunk(key, 0, 1, True, CONTEXT_A, payload)

    assert open_chunk(key, chunk, 0, 1, CONTEXT_A) == payload

    tampered = dict(chunk)
    tampered["ciphertext"] = _flip_first_byte(chunk["ciphertext"])
    assert _rejected(
        lambda: open_chunk(key, tampered, 0, 1, CONTEXT_A)
    )


def test_multi_chunk_round_trip() -> None:
    payload = b"multi-block payload " * 6
    chunks = seal_message(MASTER_A, CONTEXT_A, payload)

    assert len(chunks) > 1
    assert open_message(MASTER_A, CONTEXT_A, chunks) == payload

    tampered = [dict(chunk) for chunk in chunks]
    tampered[1]["ciphertext"] = _flip_first_byte(
        tampered[1]["ciphertext"]
    )
    assert _rejected(
        lambda: open_message(MASTER_A, CONTEXT_A, tampered)
    )


def test_final_chunk_marker_and_truncation() -> None:
    payload = b"final marker payload " * 4
    chunks = seal_message(MASTER_A, CONTEXT_A, payload)

    assert len(chunks) > 1
    assert chunks[-1]["final"] is True
    assert _rejected(
        lambda: open_message(MASTER_A, CONTEXT_A, chunks[:-1])
    )


def test_reordered_chunks_are_rejected() -> None:
    payload = b"ordering payload " * 6
    chunks = seal_message(MASTER_A, CONTEXT_A, payload)

    assert len(chunks) >= 3
    reordered = [chunks[2], chunks[0], chunks[1], *chunks[3:]]
    assert _rejected(
        lambda: open_message(MASTER_A, CONTEXT_A, reordered)
    )


def test_batch_messages_are_independent() -> None:
    first = b"first batch message"
    second = b"second batch message"
    result = aead_batch(
        {
            "first": {
                "master": MASTER_A,
                "context": CONTEXT_A,
                "data": first,
            },
            "second": {
                "master": MASTER_B,
                "context": CONTEXT_B,
                "data": second,
            },
        }
    )

    assert result["first"]["plaintext"] == first
    assert result["second"]["plaintext"] == second
    assert (
        _attempt(
            lambda: open_message(
                MASTER_B,
                CONTEXT_B,
                result["second"]["chunks"],
            )
        )
        == second
    )