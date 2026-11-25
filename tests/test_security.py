"""Adversarial checks for the chunked AEAD invariants.

These tests do not replace ``test_public.py``; they exercise every invariant
stated in the README:

* any reordering, truncation or splice is rejected,
* every chunk is bound to its index, total, final marker and full context,
* every failure surfaces as the single ``AEADChunkError`` type,
* batch messages are fully independent,
* the same plaintext encrypts to different ciphertexts that both decrypt.
"""

from __future__ import annotations

from typing import Any

import pytest

from aeadchunk import (
    AEADChunkError,
    aead_batch,
    build_aad,
    derive_key,
    open_chunk,
    open_message,
    seal_chunk,
    seal_message,
    validate_chunks,
)

MASTER_A = b"A" * 32
MASTER_B = b"B" * 32
CONTEXT_A = {"tenant": "alpha", "stream": "archive"}
CONTEXT_B = {"tenant": "beta", "stream": "archive"}


def _rejected(call) -> bool:
    with pytest.raises(AEADChunkError):
        call()
    return True


def _flip_first_byte(value: bytes) -> bytes:
    return bytes([value[0] ^ 0x01]) + value[1:]


# ---------------------------------------------------------------------------
# Round trips and non-determinism
# ---------------------------------------------------------------------------


def test_same_plaintext_encrypts_differently_but_both_open() -> None:
    payload = b"repeatable payload " * 5
    first = seal_message(MASTER_A, CONTEXT_A, payload)
    second = seal_message(MASTER_A, CONTEXT_A, payload)

    assert [c["ciphertext"] for c in first] != [
        c["ciphertext"] for c in second
    ]
    assert [c["nonce"] for c in first] != [c["nonce"] for c in second]
    assert open_message(MASTER_A, CONTEXT_A, first) == payload
    assert open_message(MASTER_A, CONTEXT_A, second) == payload


def test_empty_and_exact_boundary_messages_round_trip() -> None:
    assert open_message(MASTER_A, CONTEXT_A, seal_message(MASTER_A, CONTEXT_A, b"")) == b""
    boundary = b"x" * 32
    chunks = seal_message(MASTER_A, CONTEXT_A, boundary)
    assert len(chunks) == 1
    assert open_message(MASTER_A, CONTEXT_A, chunks) == boundary

    two = b"y" * 33
    chunks = seal_message(MASTER_A, CONTEXT_A, two)
    assert len(chunks) == 2
    assert open_message(MASTER_A, CONTEXT_A, chunks) == two


# ---------------------------------------------------------------------------
# Cross-message / cross-context chunk swapping (the headline attack)
# ---------------------------------------------------------------------------


def test_cross_context_chunk_swap_is_rejected() -> None:
    # Same master key, same chunk shape, different context: the swapped chunk
    # carries an AAD bound to the wrong context, so its tag must fail.
    payload = b"swap payload " * 7  # 84 bytes -> 3 chunks
    chunks_a = seal_message(MASTER_A, CONTEXT_A, payload)
    chunks_b = seal_message(MASTER_A, CONTEXT_B, payload)
    assert len(chunks_a) == len(chunks_b)

    spliced = [dict(c) for c in chunks_a]
    spliced[1] = dict(chunks_b[1])
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, spliced))


def test_cross_key_chunk_swap_is_rejected() -> None:
    # Different master keys, identical context: the derived chunk keys differ,
    # so the splice must be rejected even though the metadata matches.
    payload = b"swap payload " * 7
    chunks_a = seal_message(MASTER_A, CONTEXT_A, payload)
    chunks_b = seal_message(MASTER_B, CONTEXT_A, payload)

    spliced = [dict(c) for c in chunks_a]
    spliced[1] = dict(chunks_b[1])
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, spliced))


def test_replayed_chunk_from_another_message_is_rejected() -> None:
    # Two messages under the same key/context, splice a chunk in and drop one
    # of the originals: total/position metadata stays consistent only if the
    # sequence still claims a valid structure -- here we duplicate a position,
    # which the structural layer must reject.
    payload = b"replay payload " * 7
    chunks = seal_message(MASTER_A, CONTEXT_A, payload)
    replay = [dict(c) for c in chunks]
    replay[1] = dict(chunks[0])  # position 1 now carries position-0's chunk
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, replay))


def test_chunk_moved_between_contexts_is_rejected() -> None:
    chunk = seal_chunk(
        derive_key(MASTER_A, CONTEXT_A), 0, 1, True, CONTEXT_A, b"x"
    )
    _rejected(
        lambda: open_chunk(
            derive_key(MASTER_A, CONTEXT_B), chunk, 0, 1, CONTEXT_B
        )
    )
    # Same key bytes, wrong context argument: the AAD must still disagree.
    _rejected(
        lambda: open_chunk(
            derive_key(MASTER_A, CONTEXT_A), chunk, 0, 1, CONTEXT_B
        )
    )


def test_context_is_bound_into_derived_key() -> None:
    assert derive_key(MASTER_A, CONTEXT_A) != derive_key(MASTER_A, CONTEXT_B)
    chunks = seal_message(MASTER_A, CONTEXT_A, b"context key test " * 3)
    _rejected(lambda: open_message(MASTER_A, CONTEXT_B, chunks))


def test_type_confusion_contexts_have_distinct_encodings() -> None:
    # JSON-without-type-tags would conflate these; the canonical encoder must not.
    assert build_aad(0, 1, True, {"v": True}) != build_aad(
        0, 1, True, {"v": 1}
    )
    assert build_aad(0, 1, True, {"v": None}) != build_aad(
        0, 1, True, {"v": "null"}
    )
    assert build_aad(0, 1, True, {"a": "b"}) != build_aad(
        0, 1, True, {"ab": ""}
    )
    assert build_aad(0, 1, True, b"ab") != build_aad(0, 1, True, "ab")


# ---------------------------------------------------------------------------
# Reordering, duplication, truncation and appending
# ---------------------------------------------------------------------------


def test_all_permutations_of_three_chunks_are_rejected() -> None:
    import itertools

    chunks = seal_message(MASTER_A, CONTEXT_A, b"permute payload " * 6)
    assert len(chunks) >= 3
    original = tuple(c["index"] for c in chunks)
    for perm in itertools.permutations(chunks):
        if tuple(c["index"] for c in perm) == original:
            continue
        _rejected(lambda p=list(perm): open_message(MASTER_A, CONTEXT_A, p))


def test_duplicated_chunk_splice_is_rejected() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"dupe payload " * 6)
    duplicated = [dict(c) for c in chunks]
    duplicated.insert(1, dict(chunks[1]))
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, duplicated))


def test_appending_extra_chunk_is_rejected() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"append payload " * 4)
    extended = [dict(c) for c in chunks]
    extended.append(dict(chunks[-1]))
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, extended))


def test_truncation_at_every_prefix_is_rejected() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"truncate payload " * 8)
    for cut in range(len(chunks)):
        _rejected(
            lambda c=cut: open_message(MASTER_A, CONTEXT_A, chunks[:c])
        )


def test_forged_final_marker_without_tag_is_rejected() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"marker payload " * 4)
    forged = [dict(c) for c in chunks]
    forged[0]["final"] = True
    forged[-1]["final"] = False
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, forged))


def test_metadata_tampering_is_rejected_even_without_ciphertext_touch() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"meta payload " * 4)
    assert len(chunks) >= 2

    def _tamper(field: str, value: Any) -> list[dict[str, Any]]:
        tampered = [dict(c) for c in chunks]
        tampered[0][field] = value
        return tampered

    for field, value in (("index", 99), ("total", 99)):
        _rejected(lambda f=field, v=value: open_message(
            MASTER_A, CONTEXT_A, _tamper(f, v)
        ))

    # Strip the final marker off the last chunk and forge one on the first.
    marker_stripped = [dict(c) for c in chunks]
    marker_stripped[-1]["final"] = False
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, marker_stripped))

    marker_forged = [dict(c) for c in chunks]
    marker_forged[0]["final"] = True
    marker_forged[-1]["final"] = False
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, marker_forged))

    # Single-chunk message: stripping its only final marker must fail too.
    single = seal_message(MASTER_A, CONTEXT_A, b"lone")
    stripped = [dict(single[0])]
    stripped[0]["final"] = False
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, stripped))


def test_nonce_and_ciphertext_tampering_is_rejected() -> None:
    chunk = seal_chunk(
        derive_key(MASTER_A, CONTEXT_A), 0, 1, True, CONTEXT_A, b"x" * 8
    )
    tampered_nonce = dict(chunk)
    tampered_nonce["nonce"] = _flip_first_byte(chunk["nonce"])
    _rejected(
        lambda: open_chunk(
            derive_key(MASTER_A, CONTEXT_A), tampered_nonce, 0, 1, CONTEXT_A
        )
    )
    tampered_ct = dict(chunk)
    tampered_ct["ciphertext"] = _flip_first_byte(chunk["ciphertext"])
    _rejected(
        lambda: open_chunk(
            derive_key(MASTER_A, CONTEXT_A), tampered_ct, 0, 1, CONTEXT_A
        )
    )


# ---------------------------------------------------------------------------
# Uniform error type and malformed framing
# ---------------------------------------------------------------------------


def test_every_failure_is_the_same_error_type_and_message() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"uniform error " * 4)
    malformed_cases: list[Any] = [
        [],
        [object()],
        [{"index": 0, "total": 1, "final": True}],
        [
            {
                "index": 0,
                "total": 1,
                "final": True,
                "nonce": b"short",
                "ciphertext": b"0" * 20,
            }
        ],
        [
            {
                "index": "0",
                "total": 1,
                "final": True,
                "nonce": b"n" * 12,
                "ciphertext": b"0" * 20,
            }
        ],
        [
            {
                "index": 0,
                "total": 0,
                "final": True,
                "nonce": b"n" * 12,
                "ciphertext": b"0" * 20,
            }
        ],
        "not a sequence",
    ]
    messages = set()
    for case in malformed_cases:
        with pytest.raises(AEADChunkError) as excinfo:
            open_message(MASTER_A, CONTEXT_A, case)
        messages.add(str(excinfo.value))

    reordered = [chunks[1], chunks[0], *chunks[2:]]
    with pytest.raises(AEADChunkError) as excinfo:
        open_message(MASTER_A, CONTEXT_A, reordered)
    messages.add(str(excinfo.value))

    truncated = chunks[:-1]
    with pytest.raises(AEADChunkError) as excinfo:
        open_message(MASTER_A, CONTEXT_A, truncated)
    messages.add(str(excinfo.value))

    assert messages == {"message authentication failed"}


def test_open_chunk_enforces_expected_position() -> None:
    key = derive_key(MASTER_A, CONTEXT_A)
    chunks = seal_message(MASTER_A, CONTEXT_A, b"position payload " * 4)
    _rejected(lambda: open_chunk(key, chunks[1], 0, len(chunks), CONTEXT_A))
    _rejected(lambda: open_chunk(key, chunks[0], 1, len(chunks), CONTEXT_A))
    _rejected(lambda: open_chunk(key, chunks[0], 0, len(chunks) + 1, CONTEXT_A))


def test_validate_chunks_flags_structural_problems() -> None:
    chunks = seal_message(MASTER_A, CONTEXT_A, b"validate payload " * 4)
    assert validate_chunks(chunks) == []
    assert validate_chunks([])
    assert validate_chunks("nope")
    reordered = [chunks[1], chunks[0], *chunks[2:]]
    assert validate_chunks(reordered)
    truncated = chunks[:-1]
    assert validate_chunks(truncated)


# ---------------------------------------------------------------------------
# Batch independence
# ---------------------------------------------------------------------------


def test_batch_uses_each_messages_own_key_and_context() -> None:
    # Same plaintext under different masters/contexts: results must still be
    # independently openable with the matching (master, context) pair only.
    shared = b"shared batch body"
    result = aead_batch(
        {
            "first": {
                "master": MASTER_A,
                "context": CONTEXT_A,
                "data": shared,
            },
            "second": {
                "master": MASTER_B,
                "context": CONTEXT_B,
                "data": shared,
            },
        }
    )
    assert result["first"]["plaintext"] == shared
    assert result["second"]["plaintext"] == shared
    assert open_message(
        MASTER_A, CONTEXT_A, result["first"]["chunks"]
    ) == shared
    assert open_message(
        MASTER_B, CONTEXT_B, result["second"]["chunks"]
    ) == shared
    # Cross-opening must fail: keys are independently derived.
    _rejected(
        lambda: open_message(
            MASTER_A, CONTEXT_A, result["second"]["chunks"]
        )
    )
    _rejected(
        lambda: open_message(
            MASTER_B, CONTEXT_B, result["first"]["chunks"]
        )
    )
    # Same plaintext, different contexts -> different ciphertexts.
    assert (
        result["first"]["chunks"][0]["ciphertext"]
        != result["second"]["chunks"][0]["ciphertext"]
    )


def test_one_bad_batch_message_does_not_poison_the_others() -> None:
    result = aead_batch(
        {
            "first": {
                "master": MASTER_A,
                "context": CONTEXT_A,
                "data": b"one",
            },
            "second": {
                "master": MASTER_B,
                "context": CONTEXT_B,
                "data": b"two",
            },
        }
    )
    tampered = [dict(c) for c in result["first"]["chunks"]]
    tampered[0]["ciphertext"] = _flip_first_byte(tampered[0]["ciphertext"])
    _rejected(lambda: open_message(MASTER_A, CONTEXT_A, tampered))
    # The untouched message remains fully valid.
    assert (
        open_message(
            MASTER_B, CONTEXT_B, result["second"]["chunks"]
        )
        == b"two"
    )
