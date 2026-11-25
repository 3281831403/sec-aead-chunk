"""Public API for the chunked authenticated encryption helpers."""

from .chunk import (
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

__all__ = [
    "AEADChunkError",
    "aead_batch",
    "build_aad",
    "derive_key",
    "open_chunk",
    "open_message",
    "seal_chunk",
    "seal_message",
    "validate_chunks",
]