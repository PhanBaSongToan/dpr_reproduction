"""Vector indexing.

Re-exports the main index classes and helpers for convenient access::

    from src.indexing import InnerProductIndex, build_index
"""

from .faiss_index import (
    InnerProductIndex,
    IVFInnerProductIndex,
    L2Index,
    build_index,
    encode_passages_batched,
)

__all__ = [
    "InnerProductIndex",
    "IVFInnerProductIndex",
    "L2Index",
    "build_index",
    "encode_passages_batched",
]
