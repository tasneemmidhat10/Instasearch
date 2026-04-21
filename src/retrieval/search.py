"""
Retrieval-time helpers.

The canonical implementation of Top-K retrieval evaluation lives in
`src.training.evaluate` (chunked to avoid an O(N^2) similarity matrix).
This module re-exports it so callers that import from `src.retrieval.search`
get the same, memory-safe version.
"""

from ..training.evaluate import evaluate_top_k_retrieval

__all__ = ["evaluate_top_k_retrieval"]
