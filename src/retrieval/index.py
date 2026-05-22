"""FAISS HNSW index: build, search, persist, and sweep utilities.

The index is built over L2-normalised embeddings; using
``METRIC_INNER_PRODUCT`` then gives cosine similarity for free, with no
extra normalisation step at query time.

FAISS HNSW runs on CPU only (GPU is only available for IVF-family indices).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.utils.config import EMBED_DIM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List


@dataclass
class HNSWConfig:
    """All tunable knobs for the HNSW retrieval stage.

    Attributes:
        embed_dim:       Dimensionality of L2-normalised embeddings.
        M:               Max edges per node in the HNSW graph.
        ef_construction: Candidate pool size during index construction.
        ef_search:       Candidate pool size at query time (runtime-tunable).
        k_retrieve:      Number of nearest neighbours to return per query.
        metric:          ``"cosine"`` (inner product on L2-norm vectors) or ``"l2"``.
        index_dir:       Directory where the index artefacts are saved.
        index_name:      Filename stem for the three artefact files.
        k_eval:          Recall@k breakpoints reported by ``evaluate_recall``.
    """
    embed_dim:       int       = EMBED_DIM
    M:               int       = 32
    ef_construction: int       = 200
    ef_search:       int       = 128
    k_retrieve:      int       = 100
    metric:          str       = "cosine"
    index_dir:       str       = "./hnsw_index"
    index_name:      str       = "peptide_hnsw"
    k_eval:          List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])

    def __post_init__(self):
        assert self.ef_construction >= self.M, (
            f"ef_construction ({self.ef_construction}) must be >= M ({self.M})"
        )
        assert self.ef_search >= max(self.k_eval), (
            f"ef_search ({self.ef_search}) must be >= max(k_eval) ({max(self.k_eval)})"
        )


# ---------------------------------------------------------------------------
# HNSWIndex
# ---------------------------------------------------------------------------

class HNSWIndex:
    """Thin wrapper around ``faiss.IndexHNSWFlat``.

    Usage::

        cfg   = HNSWConfig()
        index = HNSWIndex(cfg).build(pep_emb)

        dists, idxs = index.search(spec_emb, k=100)
        index.save()

        # Later:
        index = HNSWIndex.load("./hnsw_index", "peptide_hnsw")
    """

    def __init__(self, cfg: HNSWConfig):
        self.cfg    = cfg
        self.index  = None
        self.id_map = None
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        ids:        Optional[np.ndarray] = None,
    ) -> "HNSWIndex":
        """Add ``embeddings`` to a fresh HNSW graph.

        Args:
            embeddings: ``float32 (N, D)`` — must be L2-normalised when
                ``cfg.metric == "cosine"``.
            ids: Optional ``int64 (N,)`` external IDs. Defaults to
                ``np.arange(N)``.

        Returns:
            ``self`` (for chaining).
        """
        if embeddings.dtype != np.float32:
            raise TypeError("FAISS requires float32 embeddings")
        N, D = embeddings.shape
        if N == 0:
            raise ValueError("Cannot build an HNSW index from an empty embedding matrix")
        if D != self.cfg.embed_dim:
            raise ValueError(f"Expected embed_dim={self.cfg.embed_dim}, got {D}")

        if ids is None:
            self.id_map = np.arange(N, dtype=np.int64)
        else:
            if len(ids) != N:
                raise ValueError(f"ids length ({len(ids)}) must match embeddings rows ({N})")
            self.id_map = np.asarray(ids, dtype=np.int64)

        metric = (faiss.METRIC_INNER_PRODUCT
                  if self.cfg.metric == "cosine" else faiss.METRIC_L2)
        self.index = faiss.IndexHNSWFlat(D, self.cfg.M, metric)
        self.index.hnsw.efConstruction = self.cfg.ef_construction
        self.index.hnsw.efSearch       = self.cfg.ef_search

        print(
            f"Building HNSW index  N={N:,}  D={D}  "
            f"M={self.cfg.M}  ef_construction={self.cfg.ef_construction}"
        )
        t0 = time.time()
        self.index.add(embeddings)
        elapsed = time.time() - t0

        n_layers = max(1, int(math.log(N) / math.log(self.cfg.M)))
        ram_mb   = N * self.cfg.M * 4 * 2 * n_layers / 1e6
        print(
            f"  Done in {elapsed:.1f}s  ({N / elapsed:,.0f} vecs/s)  "
            f"ntotal={self.index.ntotal:,}  ~{ram_mb:.0f} MB graph RAM"
        )
        self._built = True
        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        queries: np.ndarray,
        k:       Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate nearest-neighbour search.

        Args:
            queries: ``float32 (Q, D)`` query embeddings.
            k:       Number of neighbours to return. Defaults to
                     ``cfg.k_retrieve``.

        Returns:
            distances: ``float32 (Q, k)`` — cosine similarities (higher = better).
            indices:   ``int64 (Q, k)``   — positions in the original DB passed
                       to :meth:`build` (or external IDs if ``ids`` was given).
        """
        if not self._built:
            raise RuntimeError("Call build() before search()")
        if queries.dtype != np.float32:
            raise TypeError("FAISS requires float32 queries")
        k = k or self.cfg.k_retrieve
        dists, fids = self.index.search(queries, k)
        mapped = np.full(fids.shape, -1, dtype=self.id_map.dtype)
        valid = fids >= 0
        mapped[valid] = self.id_map[fids[valid]]
        return dists, mapped

    def set_ef_search(self, ef: int) -> None:
        """Tune the query-time candidate pool without rebuilding the index."""
        self.cfg.ef_search       = ef
        self.index.hnsw.efSearch = ef

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write index, id-map, and config to ``cfg.index_dir``."""
        out = Path(self.cfg.index_dir)
        out.mkdir(parents=True, exist_ok=True)
        fp_idx = out / (self.cfg.index_name + ".faiss")
        fp_ids = out / (self.cfg.index_name + "_idmap.npy")
        fp_cfg = out / (self.cfg.index_name + "_cfg.json")

        faiss.write_index(self.index, str(fp_idx))
        np.save(str(fp_ids), self.id_map)
        fp_cfg.write_text(json.dumps(asdict(self.cfg), indent=2))

        print(f"Saved index to {out}/  ({fp_idx.stat().st_size / 1e6:.1f} MB)")

    @classmethod
    def load(cls, index_dir: str, index_name: str) -> "HNSWIndex":
        """Reload an index previously saved with :meth:`save`."""
        base = Path(index_dir)
        cfg  = HNSWConfig(**json.loads((base / (index_name + "_cfg.json")).read_text()))
        obj  = cls(cfg)
        obj.index  = faiss.read_index(str(base / (index_name + ".faiss")))
        obj.id_map = np.load(str(base / (index_name + "_idmap.npy")))
        obj._built = True
        print(f"Loaded index  ntotal={obj.index.ntotal:,}  efSearch={obj.index.hnsw.efSearch}")
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def info(self) -> None:
        if not self._built:
            print("Index not built yet.")
            return
        print("\n── HNSWIndex ────────────────────────────────────")
        print(f"  ntotal          : {self.index.ntotal:,}")
        print(f"  embed_dim       : {self.cfg.embed_dim}")
        print(f"  M               : {self.cfg.M}")
        print(f"  ef_construction : {self.cfg.ef_construction}")
        print(f"  ef_search       : {self.index.hnsw.efSearch}")
        print(f"  max_layer       : {self.index.hnsw.max_level}")
        print(f"  metric          : {self.cfg.metric}")
        print("─────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Brute-force baseline
# ---------------------------------------------------------------------------

def build_exact_index(
    embeddings: np.ndarray,
    metric:     str = "cosine",
) -> faiss.Index:
    """Build an exact (brute-force) FAISS index for comparison/debugging.

    Args:
        embeddings: ``float32 (N, D)`` — L2-normalised when metric is cosine.
        metric:     ``"cosine"`` or ``"l2"``.

    Returns:
        A ``faiss.IndexFlatIP`` or ``faiss.IndexFlatL2`` with all vectors added.
    """
    if embeddings.dtype != np.float32:
        raise TypeError("FAISS requires float32 embeddings")
    D = embeddings.shape[1]
    idx = (faiss.IndexFlatIP(D) if metric == "cosine"
           else faiss.IndexFlatL2(D))
    idx.add(embeddings)
    return idx


# ---------------------------------------------------------------------------
# Hyperparameter sweeps
# ---------------------------------------------------------------------------

def sweep_ef_search(
    index:      HNSWIndex,
    queries:    np.ndarray,
    gt_seqs:    list[str],
    db_seqs:    list[str],
    ef_vals:    list[int] | None = None,
    k:          int = 10,
) -> list[dict]:
    """Measure Recall@k and latency for a range of ``ef_search`` values.

    GT matching is done by sequence string so the sweep is correct even
    when the index was built over a deduplicated peptide database.

    Args:
        index:   Fully built :class:`HNSWIndex`.
        queries: ``float32 (Q, D)`` spectrum embeddings.
        gt_seqs: Ground-truth peptide string for each query.
        db_seqs: Peptide strings in the same order as the index rows.
        ef_vals: ef_search values to test. Defaults to ``[16, 32, 64, 128, 256, 512]``.
        k:       Recall@k to report.

    Returns:
        List of ``{"ef": int, "recall": float, "ms_per_query": float}`` dicts.
    """
    ef_vals = ef_vals or [16, 32, 64, 128, 256, 512]
    restore = index.cfg.ef_search
    N = len(queries)
    rows: list[dict] = []

    for ef in ef_vals:
        index.set_ef_search(ef)
        t0 = time.time()
        _, idx_s = index.search(queries, k=k)
        lat = (time.time() - t0) / N * 1000
        hits = sum(
            any(db_seqs[int(j)] == gt_seqs[i] for j in idx_s[i] if j >= 0)
            for i in range(N)
        )
        rows.append({"ef": ef, "recall": hits / N, "ms_per_query": lat})

    index.set_ef_search(restore)
    return rows


def sweep_build_params(
    embeddings:   np.ndarray,
    queries:      np.ndarray,
    gt_seqs:      list[str],
    db_seqs:      list[str],
    m_vals:       list[int] | None = None,
    ef_c_vals:    list[int] | None = None,
    k:            int = 10,
    metric:       str = "cosine",
) -> list[dict]:
    """Grid-search over ``(M, ef_construction)`` pairs on a subset.

    Args:
        embeddings: ``float32 (N, D)`` DB embeddings.
        queries:    ``float32 (Q, D)`` query embeddings.
        gt_seqs:    Ground-truth peptide strings for each query.
        db_seqs:    Peptide strings for DB rows.
        m_vals:     HNSW M values to try.
        ef_c_vals:  ef_construction values to try.
        k:          Recall@k to report.
        metric:     ``"cosine"`` or ``"l2"``.

    Returns:
        List of ``{"M": int, "ef_construction": int, "recall": float,
        "build_s": float}`` dicts.
    """
    m_vals    = m_vals    or [16, 32, 64]
    ef_c_vals = ef_c_vals or [100, 200, 400]
    rows: list[dict] = []

    for M in m_vals:
        for ef_c in ef_c_vals:
            cfg = HNSWConfig(
                embed_dim=embeddings.shape[1],
                M=M, ef_construction=ef_c, ef_search=max(ef_c_vals),
                metric=metric,
            )
            t0  = time.time()
            idx = HNSWIndex(cfg).build(embeddings)
            build_s = time.time() - t0

            _, idx_s = idx.search(queries, k=k)
            hits = sum(
                any(db_seqs[int(j)] == gt_seqs[i] for j in idx_s[i] if j >= 0)
                for i in range(len(queries))
            )
            rows.append({
                "M": M,
                "ef_construction": ef_c,
                "recall": hits / len(queries),
                "build_s": round(build_s, 2),
            })

    return rows


__all__ = [
    "HNSWConfig",
    "HNSWIndex",
    "build_exact_index",
    "sweep_ef_search",
    "sweep_build_params",
]
