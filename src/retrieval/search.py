"""Retrieval helpers: embedding extraction, deduplication, batch/single search.

The canonical HNSW index lives in :mod:`src.retrieval.index`.
This module provides:

- :func:`extract_embeddings`    — batched encoding for any dual-encoder model.
- :func:`build_unique_peptide_db` — deduplicate a peptide list before indexing.
- :func:`retrieve_batch`        — top-k retrieval for a full query set.
- :func:`retrieve_single`       — interactive single-spectrum helper.
- :func:`evaluate_recall`       — Recall@k over the HNSW index (sequence-string GT).

The training-time Top-K evaluation (full similarity-matrix approach) lives in
:mod:`src.training.evaluate` and is re-exported here for convenience.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.retrieval.index import HNSWIndex
from src.training.evaluate import evaluate_top_k_retrieval   # re-export


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model_fn,
    dataset:    Dataset,
    mode:       str,
    batch_size: int  = 512,
    device:     Optional[torch.device] = None,
    desc:       str  = "Encoding",
) -> np.ndarray:
    """Encode all items in ``dataset`` and return a float32 numpy array.

    The models' ``ProjectionHead`` already applies L2-normalisation, so no
    post-hoc normalisation is performed here.

    Args:
        model_fn:   Callable that maps inputs to ``(B, embed_dim)`` embeddings.
        dataset:    Any ``torch.utils.data.Dataset`` whose items are
                    ``(specs, peps, pres, ...)`` tuples.
        mode:       ``"spectrum"`` → calls ``model_fn(specs, pres)``.
                    ``"peptide"``  → calls ``model_fn(peps)``.
        batch_size: Items per forward pass.
        device:     Target device. Infers from ``model_fn`` parameters when
                    ``None`` and ``model_fn`` is an ``nn.Module``; falls back
                    to CPU.
        desc:       tqdm progress-bar label.

    Returns:
        ``float32 (N, embed_dim)`` numpy array.
    """
    if device is None:
        if isinstance(model_fn, nn.Module):
            device = next(model_fn.parameters()).device
        else:
            device = torch.device("cpu")

    if isinstance(model_fn, nn.Module):
        model_fn.eval()

    out    = []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            specs, peps, pres = batch[0], batch[1], batch[2]
            if mode == "spectrum":
                z = model_fn(specs.to(device), pres.to(device))
            elif mode == "peptide":
                z = model_fn(peps.to(device))
            else:
                raise ValueError(f"mode must be 'spectrum' or 'peptide', got {mode!r}")
            out.append(z.cpu().float().numpy())

    return np.vstack(out)


# ---------------------------------------------------------------------------
# Peptide deduplication
# ---------------------------------------------------------------------------

def build_unique_peptide_db(
    sequences: List[str],
) -> Tuple[List[str], List[int], np.ndarray]:
    """Deduplicate a peptide list and return the information needed to map
    original row indices back to unique DB positions.

    Multiple spectra can share the same peptide sequence. Inserting duplicate
    embeddings into the HNSW index wastes top-k slots and breaks ground-truth
    matching when the retrieved index points to a *different* duplicate node.
    This function resolves that by keeping only the first occurrence of each
    unique sequence.

    Args:
        sequences: Peptide strings for every row in the training split, in order.

    Returns:
        unique_sequences:  Ordered list of distinct peptide strings (length U).
        unique_indices:    ``len(U)`` list — position in ``sequences`` for each
                           unique peptide's first occurrence.
        row_to_unique:     ``int64 (N,)`` array — ``row_to_unique[i]`` is the
                           unique DB id for ``sequences[i]``.

    Example::

        unique_seqs, unique_idx, row_to_uid = build_unique_peptide_db(train_seqs)
        unique_subset = Subset(train_dataset, unique_idx)
        pep_emb = extract_embeddings(model_pep, unique_subset, mode="peptide")
        hnsw.build(pep_emb)
    """
    seq_to_uid: dict[str, int] = {}
    unique_indices: List[int]  = []

    for row_idx, seq in enumerate(sequences):
        if seq not in seq_to_uid:
            seq_to_uid[seq] = len(unique_indices)
            unique_indices.append(row_idx)

    unique_sequences = [sequences[i] for i in unique_indices]
    row_to_unique    = np.array([seq_to_uid[s] for s in sequences], dtype=np.int64)

    n_total  = len(sequences)
    n_unique = len(unique_sequences)
    print(
        f"Peptide DB: {n_total:,} rows → {n_unique:,} unique  "
        f"({100 * (1 - n_unique / n_total):.1f}% duplicates removed)"
    )
    return unique_sequences, unique_indices, row_to_unique


# ---------------------------------------------------------------------------
# Batch retrieval
# ---------------------------------------------------------------------------

def retrieve_batch(
    query_emb:        np.ndarray,
    ground_truth_seqs: List[str],
    index:             HNSWIndex,
    sequences:         List[str],
    k:                 int,
) -> Tuple[List[List[Tuple[str, float]]], np.ndarray]:
    """Retrieve top-k candidates for every query spectrum.

    Ground-truth matching is done by sequence string (not by row index) so
    the function is correct when the index is built over a deduplicated
    peptide database.

    Args:
        query_emb:          ``float32 (Q, D)`` spectrum embeddings.
        ground_truth_seqs:  True peptide string for each of the Q queries.
        index:              Fully-built :class:`~src.retrieval.index.HNSWIndex`.
        sequences:          Peptide string for each DB entry (``unique_sequences``).
        k:                  Number of candidates to retrieve per query.

    Returns:
        candidates: List of Q lists. Each inner list contains
            ``(peptide_sequence, cosine_score)`` tuples sorted best-first.
        ranks:      ``int32 (Q,)`` — 1-indexed rank of the ground-truth peptide.
            ``k + 1`` means the true peptide was not found in the top-k.
    """
    dists, idxs = index.search(query_emb, k=k)   # (Q, k)

    candidates: List[List[Tuple[str, float]]] = []
    ranks = np.full(len(query_emb), k + 1, dtype=np.int32)

    valid_mask = idxs >= 0   # FAISS uses -1 for unfilled slots

    for i in range(len(query_emb)):
        row_valid = valid_mask[i]
        cands = [
            (sequences[int(idx)], float(dist))
            for idx, dist in zip(idxs[i][row_valid], dists[i][row_valid])
        ]
        candidates.append(cands)

        gt_seq = ground_truth_seqs[i]
        for rank_pos, (seq, _) in enumerate(cands, 1):
            if seq == gt_seq:
                ranks[i] = rank_pos
                break

    return candidates, ranks


# ---------------------------------------------------------------------------
# Single-spectrum retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def retrieve_single(
    spectrum:         np.ndarray,
    precursor:        np.ndarray,
    model_spec:       nn.Module,
    index:            HNSWIndex,
    sequences:        List[str],
    k:                int = 10,
    ground_truth_seq: Optional[str] = None,
    device:           Optional[torch.device] = None,
) -> Tuple[List[Tuple[str, float]], int]:
    """Retrieve candidates for a single spectrum — useful for interactive use.

    Args:
        spectrum:    ``float32 (MAX_PEAKS, 2)`` preprocessed peak array.
        precursor:   ``float32 (2,)`` — ``[precursor_mz, charge]``.
        model_spec:  Spectrum encoder (``InstaSearchSpectrumEncoder``).
        index:       Fully-built :class:`~src.retrieval.index.HNSWIndex`.
        sequences:   Peptide string for each DB entry (``unique_sequences``).
        k:           Number of candidates to retrieve.
        ground_truth_seq: If provided, the function also computes the rank of
            this sequence within the candidates.
        device:      Inference device. Inferred from ``model_spec`` when ``None``.

    Returns:
        candidates: ``[(peptide_sequence, cosine_score), ...]`` sorted best-first.
        rank:       1-indexed rank of ``ground_truth_seq``, or ``k + 1`` if not
            found / not provided.
    """
    if device is None:
        device = next(model_spec.parameters()).device

    model_spec.eval()
    x  = torch.tensor(spectrum,  dtype=torch.float32).unsqueeze(0).to(device)
    pr = torch.tensor(precursor, dtype=torch.float32).unsqueeze(0).to(device)
    z  = model_spec(x, pr).cpu().float().numpy()

    dists, idxs = index.search(z, k=k)
    candidates  = [
        (sequences[int(i)], float(d))
        for i, d in zip(idxs[0], dists[0]) if i >= 0
    ]

    rank = k + 1
    if ground_truth_seq is not None:
        for r, (seq, _) in enumerate(candidates, 1):
            if seq == ground_truth_seq:
                rank = r
                break

    return candidates, rank


# ---------------------------------------------------------------------------
# HNSW Recall@k evaluation
# ---------------------------------------------------------------------------

def evaluate_recall(
    index:      HNSWIndex,
    query_emb:  np.ndarray,
    gt_seqs:    List[str],
    db_seqs:    List[str],
    k_vals:     Optional[List[int]] = None,
) -> dict[int, float]:
    """Compute Recall@k for an HNSW index using sequence-string GT matching.

    This is the correct evaluation when the index is built over a
    deduplicated peptide database (a retrieved peptide is a hit if its
    *string* matches the ground-truth, regardless of DB row index).

    Args:
        index:     Fully-built :class:`~src.retrieval.index.HNSWIndex`.
        query_emb: ``float32 (Q, D)`` spectrum embeddings.
        gt_seqs:   Ground-truth peptide string for each query.
        db_seqs:   Peptide string for each DB entry (``unique_sequences``).
        k_vals:    Recall breakpoints. Defaults to ``[1, 5, 10, 50, 100]``.

    Returns:
        ``{k: recall_at_k, ...}`` dict.
    """
    k_vals   = k_vals or [1, 5, 10, 50, 100]
    max_k    = max(k_vals)
    Q        = len(query_emb)
    _, idxs  = index.search(query_emb.astype(np.float32), k=max_k)   # (Q, max_k)

    results: dict[int, float] = {}
    for k in k_vals:
        hits = sum(
            any(
                j >= 0 and db_seqs[int(j)] == gt_seqs[i]
                for j in idxs[i, :k]
            )
            for i in range(Q)
        )
        results[k] = hits / Q

    print("\n── Recall@k (HNSW, sequence-string GT) ──────────────")
    for k, r in results.items():
        bar = "█" * int(r * 40)
        print(f"  Recall@{k:<4d}: {r:.4f}  {bar}")
    print("─────────────────────────────────────────────────────")

    return results


def candidate_recall(
    index:      HNSWIndex,
    query_emb:  np.ndarray,
    gt_seqs:    List[str],
    db_seqs:    List[str],
    k_vals:     Optional[List[int]] = None,
) -> dict[int, float]:
    """Candidate Recall@k: fraction of unique ground-truth peptides recovered.

    For each spectrum we retrieve the k nearest database peptides in the joint
    embedding space and record a hit if the spectrum's ground-truth peptide is
    among them. A peptide is counted as recovered if at least one of its
    spectra produces a hit. No score threshold or FDR control is applied.

    Contrast with :func:`evaluate_recall`, which averages hits over spectra
    (each spectrum contributes 1/Q). Candidate recall instead averages over
    unique peptides, so an abundant peptide with many spectra contributes the
    same weight as a singleton.

    Args:
        index:     Fully-built :class:`~src.retrieval.index.HNSWIndex`.
        query_emb: ``float32 (Q, D)`` spectrum embeddings.
        gt_seqs:   Ground-truth peptide string for each query (length Q).
        db_seqs:   Peptide string for each DB entry (``unique_sequences``).
        k_vals:    Recall breakpoints. Defaults to ``[1, 5, 10, 50, 100]``.

    Returns:
        ``{k: candidate_recall_at_k, ...}`` — fraction of unique peptides in
        ``gt_seqs`` recovered by their associated spectra's top-k retrievals.
    """
    k_vals = k_vals or [1, 5, 10, 50, 100]
    max_k  = max(k_vals)
    Q      = len(query_emb)
    _, idxs = index.search(query_emb.astype(np.float32), k=max_k)   # (Q, max_k)

    unique_gt = sorted(set(gt_seqs))
    n_unique  = len(unique_gt)

    results: dict[int, float] = {}
    for k in k_vals:
        recovered: set[str] = set()
        for i in range(Q):
            gt = gt_seqs[i]
            if gt in recovered:
                continue
            for j in idxs[i, :k]:
                if j >= 0 and db_seqs[int(j)] == gt:
                    recovered.add(gt)
                    break
        results[k] = len(recovered) / n_unique if n_unique else 0.0

    print("\n── Candidate Recall@k (unique peptides recovered) ───")
    print(f"  Queries: {Q}  |  Unique GT peptides: {n_unique}")
    for k, r in results.items():
        bar = "█" * int(r * 40)
        print(f"  CandRecall@{k:<4d}: {r:.4f}  {bar}")
    print("─────────────────────────────────────────────────────")

    return results


@torch.no_grad()
def compute_fdr(
    model_spec: nn.Module,
    model_pep: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modified_sequences: Optional[List[str]] = None,
    thresholds: Optional[np.ndarray] = None,
    chunk_size: int = 4096,
    duplicate_aggregation: str = "mean",
    normalize: bool = True,
) -> dict:
    """Compute top-1 FDR against a unique modified-sequence peptide database.

    Unlike row-index evaluation, this treats a prediction as correct when the
    retrieved unique peptide sequence matches the query row's
    ``modified_sequence``. This is the right metric when the peptide database
    has been deduplicated by sequence.

    Args:
        model_spec: Spectrum encoder in eval mode.
        model_pep: Peptide encoder in eval mode.
        loader: DataLoader returning at least ``(specs, peps, pres)``. If it
            returns a fourth item, it is treated as per-row modified sequences
            when ``modified_sequences`` is not supplied.
        device: Torch device.
        modified_sequences: Per-row ground-truth modified peptide strings, e.g.
            ``test_df["modified_sequence"].tolist()`` or ``test_dataset.sequences``.
        thresholds: Cosine thresholds for FDR curves. Defaults to
            ``np.arange(0.0, 1.0, 0.05)``.
        chunk_size: Number of spectra per matrix-multiply chunk.
        duplicate_aggregation: ``"mean"`` averages duplicate peptide embeddings
            before search; ``"first"`` keeps the first row for each sequence.
        normalize: L2-normalise embeddings before cosine similarity.

    Returns:
        Dictionary with top-1 precision/FDR, threshold results, scores, and
        sequence-level prediction metadata.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.05)
    if duplicate_aggregation not in {"mean", "first"}:
        raise ValueError("duplicate_aggregation must be 'mean' or 'first'")

    model_spec.eval()
    model_pep.eval()

    all_spec_emb: list[torch.Tensor] = []
    all_pep_emb: list[torch.Tensor] = []
    batch_sequences: list[str] = []

    for batch in loader:
        specs = batch[0].to(device)
        peps = batch[1].to(device)
        pres = batch[2].to(device)

        if modified_sequences is None and len(batch) >= 4:
            batch_sequences.extend([str(seq) for seq in batch[3]])

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep = model_pep(peps)
        else:
            z_spec = model_spec(specs, pres)
            z_pep = model_pep(peps)

        if normalize:
            z_spec = F.normalize(z_spec, dim=-1)
            z_pep = F.normalize(z_pep, dim=-1)

        all_spec_emb.append(z_spec.cpu().float())
        all_pep_emb.append(z_pep.cpu().float())

    spec_emb = torch.cat(all_spec_emb, dim=0).numpy().astype(np.float32, copy=False)
    pep_emb = torch.cat(all_pep_emb, dim=0).numpy().astype(np.float32, copy=False)
    n_rows = spec_emb.shape[0]

    if modified_sequences is None:
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "sequences"):
            modified_sequences = list(dataset.sequences)
        elif hasattr(dataset, "modified_sequences"):
            modified_sequences = list(dataset.modified_sequences)
        elif hasattr(dataset, "df") and "modified_sequence" in dataset.df.columns:
            modified_sequences = dataset.df["modified_sequence"].tolist()
        elif batch_sequences:
            modified_sequences = batch_sequences
        else:
            raise ValueError(
                "Pass modified_sequences=test_df['modified_sequence'].tolist(), "
                "or make the loader return sequences as its fourth batch item."
            )

    modified_sequences = [str(seq) for seq in modified_sequences]
    if len(modified_sequences) != n_rows:
        raise ValueError(
            f"modified_sequences length ({len(modified_sequences)}) must match "
            f"encoded rows ({n_rows})"
        )

    seq_to_uid: dict[str, int] = {}
    unique_sequences: list[str] = []
    row_to_unique = np.empty(n_rows, dtype=np.int64)
    first_rows: list[int] = []

    for row_idx, seq in enumerate(modified_sequences):
        uid = seq_to_uid.get(seq)
        if uid is None:
            uid = len(unique_sequences)
            seq_to_uid[seq] = uid
            unique_sequences.append(seq)
            first_rows.append(row_idx)
        row_to_unique[row_idx] = uid

    n_unique = len(unique_sequences)
    if duplicate_aggregation == "first":
        unique_pep_emb = pep_emb[np.asarray(first_rows, dtype=np.int64)]
    else:
        unique_pep_emb = np.zeros((n_unique, pep_emb.shape[1]), dtype=np.float32)
        counts = np.zeros(n_unique, dtype=np.int64)
        np.add.at(unique_pep_emb, row_to_unique, pep_emb)
        np.add.at(counts, row_to_unique, 1)
        unique_pep_emb /= counts[:, None]
        if normalize:
            norms = np.linalg.norm(unique_pep_emb, axis=1, keepdims=True)
            unique_pep_emb /= np.where(norms == 0.0, 1.0, norms)
            unique_pep_emb = unique_pep_emb.astype(np.float32, copy=False)

    top1_unique_ids = np.empty(n_rows, dtype=np.int64)
    top1_scores = np.empty(n_rows, dtype=np.float32)
    true_scores = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        sims = spec_emb[start:end] @ unique_pep_emb.T
        top_local = np.argmax(sims, axis=1)
        row_ids = np.arange(end - start)
        top1_unique_ids[start:end] = top_local
        top1_scores[start:end] = sims[row_ids, top_local]
        true_scores[start:end] = sims[row_ids, row_to_unique[start:end]]

    predicted_sequences = [unique_sequences[int(uid)] for uid in top1_unique_ids]
    correct = top1_unique_ids == row_to_unique

    tp = int(correct.sum())
    fp = int((~correct).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fdr = 1.0 - precision

    threshold_results = []
    for t in thresholds:
        accepted = top1_scores >= t
        n_accepted = int(accepted.sum())
        if n_accepted == 0:
            threshold_results.append({
                "threshold": float(t),
                "n_accepted": 0,
                "precision": 0.0,
                "fdr": 0.0,
                "fraction_accepted": 0.0,
            })
            continue

        tp_t = int((correct & accepted).sum())
        fp_t = int((~correct & accepted).sum())
        prec_t = tp_t / (tp_t + fp_t)
        threshold_results.append({
            "threshold": float(t),
            "n_accepted": n_accepted,
            "precision": float(prec_t),
            "fdr": float(1.0 - prec_t),
            "fraction_accepted": float(n_accepted / n_rows),
        })

    def _mean_or_nan(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else float("nan")

    def _median_or_nan(values: np.ndarray) -> float:
        return float(np.median(values)) if len(values) else float("nan")

    results = {
        "n_total": int(n_rows),
        "n_unique_peptides": int(n_unique),
        "n_correct_top1": tp,
        "n_incorrect_top1": fp,
        "precision_top1": float(precision),
        "fdr_top1": float(fdr),
        "correct_top1_scores": top1_scores[correct],
        "incorrect_top1_scores": top1_scores[~correct],
        "true_pair_scores": true_scores,
        "threshold_results": threshold_results,
        "unique_sequences": unique_sequences,
        "row_to_unique": row_to_unique,
        "top1_unique_ids": top1_unique_ids,
        "top1_sequences": predicted_sequences,
        "true_sequences": modified_sequences,
        "correct_mask": correct,
    }

    print("=" * 60)
    print("        FDR EVALUATION (UNIQUE MODIFIED SEQUENCES)")
    print("=" * 60)
    print(f"  Total spectra:         {n_rows}")
    print(f"  Unique peptides:       {n_unique}")
    print(f"  Correct Top-1:         {tp} ({precision * 100:.2f}%)")
    print(f"  Incorrect Top-1:       {fp}")
    print(f"  Precision (Top-1):     {precision:.4f}")
    print(f"  FDR (Top-1):           {fdr:.4f}")
    print()
    print(
        "  Score distribution (correct Top-1):   "
        f"mean={_mean_or_nan(top1_scores[correct]):.4f}, "
        f"median={_median_or_nan(top1_scores[correct]):.4f}"
    )
    print(
        "  Score distribution (incorrect Top-1): "
        f"mean={_mean_or_nan(top1_scores[~correct]):.4f}, "
        f"median={_median_or_nan(top1_scores[~correct]):.4f}"
    )
    print()
    print("  FDR at score thresholds:")
    print(f"  {'Threshold':>10} {'Accepted':>10} {'Precision':>10} {'FDR':>10} {'% Kept':>10}")
    for row in threshold_results:
        if row["n_accepted"] > 0:
            print(
                f"  {row['threshold']:>10.2f} {row['n_accepted']:>10} "
                f"{row['precision']:>10.4f} {row['fdr']:>10.4f} "
                f"{row['fraction_accepted'] * 100:>9.1f}%"
            )
    print("=" * 60)

    return results


def _make_decoy_tokens(
    target_tokens: np.ndarray,
    pad_idx: int,
    strategy: str,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create one token-level decoy for each unique target peptide."""
    if strategy not in {"reverse-inner", "reverse", "shuffle"}:
        raise ValueError("strategy must be 'reverse-inner', 'reverse', or 'shuffle'")

    rng = np.random.default_rng(seed)
    decoys = np.array(target_tokens, copy=True)
    keep = np.ones(target_tokens.shape[0], dtype=bool)

    for i in range(target_tokens.shape[0]):
        valid = target_tokens[i] != pad_idx
        residues = target_tokens[i, valid]
        n_res = int(residues.shape[0])
        if n_res <= 1:
            keep[i] = False
            continue

        if strategy == "reverse-inner":
            if n_res <= 3:
                candidate = residues.copy()
            else:
                candidate = residues.copy()
                candidate[1:-1] = candidate[1:-1][::-1]
        elif strategy == "reverse":
            candidate = residues[::-1]
        else:
            candidate = residues.copy()

        if strategy == "shuffle" or np.array_equal(candidate, residues):
            candidate = residues.copy()
            shuffle_slice = slice(1, -1) if strategy == "reverse-inner" and n_res > 3 else slice(None)
            shuffle_residues = residues[shuffle_slice]
            for _ in range(16):
                permuted_inner = shuffle_residues[rng.permutation(len(shuffle_residues))]
                permuted = residues.copy()
                permuted[shuffle_slice] = permuted_inner
                if not np.array_equal(permuted, residues):
                    candidate = permuted
                    break

        if np.array_equal(candidate, residues):
            if strategy == "reverse-inner" and n_res > 3:
                candidate = residues.copy()
                candidate[1:-1] = np.roll(candidate[1:-1], 1)
            else:
                candidate = np.roll(residues, 1)

        if np.array_equal(candidate, residues):
            keep[i] = False
            continue

        decoys[i, valid] = candidate

    return decoys[keep], keep


@torch.no_grad()
def compute_tda_fdr(
    model_spec: nn.Module,
    model_pep: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modified_sequences: Optional[List[str]] = None,
    thresholds: Optional[np.ndarray] = None,
    chunk_size: int = 1024,
    decoy_strategy: str = "reverse-inner",
    pad_idx: Optional[int] = None,
    seed: Optional[int] = 0,
    normalize: bool = True,
) -> dict:
    """Estimate FDR with target-decoy competition over unique peptides.

    The target database is the set of unique ``modified_sequence`` values. One
    token-level reverse-inner decoy is generated per unique target peptide,
    then each spectrum is searched against the combined target+decoy database.
    FDR is estimated as:

    ``decoy top-1 hits / target top-1 hits``

    at each score threshold, with the usual target-decoy competition assumption
    that target and decoy databases have the same size.

    Args:
        model_spec: Spectrum encoder in eval mode.
        model_pep: Peptide encoder in eval mode.
        loader: DataLoader returning at least ``(specs, peps, pres)``.
        device: Torch device.
        modified_sequences: Per-row target peptide strings, usually
            ``test_df["modified_sequence"].tolist()`` or ``test_dataset.sequences``.
        thresholds: Score thresholds for the FDR curve. Defaults to
            ``np.arange(0.0, 1.0, 0.05)``.
        chunk_size: Number of spectra per target-decoy search chunk.
        decoy_strategy: ``"reverse-inner"`` keeps terminal residues fixed and
            reverses only internal residues. ``"reverse"`` and ``"shuffle"``
            are also available.
        pad_idx: Peptide padding token. Defaults to ``model_pep.aa_embed.padding_idx``
            when available, otherwise 0.
        seed: RNG seed used by shuffled fallback decoys.
        normalize: L2-normalise embeddings before cosine search.

    Returns:
        Dictionary containing top-1 target/decoy calls, threshold-level FDR,
        and per-spectrum q-values from sorted target-decoy competition.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.05)
    if pad_idx is None:
        pad_idx = int(getattr(getattr(model_pep, "aa_embed", None), "padding_idx", 0) or 0)

    model_spec.eval()
    model_pep.eval()

    all_spec_emb: list[torch.Tensor] = []
    all_pep_emb: list[torch.Tensor] = []
    all_pep_tokens: list[torch.Tensor] = []
    batch_sequences: list[str] = []

    for batch in loader:
        specs = batch[0].to(device)
        peps = batch[1].to(device)
        pres = batch[2].to(device)

        if modified_sequences is None and len(batch) >= 4:
            batch_sequences.extend([str(seq) for seq in batch[3]])

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                z_spec = model_spec(specs, pres)
                z_pep = model_pep(peps)
        else:
            z_spec = model_spec(specs, pres)
            z_pep = model_pep(peps)

        if normalize:
            z_spec = F.normalize(z_spec, dim=-1)
            z_pep = F.normalize(z_pep, dim=-1)

        all_spec_emb.append(z_spec.cpu().float())
        all_pep_emb.append(z_pep.cpu().float())
        all_pep_tokens.append(peps.cpu())

    spec_emb = torch.cat(all_spec_emb, dim=0).numpy().astype(np.float32, copy=False)
    pep_emb = torch.cat(all_pep_emb, dim=0).numpy().astype(np.float32, copy=False)
    pep_tokens = torch.cat(all_pep_tokens, dim=0).numpy()
    n_rows = spec_emb.shape[0]

    if modified_sequences is None:
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "sequences"):
            modified_sequences = list(dataset.sequences)
        elif hasattr(dataset, "modified_sequences"):
            modified_sequences = list(dataset.modified_sequences)
        elif hasattr(dataset, "df") and "modified_sequence" in dataset.df.columns:
            modified_sequences = dataset.df["modified_sequence"].tolist()
        elif batch_sequences:
            modified_sequences = batch_sequences
        else:
            raise ValueError(
                "Pass modified_sequences=test_df['modified_sequence'].tolist(), "
                "or make the loader return sequences as its fourth batch item."
            )

    modified_sequences = [str(seq) for seq in modified_sequences]
    if len(modified_sequences) != n_rows:
        raise ValueError(
            f"modified_sequences length ({len(modified_sequences)}) must match "
            f"encoded rows ({n_rows})"
        )

    unique_sequences, first_rows, row_to_unique = build_unique_peptide_db(modified_sequences)
    first_rows_arr = np.asarray(first_rows, dtype=np.int64)
    unique_target_emb = pep_emb[first_rows_arr]
    unique_target_tokens = pep_tokens[first_rows_arr]

    decoy_tokens, keep_mask = _make_decoy_tokens(
        unique_target_tokens,
        pad_idx=pad_idx,
        strategy=decoy_strategy,
        seed=seed,
    )
    kept_target_emb = unique_target_emb[keep_mask]
    kept_sequences = [seq for seq, keep_seq in zip(unique_sequences, keep_mask) if keep_seq]
    kept_uids = np.flatnonzero(keep_mask).astype(np.int64)

    if len(kept_sequences) == 0:
        raise ValueError("No valid decoys could be generated from the unique peptides")

    decoy_batches: list[torch.Tensor] = []
    token_batch_size = 2048
    for start in range(0, decoy_tokens.shape[0], token_batch_size):
        end = min(start + token_batch_size, decoy_tokens.shape[0])
        token_batch = torch.as_tensor(decoy_tokens[start:end], dtype=torch.long, device=device)
        z_decoy = model_pep(token_batch)
        if normalize:
            z_decoy = F.normalize(z_decoy, dim=-1)
        decoy_batches.append(z_decoy.cpu().float())

    decoy_emb = torch.cat(decoy_batches, dim=0).numpy().astype(np.float32, copy=False)
    combined_emb = np.vstack([kept_target_emb, decoy_emb]).astype(np.float32, copy=False)
    is_decoy_db = np.concatenate([
        np.zeros(len(kept_sequences), dtype=bool),
        np.ones(len(kept_sequences), dtype=bool),
    ])
    origin_uid_db = np.concatenate([kept_uids, kept_uids])

    top1_db_idx = np.empty(n_rows, dtype=np.int64)
    top1_scores = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        sims = spec_emb[start:end] @ combined_emb.T
        top_local = np.argmax(sims, axis=1)
        row_ids = np.arange(end - start)
        top1_db_idx[start:end] = top_local
        top1_scores[start:end] = sims[row_ids, top_local]

    top1_is_decoy = is_decoy_db[top1_db_idx]
    top1_origin_uid = origin_uid_db[top1_db_idx]
    top1_origin_sequence = [unique_sequences[int(uid)] for uid in top1_origin_uid]

    n_decoy_top1 = int(top1_is_decoy.sum())
    n_target_top1 = int((~top1_is_decoy).sum())
    fdr_top1 = min(1.0, n_decoy_top1 / n_target_top1) if n_target_top1 else 1.0

    threshold_results = []
    for threshold in thresholds:
        accepted = top1_scores >= threshold
        n_accepted = int(accepted.sum())
        target_hits = int((accepted & ~top1_is_decoy).sum())
        decoy_hits = int((accepted & top1_is_decoy).sum())
        fdr = min(1.0, decoy_hits / target_hits) if target_hits else (1.0 if decoy_hits else 0.0)
        threshold_results.append({
            "threshold": float(threshold),
            "n_accepted": n_accepted,
            "target_hits": target_hits,
            "decoy_hits": decoy_hits,
            "fdr": float(fdr),
            "fraction_accepted": float(n_accepted / n_rows),
        })

    order = np.argsort(-top1_scores)
    sorted_is_decoy = top1_is_decoy[order]
    sorted_scores = top1_scores[order]
    cum_decoys = np.cumsum(sorted_is_decoy)
    cum_targets = np.cumsum(~sorted_is_decoy)
    fdr_by_rank = np.divide(
        cum_decoys,
        np.maximum(cum_targets, 1),
        dtype=np.float64,
    )
    fdr_by_rank = np.minimum(fdr_by_rank, 1.0)
    qvalue_sorted = np.minimum.accumulate(fdr_by_rank[::-1])[::-1]
    qvalues = np.empty(n_rows, dtype=np.float64)
    qvalues[order] = qvalue_sorted

    results = {
        "n_total": int(n_rows),
        "n_unique_targets": int(len(unique_sequences)),
        "n_targets_with_decoys": int(len(kept_sequences)),
        "n_skipped_targets": int(len(unique_sequences) - len(kept_sequences)),
        "n_target_top1": n_target_top1,
        "n_decoy_top1": n_decoy_top1,
        "fdr_top1": float(fdr_top1),
        "top1_scores": top1_scores,
        "top1_is_decoy": top1_is_decoy,
        "top1_origin_uid": top1_origin_uid,
        "top1_origin_sequence": top1_origin_sequence,
        "qvalues": qvalues,
        "score_order": order,
        "sorted_scores": sorted_scores,
        "sorted_fdr": fdr_by_rank,
        "sorted_qvalues": qvalue_sorted,
        "threshold_results": threshold_results,
        "unique_target_sequences": unique_sequences,
        "target_sequences_with_decoys": kept_sequences,
        "row_to_unique": row_to_unique,
        "decoy_strategy": decoy_strategy,
    }

    print("=" * 60)
    print("        TARGET-DECOY FDR ESTIMATION")
    print("=" * 60)
    print(f"  Spectra searched:      {n_rows}")
    print(f"  Unique targets:        {len(unique_sequences)}")
    print(f"  Targets with decoys:   {len(kept_sequences)}")
    print(f"  Top-1 target hits:     {n_target_top1}")
    print(f"  Top-1 decoy hits:      {n_decoy_top1}")
    print(f"  Estimated FDR top-1:   {fdr_top1:.4f}")
    print()
    print("  FDR at score thresholds:")
    print(f"  {'Threshold':>10} {'Accepted':>10} {'Targets':>10} {'Decoys':>10} {'FDR':>10} {'% Kept':>10}")
    for row in threshold_results:
        if row["n_accepted"] > 0:
            print(
                f"  {row['threshold']:>10.2f} {row['n_accepted']:>10} "
                f"{row['target_hits']:>10} {row['decoy_hits']:>10} "
                f"{row['fdr']:>10.4f} {row['fraction_accepted'] * 100:>9.1f}%"
            )
    print("=" * 60)

    return results


__all__ = [
    "extract_embeddings",
    "build_unique_peptide_db",
    "retrieve_batch",
    "retrieve_single",
    "evaluate_recall",
    "candidate_recall",
    "compute_fdr",
    "compute_tda_fdr",
    "evaluate_top_k_retrieval",
]
