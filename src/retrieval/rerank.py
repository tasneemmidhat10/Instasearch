"""Neural rescoring with InstaNovo as a learned score function.

Re-purposes the pretrained InstaNovo de novo sequencing transformer as a
PSM scoring function, analogous to Casanovo-DB (Ananth & Sanders et al.,
*Bioinformatics* 2024).

Given a spectrum and a candidate peptide ``s_1, ..., s_n``, the decoder is
run under teacher forcing and per-residue log-probabilities
``log p(s_i | s_{<i}, spectrum)`` are read out and combined into a single
PSM score. Two reductions are supported:

- ``"mean"`` (default, Casanovo-DB style): the mean log-probability — i.e.
  the log of the geometric mean of per-residue probabilities. Penalises
  uncertain positions more harshly and was found to dramatically improve
  calibration in the database-search setting.
- ``"sum"``: the joint sequence log-probability.

Typical usage::

    from src.retrieval.rerank import (
        load_instanovo_rescorer,
        instanovo_score,
        rescore_and_rerank,
    )

    model, residue_set, _ = load_instanovo_rescorer("instanovo-v1.2.0", device)

    # Score a list of candidate peptides against ONE spectrum:
    scores = instanovo_score(
        model, candidates=["PEPTIDE", "PEPTIDR"],
        spectra=spec_tensor, precursors=prec_tensor,
    )

    # Two-stage reranking over the full test set:
    reranked, ranks = rescore_and_rerank(
        model, stage1_candidates, spec_tensor, pre_tensor,
        ground_truth_seqs, rescore_top_n=50,
    )
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.retrieval.search import build_unique_peptide_db
from src.utils.instanovo_loader import load_instanovo_backbone

from instanovo.transformer.model import InstaNovo
from instanovo.utils.residues import ResidueSet


PROTON_MASS_AMU: float = 1.00727647


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_instanovo_rescorer(
    checkpoint: str = "instanovo-v1.2.0",
    device:     torch.device | str = "cpu",
    *,
    freeze: bool = True,
) -> tuple[InstaNovo, ResidueSet, dict]:
    """Load a pretrained InstaNovo model for rescoring.

    Args:
        checkpoint: Model id for ``InstaNovo.from_pretrained`` (e.g.
            ``"instanovo-v1.2.0"``) or a local ``.ckpt`` path.
        device: Target device.
        freeze: If True, freeze all parameters and call ``.eval()``.

    Returns:
        ``(model, residue_set, config)``
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model, config, _ = load_instanovo_backbone(checkpoint, device, freeze=freeze)
    return model, model.residue_set, config


# ---------------------------------------------------------------------------
# Candidate encoding
# ---------------------------------------------------------------------------

def encode_candidates(
    candidates:  Iterable[str],
    residue_set: ResidueSet,
    device:      torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize, reverse, EOS-append, and pad a batch of candidate peptides.

    InstaNovo decodes right-to-left, so candidates must be reversed before
    teacher forcing

    Args:
        candidates:  Peptide strings to encode.
        residue_set: The model's residue vocabulary.
        device:      Target device for the returned tensors.

    Returns:
        peptides:      ``LongTensor (C, L)`` — padded with ``PAD_INDEX``.
        peptides_mask: ``BoolTensor (C, L)`` — ``True`` at PAD positions.
    """
    encoded = [
        residue_set.encode(
            residue_set.tokenize(p)[::-1],
            add_eos=True,
            return_tensor="pt",
        )
        for p in candidates
    ]
    if not encoded:
        raise ValueError("`candidates` must be non-empty")

    lengths  = torch.tensor([t.shape[0] for t in encoded], dtype=torch.long)
    peptides = torch.nn.utils.rnn.pad_sequence(
        encoded, batch_first=True, padding_value=residue_set.PAD_INDEX
    )
    L             = peptides.shape[1]
    peptides_mask = torch.arange(L, dtype=torch.long)[None, :] >= lengths[:, None]
    return peptides.to(device), peptides_mask.to(device)


# ---------------------------------------------------------------------------
# Core score function
# ---------------------------------------------------------------------------

@torch.no_grad()
def instanovo_score(
    model:      InstaNovo,
    candidates: list[str],
    *,
    spectra:           Optional[torch.Tensor] = None,
    precursors:        Optional[torch.Tensor] = None,
    spectra_mask:      Optional[torch.Tensor] = None,
    spectra_embedding: Optional[torch.Tensor] = None,
    reduction:         str = "mean",
) -> torch.Tensor:
    """A scoring function for candidate peptides against ONE spectrum.

    Provide either raw spectrum inputs (``spectra``, ``precursors``,
    optionally ``spectra_mask``) **or** a pre-computed encoder output
    (``spectra_embedding``) — useful when batching multiple spectra and
    reusing the same encoder output for multiple candidate lists.

    Args:
        model:             Pretrained ``InstaNovo`` (on target device, in eval mode).
        candidates:        List of C candidate peptide strings.
        spectra:           ``FloatTensor (1, P, 2)`` — peaks (m/z, intensity).
        precursors:        ``FloatTensor (1, 3)`` — ``[mass, charge, m/z]``.
        spectra_mask:      Optional ``BoolTensor (1, P)`` — ``True`` at padded peaks.
        spectra_embedding: Optional ``FloatTensor (1, T, d_model)`` — output of
            ``model._encoder`` / ``model._flash_encoder``. If provided, the
            encoder forward pass is skipped entirely.
        reduction:         ``"mean"`` (Casanovo-DB) or ``"sum"`` (joint log-prob).

    Returns:
        ``FloatTensor (C,)`` — one score per candidate, on CPU.
            Higher is better.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    if (spectra is None or precursors is None) and spectra_embedding is None:
        raise ValueError("Pass either (spectra, precursors) or spectra_embedding")
    if not candidates:
        raise ValueError("`candidates` must be non-empty")

    device = next(model.parameters()).device

    # 1. Encode the spectrum once (skipped when a precomputed embedding is given).
    if spectra_embedding is None:
        if model.use_flash_attention:
            spectra_embedding, spectra_mask = model._flash_encoder(
                spectra.to(device), precursors.to(device)
            )
        else:
            spectra_embedding, spectra_mask = model._encoder(
                spectra.to(device),
                precursors.to(device),
                spectra_mask.to(device) if spectra_mask is not None else None,
            )

    # 2. Encode candidates (reversed, EOS-terminated, padded).
    peptides, peptides_mask = encode_candidates(candidates, model.residue_set, device)
    C = peptides.shape[0]

    # 3. Replicate the spectrum embedding once per candidate.
    spec_emb = spectra_embedding.expand(C, -1, -1).contiguous()
    spec_msk = spectra_mask.expand(C, -1).contiguous() if spectra_mask is not None else None

    # 4. Teacher-forced decode → per-residue log-probabilities.
    if model.use_flash_attention:
        logits = model._flash_decoder(spec_emb, peptides, spec_msk, peptides_mask, add_bos=True)
    else:
        logits = model._decoder(spec_emb, peptides, spec_msk, peptides_mask, add_bos=True)

    log_probs = F.log_softmax(logits, dim=-1)
    seq_logp  = torch.gather(log_probs, -1, peptides.unsqueeze(-1)).squeeze(-1)
    seq_logp  = seq_logp.masked_fill(peptides_mask, 0.0)

    summed = seq_logp.sum(dim=-1)
    if reduction == "sum":
        return summed.cpu()
    valid = (~peptides_mask).sum(dim=-1).clamp(min=1).float()
    return (summed / valid).cpu()


# ---------------------------------------------------------------------------
# Two-stage reranking
# ---------------------------------------------------------------------------

def rescore_and_rerank(
    model:              InstaNovo,
    stage1_candidates:  List[List[Tuple[str, float]]],
    spec_tensor:        torch.Tensor,
    pre_tensor:         torch.Tensor,
    ground_truth_seqs:  List[str],
    rescore_top_n:      int,
    rescore_batch:      int = 1,
    reduction:          str = "mean",
    k_retrieve:         Optional[int] = None,
) -> Tuple[List[List[Tuple[str, float]]], np.ndarray]:
    """Rescore and rerank the top-N Stage-1 candidates with the neural scorer.

    For each query spectrum:
    1. Take the top-``rescore_top_n`` candidates from Stage 1.
    2. Score them with :func:`instanovo_score` (mean log-prob per residue).
    3. Sort by neural score — the reranked block replaces the original head.
    4. Append remaining Stage-1 candidates (rank > rescore_top_n) in their
       original cosine-similarity order, so Recall@k for k > rescore_top_n
       is never degraded.

    The precursor tensor ``pre_tensor`` uses the retrieval convention
    ``[precursor_mz, charge]`` and is converted internally to the InstaNovo
    convention ``[mass, charge, m/z]``.

    Args:
        model:              Pretrained ``InstaNovo`` (on target device, eval mode).
        stage1_candidates:  Output of :func:`~src.retrieval.search.retrieve_batch` —
            list of Q lists of ``(peptide_sequence, cosine_score)`` tuples.
        spec_tensor:        ``FloatTensor (Q, MAX_PEAKS, 2)`` preprocessed spectra.
        pre_tensor:         ``FloatTensor (Q, 2)`` — ``[precursor_mz, charge]``.
        ground_truth_seqs:  True peptide string for each of the Q queries.
        rescore_top_n:      Number of Stage-1 candidates to pass to the neural scorer.
            Must be ≤ len(stage1_candidates[i]) for all i.
        rescore_batch:      Spectra processed per outer loop iteration. Larger
            values amortise the encoder cost at the expense of GPU memory.
        reduction:          ``"mean"`` or ``"sum"`` — passed to
            :func:`instanovo_score`.
        k_retrieve:         Used as the "not-found" sentinel value in the returned
            ranks array. Defaults to ``max(len(c) for c in stage1_candidates)``.

    Returns:
        reranked_candidates: List of Q lists — ``(peptide_sequence, neural_score)``
            sorted by descending neural score for the rescored block, with the
            Stage-1 tail appended.
        rescored_ranks:      ``int32 (Q,)`` — 1-indexed rank of the true peptide
            after reranking. ``k_retrieve + 1`` means not found at all.
    """
    Q          = len(stage1_candidates)
    device     = next(model.parameters()).device
    k_sentinel = k_retrieve or max(len(c) for c in stage1_candidates)

    rescored_ranks      = np.full(Q, k_sentinel + 1, dtype=np.int32)
    reranked_candidates: List[Optional[List[Tuple[str, float]]]] = [None] * Q

    model.eval()

    for batch_start in tqdm(
        range(0, Q, rescore_batch),
        desc=f"Rescoring top-{rescore_top_n}",
        total=(Q + rescore_batch - 1) // rescore_batch,
    ):
        for i in range(batch_start, min(batch_start + rescore_batch, Q)):
            cands_all  = stage1_candidates[i]
            cands_top  = cands_all[:rescore_top_n]
            cands_tail = cands_all[rescore_top_n:]

            if not cands_top:
                reranked_candidates[i] = cands_all
                continue

            seqs_top = [s for s, _ in cands_top]

            # Convert precursor: [mz, charge] → [mass, charge, mz]
            pre_i  = pre_tensor[i]
            mz_i   = float(pre_i[0])
            ch_i   = float(pre_i[1])
            mass_i = (mz_i - PROTON_MASS_AMU) * max(ch_i, 1)
            spec_i       = spec_tensor[i].unsqueeze(0).to(device)
            precursors_i = torch.tensor([[mass_i, ch_i, mz_i]], dtype=torch.float32,
                                        device=device)

            scores_top = instanovo_score(
                model, seqs_top,
                spectra=spec_i, precursors=precursors_i,
                reduction=reduction,
            )

            order        = torch.argsort(scores_top, descending=True).numpy()
            reranked_top = [(seqs_top[j], float(scores_top[j])) for j in order]
            reranked_all = reranked_top + cands_tail
            reranked_candidates[i] = reranked_all

            gt_seq = ground_truth_seqs[i]
            for r, (seq, _) in enumerate(reranked_all, 1):
                if seq == gt_seq:
                    rescored_ranks[i] = r
                    break

    return reranked_candidates, rescored_ranks


# ---------------------------------------------------------------------------
# Convenience drivers
# ---------------------------------------------------------------------------

def precursor_features(
    precursor_mz:     float,
    precursor_charge: float,
) -> torch.Tensor:
    """Build the ``(1, 3)`` precursor tensor expected by InstaNovo: ``[mass, charge, m/z]``."""
    mass = (precursor_mz - PROTON_MASS_AMU) * max(precursor_charge, 1)
    return torch.tensor([[mass, precursor_charge, precursor_mz]], dtype=torch.float32)


def rescore_spectrum(
    model:            InstaNovo,
    spectrum_id:      str,
    spectrum_peaks:   torch.Tensor,
    precursor_mz:     float,
    precursor_charge: float,
    candidates:       list[str],
    *,
    reduction: str = "mean",
) -> pd.DataFrame:
    """Rescore one spectrum against a list of candidates.

    Args:
        spectrum_peaks: ``FloatTensor (P, 2)`` — already preprocessed peaks.

    Returns:
        ``pd.DataFrame`` with columns ``spectrum_id, peptide, score``
        sorted by descending score.
    """
    spectra    = spectrum_peaks.unsqueeze(0).float()
    precursors = precursor_features(precursor_mz, precursor_charge)
    scores     = instanovo_score(
        model, candidates,
        spectra=spectra, precursors=precursors, reduction=reduction,
    )
    return pd.DataFrame({
        "spectrum_id": spectrum_id,
        "peptide":     candidates,
        "score":       scores.tolist(),
    }).sort_values("score", ascending=False, ignore_index=True)


def rescore_psms_from_embeddings(
    model:   InstaNovo,
    payload: Mapping[str, dict],
    *,
    reduction: str = "mean",
) -> pd.DataFrame:
    """Rescore a dict of pre-computed encoder embeddings + candidates.

    ``payload`` shape::

        {
            "<spectrum_id>": {
                "embedding":      FloatTensor (T, d_model),
                "embedding_mask": BoolTensor (T,) | None,
                "candidates":     list[str],
            },
            ...
        }

    Returns:
        ``pd.DataFrame`` with columns ``spectrum_id, peptide, score``,
        sorted by ``(spectrum_id, -score)``.
    """
    device = next(model.parameters()).device
    rows: list[dict] = []
    for sid, item in payload.items():
        cands = item.get("candidates")
        if not cands:
            continue
        emb = item["embedding"].to(device).unsqueeze(0)
        msk = item.get("embedding_mask")
        if msk is not None:
            msk = msk.to(device).unsqueeze(0)
        scores = instanovo_score(
            model, list(cands),
            spectra_embedding=emb, spectra_mask=msk, reduction=reduction,
        )
        for pep, sc in zip(cands, scores.tolist(), strict=True):
            rows.append({"spectrum_id": sid, "peptide": pep, "score": sc})

    return pd.DataFrame(rows).sort_values(
        ["spectrum_id", "score"], ascending=[True, False], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Target-decoy FDR after neural rescoring
# ---------------------------------------------------------------------------

def _extract_modified_sequences(
    loader: DataLoader,
    modified_sequences: Optional[Sequence[str]],
    batch_sequences: Sequence[str],
) -> list[str]:
    """Resolve per-row modified peptide strings from args, dataset, or batches."""
    if modified_sequences is not None:
        return [str(seq) for seq in modified_sequences]

    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "sequences"):
        return [str(seq) for seq in dataset.sequences]
    if hasattr(dataset, "modified_sequences"):
        return [str(seq) for seq in dataset.modified_sequences]
    if hasattr(dataset, "df") and "modified_sequence" in dataset.df.columns:
        return [str(seq) for seq in dataset.df["modified_sequence"].tolist()]
    if batch_sequences:
        return [str(seq) for seq in batch_sequences]

    raise ValueError(
        "Pass modified_sequences=test_df['modified_sequence'].tolist(), "
        "or make the loader return modified sequences as its fourth batch item."
    )


def _make_reverse_inner_decoy_sequences(
    target_sequences: Sequence[str],
    residue_set: ResidueSet,
) -> tuple[list[str], np.ndarray]:
    """Generate strict reverse-inner decoys and a keep mask.

    Terminal tokens are preserved. Targets whose reverse-inner decoy is
    identical, overlaps the target database, or duplicates a previous decoy are
    skipped so the final target and decoy databases remain one-to-one.
    """
    target_set = set(target_sequences)
    seen_decoys: set[str] = set()
    decoys: list[str] = []
    keep = np.zeros(len(target_sequences), dtype=bool)

    for i, sequence in enumerate(target_sequences):
        tokens = residue_set.tokenize(sequence)
        if len(tokens) < 4:
            continue

        decoy_tokens = [tokens[0], *tokens[1:-1][::-1], tokens[-1]]
        decoy = residue_set.detokenize(decoy_tokens)
        if decoy == sequence or decoy in target_set or decoy in seen_decoys:
            continue

        keep[i] = True
        seen_decoys.add(decoy)
        decoys.append(decoy)

    return decoys, keep


def _encode_peptide_strings(
    sequences: Sequence[str],
    residue_set: ResidueSet,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode peptide strings into the fixed-width token format used by model_pep."""
    encoded = [
        residue_set.encode(
            residue_set.tokenize(seq)[:max_len],
            add_eos=False,
            return_tensor="pt",
            pad_length=max_len,
        )
        for seq in sequences
    ]
    if not encoded:
        raise ValueError("Cannot encode an empty peptide sequence list")
    return torch.stack(encoded, dim=0).long().to(device)


def _topk_desc(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k column indices and scores for each row, sorted descending."""
    k = min(k, scores.shape[1])
    if k == scores.shape[1]:
        idx = np.argsort(-scores, axis=1)
    else:
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(scores.shape[0])[:, None]
        idx = idx[row, np.argsort(-scores[row, idx], axis=1)]
    row = np.arange(scores.shape[0])[:, None]
    return idx, scores[row, idx]


def _instanovo_precursor_from_row(
    precursor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert loader precursor rows to InstaNovo [mass, charge, mz]."""
    values = precursor.detach().cpu().float().view(-1)
    if values.numel() >= 3:
        mass, charge, mz = float(values[0]), float(values[1]), float(values[2])
    elif values.numel() == 2:
        mz, charge = float(values[0]), float(values[1])
        mass = (mz - PROTON_MASS_AMU) * max(charge, 1.0)
    else:
        raise ValueError(f"Expected precursor row with 2 or 3 values, got {values.numel()}")
    return torch.tensor([[mass, charge, mz]], dtype=torch.float32, device=device)


def _tda_fdr(decoy_hits: int, target_hits: int, decoy_factor: float = 1.0) -> float:
    """Target-decoy FDR estimate with clipping to [0, 1]."""
    if target_hits == 0:
        return 1.0 if decoy_hits else 0.0
    return float(min(1.0, decoy_factor * decoy_hits / target_hits))


@torch.no_grad()
def compute_neural_tda_fdr(
    model_spec: nn.Module,
    model_pep: nn.Module,
    rescorer_model: InstaNovo,
    loader: DataLoader,
    device: torch.device | str,
    *,
    modified_sequences: Optional[Sequence[str]] = None,
    peptide_residue_set: Optional[ResidueSet] = None,
    stage1_top_k: int = 100,
    retrieval_chunk_size: int = 256,
    peptide_batch_size: int = 2048,
    thresholds: Optional[Sequence[float]] = None,
    score_mode: str = "geometric_mean",
    normalize: bool = True,
    return_candidates: bool = False,
) -> dict:
    """Estimate final-pipeline FDR with neural target-decoy rescoring.

    Pipeline:
    1. Deduplicate the test/database peptides by ``modified_sequence``.
    2. Generate one strict reverse-inner decoy per usable unique target.
    3. Search spectra against the combined target+decoy DB with Stage-1 cosine metric.
    4. Rescore each spectrum's top-k candidates using :func:`instanovo_score`
       with ``reduction="mean"``: the scoring functionmean per-residue log-prob.
    5. Take the neural top-1 and estimate FDR as decoy hits / target hits.

    ``score_mode="geometric_mean"`` exponentiates the mean log-probability to
    give the geometric mean per-residue probability in ``[0, 1]``. Ranking is
    identical to using ``"mean_logp"`` because exp is monotonic.
    """
    if stage1_top_k < 1:
        raise ValueError("stage1_top_k must be >= 1")
    if score_mode not in {"geometric_mean", "mean_logp"}:
        raise ValueError("score_mode must be 'geometric_mean' or 'mean_logp'")

    device = torch.device(device) if not isinstance(device, torch.device) else device
    rescorer_device = next(rescorer_model.parameters()).device
    residue_set = peptide_residue_set or rescorer_model.residue_set

    model_spec.eval()
    model_pep.eval()
    rescorer_model.eval()

    spec_emb_batches: list[torch.Tensor] = []
    pep_emb_batches: list[torch.Tensor] = []
    spec_batches: list[torch.Tensor] = []
    precursor_batches: list[torch.Tensor] = []
    batch_sequences: list[str] = []
    peptide_width: Optional[int] = None

    for batch in tqdm(loader, desc="Encoding spectra/targets"):
        specs = batch[0].to(device)
        peps = batch[1].to(device)
        pres = batch[2].to(device)
        if peptide_width is None:
            peptide_width = int(peps.shape[1])

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

        spec_emb_batches.append(z_spec.cpu().float())
        pep_emb_batches.append(z_pep.cpu().float())
        spec_batches.append(batch[0].cpu().float())
        precursor_batches.append(batch[2].cpu().float())

    spec_emb = torch.cat(spec_emb_batches, dim=0).numpy().astype(np.float32, copy=False)
    pep_emb = torch.cat(pep_emb_batches, dim=0).numpy().astype(np.float32, copy=False)
    spec_tensor = torch.cat(spec_batches, dim=0)
    precursor_tensor = torch.cat(precursor_batches, dim=0)
    original_n_rows = spec_emb.shape[0]
    if peptide_width is None:
        raise ValueError("Loader produced no batches")

    sequences = _extract_modified_sequences(loader, modified_sequences, batch_sequences)
    if len(sequences) != original_n_rows:
        raise ValueError(
            f"modified_sequences length ({len(sequences)}) must match encoded rows ({original_n_rows})"
        )

    unique_targets, first_rows, row_to_unique = build_unique_peptide_db(sequences)
    first_rows_arr = np.asarray(first_rows, dtype=np.int64)
    unique_target_emb = pep_emb[first_rows_arr]

    decoy_sequences, keep_mask = _make_reverse_inner_decoy_sequences(unique_targets, residue_set)
    kept_uids = np.flatnonzero(keep_mask).astype(np.int64)
    target_sequences = [unique_targets[int(uid)] for uid in kept_uids]
    target_emb = unique_target_emb[keep_mask]

    if len(decoy_sequences) == 0:
        raise ValueError("No valid reverse-inner decoys could be generated")

    analysis_mask = np.isin(row_to_unique, kept_uids)
    original_row_indices = np.flatnonzero(analysis_mask).astype(np.int64)
    n_excluded_no_decoy = int(original_n_rows - original_row_indices.size)
    if original_row_indices.size == 0:
        raise ValueError("No spectra remain after enforcing 1:1 target-decoy pairing")
    if n_excluded_no_decoy:
        print(
            f"  Excluding {n_excluded_no_decoy:,} spectra whose target peptide "
            "did not yield a valid 1:1 reverse-inner decoy"
        )

    spec_emb = spec_emb[analysis_mask]
    spec_tensor = spec_tensor[analysis_mask]
    precursor_tensor = precursor_tensor[analysis_mask]
    sequences = [sequences[int(i)] for i in original_row_indices]
    row_to_unique = row_to_unique[analysis_mask]
    n_rows = spec_emb.shape[0]

    decoy_tokens = _encode_peptide_strings(decoy_sequences, residue_set, peptide_width, device)
    num_embeddings = getattr(getattr(model_pep, "aa_embed", None), "num_embeddings", None)
    if num_embeddings is not None and int(decoy_tokens.max().item()) >= int(num_embeddings):
        raise ValueError(
            "Decoy token IDs exceed model_pep.aa_embed.num_embeddings. "
            "Pass the same ResidueSet used to train model_pep as peptide_residue_set."
        )

    decoy_emb_batches: list[torch.Tensor] = []
    for start in range(0, decoy_tokens.shape[0], peptide_batch_size):
        end = min(start + peptide_batch_size, decoy_tokens.shape[0])
        z_decoy = model_pep(decoy_tokens[start:end])
        if normalize:
            z_decoy = F.normalize(z_decoy, dim=-1)
        decoy_emb_batches.append(z_decoy.cpu().float())
    decoy_emb = torch.cat(decoy_emb_batches, dim=0).numpy().astype(np.float32, copy=False)

    db_emb = np.vstack([target_emb, decoy_emb]).astype(np.float32, copy=False)
    db_sequences = target_sequences + decoy_sequences
    db_is_decoy = np.concatenate([
        np.zeros(len(target_sequences), dtype=bool),
        np.ones(len(decoy_sequences), dtype=bool),
    ])
    db_origin_uid = np.concatenate([kept_uids, kept_uids])
    decoy_factor = len(target_sequences) / max(len(decoy_sequences), 1)
    top_k = min(stage1_top_k, db_emb.shape[0])

    final_db_idx = np.full(n_rows, -1, dtype=np.int64)
    final_scores = np.full(n_rows, -np.inf, dtype=np.float32)
    final_mean_logp = np.full(n_rows, -np.inf, dtype=np.float32)
    final_stage1_cosine = np.full(n_rows, np.nan, dtype=np.float32)
    candidate_records = [] if return_candidates else None

    for start in tqdm(range(0, n_rows, retrieval_chunk_size), desc="Stage-1 search + neural TDA"):
        end = min(start + retrieval_chunk_size, n_rows)
        sims = spec_emb[start:end] @ db_emb.T
        top_idx, top_cos = _topk_desc(sims, top_k)

        for local_i, row_i in enumerate(range(start, end)):
            cand_idx = top_idx[local_i]
            cand_cos = top_cos[local_i]
            cand_sequences = [db_sequences[int(j)] for j in cand_idx]

            spec_i = spec_tensor[row_i].unsqueeze(0).to(rescorer_device)
            precursor_i = _instanovo_precursor_from_row(precursor_tensor[row_i], rescorer_device)
            mean_logp = instanovo_score(
                rescorer_model,
                cand_sequences,
                spectra=spec_i,
                precursors=precursor_i,
                reduction="mean",
            ).numpy()

            neural_scores = np.exp(mean_logp) if score_mode == "geometric_mean" else mean_logp
            order = np.argsort(-neural_scores)
            best_pos = int(order[0])
            best_db_idx = int(cand_idx[best_pos])

            final_db_idx[row_i] = best_db_idx
            final_scores[row_i] = float(neural_scores[best_pos])
            final_mean_logp[row_i] = float(mean_logp[best_pos])
            final_stage1_cosine[row_i] = float(cand_cos[best_pos])

            if candidate_records is not None:
                candidate_records.append([
                    {
                        "db_index": int(cand_idx[pos]),
                        "sequence": db_sequences[int(cand_idx[pos])],
                        "is_decoy": bool(db_is_decoy[int(cand_idx[pos])]),
                        "stage1_cosine": float(cand_cos[pos]),
                        "mean_logp": float(mean_logp[pos]),
                        "score": float(neural_scores[pos]),
                    }
                    for pos in order
                ])

    final_is_decoy = db_is_decoy[final_db_idx]
    final_origin_uid = db_origin_uid[final_db_idx]
    final_sequences = [db_sequences[int(i)] for i in final_db_idx]
    final_peptide_length = np.asarray(
        [len(residue_set.tokenize(seq)) for seq in final_sequences],
        dtype=np.int32,
    )
    final_charge = np.asarray(
        [
            float(row.detach().cpu().float().view(-1)[1])
            for row in precursor_tensor
        ],
        dtype=np.float32,
    )

    target_hits_total = int((~final_is_decoy).sum())
    decoy_hits_total = int(final_is_decoy.sum())
    fdr_top1 = _tda_fdr(decoy_hits_total, target_hits_total, decoy_factor=decoy_factor)

    if thresholds is None:
        finite_scores = final_scores[np.isfinite(final_scores)]
        thresholds_arr = np.unique(np.quantile(finite_scores, np.linspace(0.0, 0.99, 21)))
    else:
        thresholds_arr = np.asarray(list(thresholds), dtype=np.float64)

    threshold_results = []
    for threshold in thresholds_arr:
        accepted = final_scores >= threshold
        n_accepted = int(accepted.sum())
        target_hits = int((accepted & ~final_is_decoy).sum())
        decoy_hits = int((accepted & final_is_decoy).sum())
        fdr = _tda_fdr(decoy_hits, target_hits, decoy_factor=decoy_factor)
        threshold_results.append({
            "threshold": float(threshold),
            "n_accepted": n_accepted,
            "target_hits": target_hits,
            "decoy_hits": decoy_hits,
            "fdr": fdr,
            "fraction_accepted": float(n_accepted / n_rows),
        })

    order = np.argsort(-final_scores)
    sorted_is_decoy = final_is_decoy[order]
    sorted_scores = final_scores[order]
    cum_decoys = np.cumsum(sorted_is_decoy)
    cum_targets = np.cumsum(~sorted_is_decoy)
    fdr_by_rank = np.array([
        _tda_fdr(int(d), int(t), decoy_factor=decoy_factor)
        for d, t in zip(cum_decoys, cum_targets)
    ])
    qvalue_sorted = np.minimum.accumulate(fdr_by_rank[::-1])[::-1]
    qvalues = np.empty(n_rows, dtype=np.float64)
    qvalues[order] = qvalue_sorted

    results = {
        "n_total": int(n_rows),
        "n_input_spectra": int(original_n_rows),
        "n_excluded_no_decoy": n_excluded_no_decoy,
        "n_unique_targets": int(len(unique_targets)),
        "n_targets_with_decoys": int(len(target_sequences)),
        "n_skipped_targets": int(len(unique_targets) - len(target_sequences)),
        "stage1_top_k": int(top_k),
        "score_mode": score_mode,
        "n_target_top1": target_hits_total,
        "n_decoy_top1": decoy_hits_total,
        "fdr_top1": float(fdr_top1),
        "final_scores": final_scores,
        "final_mean_logp": final_mean_logp,
        "final_stage1_cosine": final_stage1_cosine,
        "final_is_decoy": final_is_decoy,
        "final_sequences": final_sequences,
        "final_origin_uid": final_origin_uid,
        "final_charge": final_charge,
        "final_peptide_length": final_peptide_length,
        "true_sequences": sequences,
        "original_row_indices": original_row_indices,
        "qvalues": qvalues,
        "score_order": order,
        "sorted_scores": sorted_scores,
        "sorted_fdr": fdr_by_rank,
        "sorted_qvalues": qvalue_sorted,
        "threshold_results": threshold_results,
        "unique_target_sequences": unique_targets,
        "target_sequences_with_decoys": target_sequences,
        "decoy_sequences": decoy_sequences,
        "row_to_unique": row_to_unique,
        "target_score_distribution": final_scores[~final_is_decoy],
        "decoy_score_distribution": final_scores[final_is_decoy],
    }
    if candidate_records is not None:
        results["reranked_candidates"] = candidate_records

    print("=" * 72)
    print("        NEURAL TARGET-DECOY FDR (CASANOVO-DB MEAN LOG-PROB)")
    print("=" * 72)
    print(f"  Spectra searched:       {n_rows}")
    if n_excluded_no_decoy:
        print(f"  Spectra excluded:       {n_excluded_no_decoy}")
    print(f"  Unique targets:         {len(unique_targets)}")
    print(f"  Targets with decoys:    {len(target_sequences)}")
    print(f"  Stage-1 candidates:     top-{top_k}")
    print(f"  Final score mode:       {score_mode}")
    print(f"  Top-1 target hits:      {target_hits_total}")
    print(f"  Top-1 decoy hits:       {decoy_hits_total}")
    print(f"  Estimated FDR top-1:    {fdr_top1:.4f}")
    print()
    print("  FDR at final-score thresholds:")
    print(f"  {'Threshold':>12} {'Accepted':>10} {'Targets':>10} {'Decoys':>10} {'FDR':>10} {'% Kept':>10}")
    for row in threshold_results:
        if row["n_accepted"] > 0:
            print(
                f"  {row['threshold']:>12.5f} {row['n_accepted']:>10} "
                f"{row['target_hits']:>10} {row['decoy_hits']:>10} "
                f"{row['fdr']:>10.4f} {row['fraction_accepted'] * 100:>9.1f}%"
            )
    print("=" * 72)

    return results


__all__ = [
    "load_instanovo_rescorer",
    "encode_candidates",
    "instanovo_score",
    "rescore_and_rerank",
    "compute_neural_tda_fdr",
    "precursor_features",
    "rescore_spectrum",
    "rescore_psms_from_embeddings",
]
