"""Decoy peptide generation for AlignUniformLoss's optional margin term."""
import numpy as np


def make_shuffled_decoys(peps: np.ndarray, pad_idx: int = 0,
                         seed: int | None = None) -> np.ndarray:
    """Per-row shuffle of non-pad tokens to produce same-shape decoy peptides.

    Args:
        peps: int array of shape (N, L) — encoded peptide tokens, padded with
            `pad_idx` on the right.
        pad_idx: pad token index (positions == pad_idx are left untouched).
        seed: optional RNG seed for reproducibility.

    Returns:
        decoys: int array of shape (N, L). Each row's non-pad tokens are
            permuted (with replacement avoided where possible) so the
            resulting peptide differs from the original while keeping the
            same length.
    """
    rng = np.random.default_rng(seed)
    decoys = np.array(peps, copy=True)

    for i in range(decoys.shape[0]):
        valid = decoys[i] != pad_idx
        n = int(valid.sum())
        if n <= 1:
            continue

        non_pad = decoys[i, valid]
        # Try a few permutations to avoid the identity for diverse decoys.
        for _ in range(5):
            perm = rng.permutation(n)
            if not np.array_equal(perm, np.arange(n)):
                break
        decoys[i, valid] = non_pad[perm]

    return decoys
