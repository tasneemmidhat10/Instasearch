import os
import sys
import numpy as np
import pandas as pd

from src.models.peptide_encoder import AA_VOCAB
from ..utils.config import MAX_PEAKS, MAX_PEPTIDE_LEN

# Ensure InstaNovo is importable so we can pull PROTON_MASS_AMU and ResidueSet
# without forcing every caller to set sys.path themselves.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_INSTANOVO_PATH = os.path.join(_PROJECT_ROOT, "InstaNovo")
if _INSTANOVO_PATH not in sys.path:
    sys.path.insert(0, _INSTANOVO_PATH)

try:
    from instanovo.constants import PROTON_MASS_AMU
except Exception:  # pragma: no cover — fallback if InstaNovo is unavailable
    PROTON_MASS_AMU = 1.007276


def preprocess_spectrum(mz_array, intensity_array, max_peaks=MAX_PEAKS):
    if mz_array is None or intensity_array is None:
        return np.zeros((max_peaks, 2))

    mz_arr = np.array(mz_array, dtype=np.float64)
    int_arr = np.array(intensity_array, dtype=np.float64)

    if len(mz_arr) == 0 or len(int_arr) == 0:
        return np.zeros((max_peaks, 2))

    if len(mz_arr) != len(int_arr):
        min_len = min(len(mz_arr), len(int_arr))
        mz_arr = mz_arr[:min_len]
        int_arr = int_arr[:min_len]

    if len(mz_arr) > max_peaks:
        top_idx = np.argsort(int_arr)[-max_peaks:]
        top_idx = np.sort(top_idx)
        mz_arr, int_arr = mz_arr[top_idx], int_arr[top_idx]

    # InstaNovo's training-time intensity scaling: root then L2-normalise.
    int_arr = np.sqrt(int_arr)
    l2_norm = np.sqrt(np.sum(int_arr ** 2))
    int_arr_norm = int_arr / l2_norm if l2_norm > 0 else int_arr

    spectrum = np.zeros((max_peaks, 2))
    spectrum[:len(mz_arr), 0] = mz_arr
    spectrum[:len(int_arr), 1] = int_arr_norm
    return spectrum


def preprocess_peptide(sequence, max_len=MAX_PEPTIDE_LEN):
    """Legacy 26-AA tokenizer. Kept for back-compat with the existing notebook;
    new code should use `preprocess_peptide_residueset` (PTM-aware)."""
    if not isinstance(sequence, str) or len(sequence) == 0:
        return np.zeros(max_len, dtype=np.int64)

    sequence = sequence[:max_len]
    unk_idx = AA_VOCAB.get("X", 0)
    indices = [AA_VOCAB.get(aa, unk_idx) for aa in sequence]
    indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


def preprocess_peptide_residueset(sequence, residue_set, max_len=MAX_PEPTIDE_LEN):
    """ResidueSet-based tokenizer with UNIMOD PTM support.

    Returns an int64 numpy array of shape (max_len,), padded with
    `residue_set.PAD_INDEX`.
    """
    pad_idx = residue_set.PAD_INDEX
    if not isinstance(sequence, str) or len(sequence) == 0:
        return np.full(max_len, pad_idx, dtype=np.int64)

    tokens = residue_set.tokenize(sequence)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]

    encoded = residue_set.encode(tokens, return_tensor=None)  # numpy
    encoded = np.asarray(encoded, dtype=np.int64)

    out = np.full(max_len, pad_idx, dtype=np.int64)
    out[:len(encoded)] = encoded
    return out


def preprocess_dataset(df, residue_set=None, max_len=MAX_PEPTIDE_LEN,
                       max_peaks=MAX_PEAKS):
    """Preprocess a pandas DataFrame into model-ready tensors.

    Args:
        df: DataFrame with columns mz_array, intensity_array, sequence,
            precursor_mz, precursor_charge.
        residue_set: optional InstaNovo ResidueSet. If provided, peptides are
            tokenized via ResidueSet (PTM-aware). If None, falls back to the
            legacy 26-AA vocab (no PTMs).
        max_len: peptide max length.
        max_peaks: spectrum max peaks.

    Returns:
        spec_tensors: (N, max_peaks, 2)  float64
        pep_tokens:   (N, max_len)       int64
        precursors:   (N, 3)             float64 — [mass, charge, mz]
                      (InstaNovo convention).
    """
    required_cols = ['mz_array', 'intensity_array', 'sequence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    n = len(df)
    spec_tensors = np.zeros((n, max_peaks, 2))
    pad_idx = residue_set.PAD_INDEX if residue_set is not None else 0
    pep_tokens = np.full((n, max_len), pad_idx, dtype=np.int64)
    precursors = np.zeros((n, 3))  # [mass, charge, mz] — InstaNovo convention

    valid_indices = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            seq = row['sequence']
            mz = row['mz_array']
            intensity = row['intensity_array']

            if not isinstance(seq, str) or len(seq) == 0:
                continue
            if not isinstance(mz, (list, np.ndarray)) and pd.isna(mz):
                continue
            if not isinstance(intensity, (list, np.ndarray)) and pd.isna(intensity):
                continue

            spec_tensors[i] = preprocess_spectrum(mz, intensity, max_peaks=max_peaks)

            if residue_set is not None:
                pep_tokens[i] = preprocess_peptide_residueset(seq, residue_set,
                                                              max_len=max_len)
            else:
                pep_tokens[i] = preprocess_peptide(seq, max_len=max_len)

            charge_val = row.get('precursor_charge', 0)
            if charge_val is None or (isinstance(charge_val, float) and np.isnan(charge_val)):
                charge_val = 0
            mz_val = row.get('precursor_mz', 0.0) or 0.0
            mass_val = mz_val * charge_val - charge_val * PROTON_MASS_AMU
            precursors[i] = [mass_val, charge_val, mz_val]

            valid_indices.append(i)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    return spec_tensors[valid_indices], pep_tokens[valid_indices], precursors[valid_indices]
