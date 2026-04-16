import numpy as np
import pandas as pd
from ..models.peptide_encoder import AA_VOCAB
from ..utils.config import MAX_PEAKS, MAX_PEPTIDE_LEN

def preprocess_spectrum(mz_array, intensity_array, max_peaks=MAX_PEAKS):
    # Input Validation
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

    # Peak Filtering
    if len(mz_arr) > max_peaks:
        top_idx = np.argsort(int_arr)[-max_peaks:]
        top_idx = np.sort(top_idx)
        mz_arr, int_arr = mz_arr[top_idx], int_arr[top_idx]

    # Normalization
    l2_norm = np.sqrt(np.sum(int_arr ** 2))
    int_arr_norm = int_arr / l2_norm if l2_norm > 0 else int_arr

    # Create spectrum array [mz, intensity]
    spectrum = np.zeros((max_peaks, 2))
    spectrum[:len(mz_arr), 0] = mz_arr
    spectrum[:len(int_arr), 1] = int_arr_norm

    return spectrum

def preprocess_peptide(sequence, max_len=MAX_PEPTIDE_LEN):
    # Input Validation
    if not isinstance(sequence, str) or len(sequence) == 0:
        return np.zeros(max_len, dtype=np.int64)

    sequence = sequence[:max_len]
    indices = [AA_VOCAB.get(aa, 0) for aa in sequence]
    indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)

def preprocess_dataset(df):
    # Required columns check
    required_cols = ['mz_array', 'intensity_array', 'sequence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    n = len(df)
    spec_tensors = np.zeros((n, MAX_PEAKS, 2))
    pep_tokens = np.zeros((n, MAX_PEPTIDE_LEN), dtype=np.int64)
    precursors = np.zeros((n, 2)) # [mz, charge]

    valid_indices = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            # Safely check for missing values to avoid array truth value ambiguity and NaN issues
            seq = row['sequence']
            mz = row['mz_array']
            intensity = row['intensity_array']

            if not isinstance(seq, str) or len(seq) == 0:
                continue
            if not isinstance(mz, (list, np.ndarray)) and pd.isna(mz):
                continue
            if not isinstance(intensity, (list, np.ndarray)) and pd.isna(intensity):
                continue

            spec_tensors[i] = preprocess_spectrum(mz, intensity)
            pep_tokens[i] = preprocess_peptide(seq)
            charge_val = row.get('precursor_charge', 0)
            if charge_val is None or (isinstance(charge_val, float) and np.isnan(charge_val)):
                charge_val = 0
            precursors[i] = [row.get('precursor_mz', 0), charge_val]
            valid_indices.append(i)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    # Return only successfully processed rows
    return spec_tensors[valid_indices], pep_tokens[valid_indices], precursors[valid_indices]