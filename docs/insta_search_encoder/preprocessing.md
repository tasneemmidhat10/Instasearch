# Preprocessing contract

The frozen backbone was trained on a specific input distribution. If we feed it out-of-distribution tensors, the features degrade — the projection head can't recover what the backbone never saw. This page documents the preprocessing assumptions that `InstaSearchSpectrumEncoder.forward` relies on.

## What InstaNovo was trained on

- **Top 200 peaks per spectrum**, selected by intensity (highest intensities kept).
- **Root-scaled intensities**: `intensity <- sqrt(intensity)`.
- **L2-normalised intensities** after the root-scale, so each spectrum's intensity vector has unit norm.
- Spectra shorter than 200 peaks are zero-padded to `[MAX_PEAKS, 2]`.

Matching all three is how we keep the frozen encoder in-distribution.

## How our preprocessing implements it

### Config

[src/utils/config.py](../../src/utils/config.py):

```python
MAX_PEAKS = 200        # top-k cap matches InstaNovo training
MAX_PEPTIDE_LEN = 42
```

The notebook's inline config (cell `6814abeb`) mirrors this.

### `preprocess_spectrum`

[src/data/preprocess.py](../../src/data/preprocess.py) and the inline copy in notebook cell `23a7ba3b`:

```python
def preprocess_spectrum(mz_array, intensity_array, max_peaks=MAX_PEAKS):
    ...
    # 1. Drop to top-k peaks by intensity
    if len(mz_arr) > max_peaks:
        top_idx = np.argsort(int_arr)[-max_peaks:]
        top_idx = np.sort(top_idx)
        mz_arr, int_arr = mz_arr[top_idx], int_arr[top_idx]

    # 2. Root-scale, then L2-normalise
    int_arr = np.sqrt(int_arr)
    l2_norm = np.sqrt(np.sum(int_arr ** 2))
    int_arr_norm = int_arr / l2_norm if l2_norm > 0 else int_arr

    # 3. Zero-pad to [MAX_PEAKS, 2]
    spectrum = np.zeros((max_peaks, 2))
    spectrum[:len(mz_arr), 0] = mz_arr
    spectrum[:len(int_arr), 1] = int_arr_norm
    return spectrum
```

All three steps (top-k, root-scale + L2, zero-pad) line up with what InstaNovo saw during its pretraining.

## How the encoder consumes it

Inside `forward`, the zero-padding is reused as the padding signal:

```python
x_mask = (x.abs().sum(dim=-1) == 0)
```

A row where both m/z and scaled intensity are zero is treated as padded. That convention is consistent with step 3 above — real peaks always have a nonzero m/z even if their intensity were zero, so the combined `abs().sum` across the `(mz, intensity)` pair is a reliable padding indicator for our preprocessing output.

## Precursor inputs

`precursors` is the `[B, 2]` tensor of `(precursor_mz, precursor_charge)` pairs assembled in `preprocess_dataset`. The forward pass clamps `charge >= 1` because InstaNovo indexes its charge embedding by `charge.int() - 1`; a stored charge of 0 (used as a fallback when the column is missing or NaN) would otherwise produce index `-1`. The clamp pins fallback rows to the `charge=1` embedding rather than reading from an unintended slot.

## What happens if you skip this

- **No top-k cap or a larger cap**: the backbone sees longer sequences than it ever was trained on. It still runs, but attention patterns drift and the pooled features get noisier.
- **L2 only, no sqrt**: the intensity dynamic range is compressed differently than at training time. Strong peaks dominate more than the backbone expects.
- **Wrong padding convention**: if `x_mask` misaligns with actual padding, the backbone either attends to zero-vector "peaks" or masks out real signal. Either way the pooled features degrade.
