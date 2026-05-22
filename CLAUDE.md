# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**InstaSearch** — a dual-encoder retrieval + neural rescoring pipeline for mass-spec proteomics.
A spectrum encoder (built around a frozen pretrained **InstaNovo** transformer backbone) and a
peptide encoder (small transformer over a PTM-aware `ResidueSet` vocabulary) are trained jointly
with a contrastive objective so that matching (spectrum, peptide) pairs are close in a shared L2-
normalised embedding space. At inference, an HNSW (FAISS) index over peptide embeddings retrieves
top-k candidates per spectrum, which can then be neurally rescored by running InstaNovo's decoder
under teacher forcing (Casanovo-DB–style PSM score).

## Common commands

The project uses **Hydra** for configuration and a `src/` layout (package = `instasearch`,
version pinned to Python `>=3.10,<3.14`).

```powershell
# Install (dev + InstaNovo extras both usually needed locally)
pip install -e ".[dev,instanovo]"

# Train with default CLIP contrastive loss (configs/dual_encoder.yaml)
python train_script.py

# Swap loss / dataset / any field via Hydra CLI overrides
python train_script.py loss=align_uniform
python train_script.py loss=align_uniform loss.use_decoy=true
python train_script.py dataset.split='train[:2000]' optim.num_epochs=2 dataset.batch_size=64

# Lint / type-check (configured in pyproject.toml)
ruff check .
mypy src

# Tests
pytest                              # full suite
pytest path/to/test_file.py::test_x # single test
```

Checkpoints land in `./checkpoints/` (`checkpoint_epoch_N.pt`, `final_model.pt`,
`interrupted_checkpoint.pt` on Ctrl-C). Each checkpoint stores both encoders, the loss module's
state (the learnable temperature lives there for CLIP), optimizer state, history, and the resolved
Hydra config.

## Architecture overview

```
train_script.py            # Hydra entry point — composes configs/dual_encoder.yaml
configs/                   # Hydra config groups: model/ loss/ dataset/ residues/ optim/
InstaNovo/                 # VENDORED upstream InstaNovo package — DO NOT MODIFY.
                           # train_script.py and src/utils/instanovo_loader.py prepend
                           # this path to sys.path so `import instanovo...` resolves here.
src/
  models/
    insta_search_spectrum_encoder.py   # Frozen InstaNovo encoder + projection head + L2-norm
    peptide_encoder.py                 # Small transformer + CLS pooling + projection head
    joint_model.py                     # Shared TransformerEncoderBlock, ProjectionHead
  data/
    preprocess.py        # MS peak filtering/normalisation, peptide tokenisation via ResidueSet
    dataset.py           # SpectraPeptideDataset (specs, peps, pres, [decoy])
    decoys.py            # make_shuffled_decoys — used when loss.use_decoy=true
  training/
    train.py             # train_epoch / validate — handles AMP, grad clip, decoy batches
    loss.py              # CLIPContrastiveLoss (symmetric InfoNCE, learnable temp)
    align_uniform_loss.py# Wang & Isola align+uniform + VICReg variance + optional decoy margin
    evaluate.py          # evaluate_top_k_retrieval (full sim-matrix Top-K; train-time eval)
  retrieval/             # Two-stage pipeline (re-exported from src/retrieval/__init__.py)
    index.py             # HNSWIndex / HNSWConfig (FAISS IndexHNSWFlat, cosine via IP-on-L2)
    search.py            # extract_embeddings, build_unique_peptide_db, retrieve_batch/single,
                         # evaluate_recall, compute_fdr (unique-sequence), compute_tda_fdr
                         # (target-decoy with reverse-inner / reverse / shuffle decoys)
    rerank.py            # InstaNovo as PSM scorer: instanovo_score / rescore_and_rerank
                         # + neural-rescored TDA FDR + precursor_features helper
  utils/
    config.py            # Legacy module-level hyperparameter constants (see note below)
    instanovo_loader.py  # load_instanovo_backbone — wraps InstaNovo.from_pretrained
    constants.py, visualization.py
```

### Critical design facts (non-obvious from a single file)

- **Vendored InstaNovo.** `InstaNovo/` is a submodule-style copy of the upstream repo, used as
  the spectrum backbone *and* as the neural rescorer. `train_script.py`, `src/utils/instanovo_loader.py`,
  and `src/data/preprocess.py` each insert `InstaNovo/` into `sys.path` before importing
  `instanovo.*`. Treat that directory as read-only — modifications break upstream parity.

- **Precursor tensor convention.** The spectrum encoder expects `precursors` shaped `(B, 3) =
  [precursor_mass, precursor_charge, precursor_mz]` to match InstaNovo's convention.
  `InstaSearchSpectrumEncoder` clamps charge to `>=1` because `instanovo._encoder` indexes
  `charge_encoder` by `charge.int() - 1`.

- **L2 normalisation lives in the projection heads**, not at search time. Both encoders apply
  `F.normalize(..., p=2, dim=-1)` to their outputs, so the HNSW index uses
  `METRIC_INNER_PRODUCT` to get cosine similarity for free. Don't re-normalise downstream.

- **Peptide DB must be deduplicated before indexing.** Many spectra share the same peptide;
  inserting duplicates into HNSW wastes top-k slots and breaks ground-truth matching by row
  index. `build_unique_peptide_db` returns `(unique_sequences, first_occurrence_indices,
  row_to_uid)`. **Ground-truth matching after dedup is by sequence string, not row index** —
  `retrieve_batch`, `evaluate_recall`, `compute_fdr`, and `compute_tda_fdr` are all written
  this way.

- **TDA FDR** in `search.compute_tda_fdr` builds **one token-level decoy per unique target**
  (default strategy: `reverse-inner` — keeps terminal residues fixed, reverses internal ones),
  searches against the combined target+decoy DB, and estimates FDR = decoy hits / target hits
  at each score threshold plus per-spectrum q-values.

- **Hydra-instantiated losses.** `train_script.py` reads `cfg.loss`, strips `name` and
  `use_decoy` (non-loss fields), and calls `hydra.utils.instantiate(loss_cfg)`. The `_target_`
  field in each `configs/loss/*.yaml` is the import path of the loss class. Adding a new loss
  means dropping a yaml in `configs/loss/` with a `_target_` and matching the
  `(loss, acc, diagnostics)` 3-tuple return contract.

- **Decoy plumbing.** `loss.use_decoy=true` causes `train_script.py` to call
  `make_shuffled_decoys` and pass them as the dataset's fourth item. `_split_batch` in
  `training/train.py` then routes 4-tuples to `loss_fn(z_spec, z_pep, z_decoy=z_decoy)`.
  Losses that don't use decoys (e.g. `CLIPContrastiveLoss`) just ignore the kwarg.

- **Two config systems coexist.** `src/utils/config.py` holds the **legacy** module-level
  constants (`D_MODEL`, `EMBED_DIM`, `INIT_TEMP`, `MAX_PEPTIDE_LEN`, etc.) that the older
  notebooks and several `src/models/*.py` default arguments still depend on. The Hydra configs
  in `configs/` are the source of truth for **training runs** — they pass explicit values to
  encoder constructors, which override the legacy defaults. When changing hyperparameters,
  prefer editing the Hydra YAML over `src/utils/config.py`.

- **The notebooks under `notebooks/` are exploratory** — they predate the refactor into
  `src/` and `train_script.py`. The canonical training path is `train_script.py`; the
  retrieval / FDR / rescoring notebooks (`*_hnsw_pipeline.ipynb`,
  `two_stage_retrieval_rescoring.ipynb`, `dual_encoder_fdr.ipynb`) drive the
  `src/retrieval/*` modules.

## Known repository hygiene issues

- `README.md` contains **unresolved Git merge-conflict markers** (`<<<<<<<`, `=======`,
  `>>>>>>>`). The `src/README.md` block inside it is mostly stale (predates the
  `InstaSearchSpectrumEncoder` / Hydra refactor and references a non-existent `retreieval/`
  typo dir). If you touch the README, resolve the conflict and update the code samples to
  the current `train_script.py` flow.
- `pyproject.toml` and `requirements.txt` overlap. `pyproject.toml` is canonical; the
  `instanovo` extras are also pinned at top-level in `requirements.txt`.
- `OPEN_QUESTIONS.md` is currently empty.
