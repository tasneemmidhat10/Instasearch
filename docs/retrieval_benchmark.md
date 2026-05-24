# Retrieval Benchmark — `benchmarks.py` + `retrieval_benchmark_hela_brodae.ipynb`

This pair (one library module + one driver notebook) is the **evaluation
harness** for the trained dual encoders. It reproduces the workflow originally
prototyped in `Data - Copy/` (Gabriel's analysis) using the production
`src/retrieval/` modules, on the two reference datasets: **HeLa** (human, large)
and **Brodae** (a smaller in-house set).

| File | Role |
|------|------|
| `src/retrieval/benchmarks.py` | Library — configs, loaders, encoders, Stage-1 retrieval, FDR estimators, neural rescoring, Percolator export. Pure functions, no I/O side effects beyond writing the requested artefacts under `bench_cfg.out_dir`. |
| `notebooks/retrieval_benchmark_hela_brodae.ipynb` | Thin driver — sets paths, instantiates the encoders, calls the entry points in order, plots results. The only cells you normally edit are §2 (paths) and §3 (checkpoint paths). |

---

## What the benchmark *does*, in order

```
PSM CSV ─┐
         ├─► load_psm_table ──► confident GT (q ≤ 0.01)
         │
Digest CSV ─► load_target_peptides ──► target peptide DB
Decoy FASTA ─► load_decoy_peptides ──► external decoy DB
mzML / MGF  ─► load_ms_spectra ──► (file, scan) → spectrum index
         │
         ├─► join_psm_with_spectra ──► SpectraPeptideDataset (model-ready)
         │
         └─► build_database_index ──► HNSW over (targets ∪ decoys)
                       │
                       ├─► extract_embeddings (spectra)
                       ├─► retrieve_batch ──► Recall@k
                       ├─► compute_fdr            (top-1 precision)
                       ├─► compute_tda_fdr        (internal token-decoys)
                       ├─► compute_external_tda_fdr (Gabriel external decoys)
                       ├─► rescore_and_rerank     (InstaNovo decoder, optional)
                       └─► export_percolator_tsv + run_percolator (optional)
```

Each arrow corresponds to one function in `benchmarks.py`; each block in the
notebook calls one of them.

---

## `src/retrieval/benchmarks.py` — module reference

### 1. Config (`BenchmarkConfig`, presets)
- `BenchmarkConfig` dataclass — every path and knob for a single dataset run.
  Carries `psm_csv`, `digest_csv`, `decoy_source`, `ms_files`, `fixed_mods`,
  `var_mod_map`, `psm_qvalue_cutoff` (default 0.01), `k_eval`, rescoring
  hyperparams, output dir, optional `percolator_exe`.
- `HELA_PRESET(data_root, ms_files)` / `BRODAE_PRESET(data_root, ms_files)` —
  fill the defaults that match the filenames in `Data - Copy/` (e.g.
  `helaqc_PSM.csv`, `Hela trypsin digest.csv`,
  `human_extended_normalized.fasta`). To run on the GCP VM you change **only**
  `data_root` and `ms_files`; everything else carries over.

### 2. Modification formatting (Sequest → UNIMOD)
- `format_modified_sequence(sequence, modifications, fixed_mods, var_mod_map)`
  — converts Sequest-style `Sequence + Modifications` (e.g.
  `"DMHGVTSHLSNQELQDLVEFMK"`, `"M2(Oxidation)"`) into the UNIMOD-tagged form
  the peptide encoder expects (`"DM[UNIMOD:35]HGVTSHLSNQELQDLVEFMK"`).
  - Variable PTMs come from `var_mod_map` (defaults cover Oxidation,
    Carbamidomethyl, Phospho, Acetyl, Deamidated). Unknown names are silently
    skipped to avoid poisoning the whole sequence.
  - `fixed_mods="C[UNIMOD:4]"` injects `[UNIMOD:4]` after every cysteine.

### 3. CSV loaders
- `_normalise_col(df, candidates)` — case- and whitespace-insensitive column
  lookup. Lets the same code consume CSVs with slightly different headers
  (e.g. `"m/z [Da]"` vs `"m/z"` vs `"Precursor m/z"`).
- `load_psm_table(psm_csv, qvalue_cutoff, fixed_mods, var_mod_map)` — parses
  Proteome Discoverer / Sequest PSM output into a normalised DataFrame
  `[scan_id, ms_file, sequence, modified_sequence, charge, mz, qvalue]`.
  Drops rows with no scan / no m/z, applies the q-value cut-off, and stems
  `Spectrum File` so it matches the `.mzML` / `.mgf` basenames.
- `load_target_peptides(digest_csv, fixed_mods)` — reads a tryptic-digest CSV
  and returns `(unique_peptides, peptide_to_proteins)`. Drops peptides
  containing **U / O / X** (selenocysteine, pyrrolysine, ambiguous), applies
  the fixed Carbamidomethyl, deduplicates. Up front it does
  `dropna(subset=[col_pep, col_prot])` to skip empty rows — `pandas
  .astype(str)` does **not** stringify `np.nan` on object dtype, so unguarded
  NaN rows would crash the `c in raw_seq` check.
- `load_decoy_peptides(source, fixed_mods, min_length, max_length)` — accepts
  either a CSV (one peptide per row) or a FASTA. For FASTAs it tryptic-digests
  (cleave after K/R), filters by length and again strips U/O/X.

### 4. Spectrum loaders / joiners
- `load_ms_spectra(ms_files)` — indexes MS2 spectra from `.mzML` (via
  `pyteomics.mzml`) or `.mgf` (via `pyteomics.mgf`), keyed by
  `(filename_stem, scan_number)`. Returns
  `{key: {mz_array, intensity_array, precursor_mz, charge}}`.
- `join_psm_with_spectra(psm_df, spectra_index, residue_set, ...)` — for each
  confident PSM, look up its spectrum, preprocess peaks
  (`preprocess_spectrum`), tokenise its peptide (`preprocess_peptide_residueset`),
  derive precursor `(mass, charge, mz)` matching InstaNovo's
  `(B, 3)` convention, and stack everything into a `SpectraPeptideDataset`.
  Also returns `kept_df` — the subset of PSMs that successfully joined, in the
  dataset's row order. Raises if **nothing** joined (usually means
  `ms_files` basenames don't match `Spectrum File` in the PSM CSV).

### 5. Index construction
- `build_database_index(model_pep, target_seqs, decoy_seqs, hnsw_cfg, ...)` —
  encodes targets ∪ decoys (decoys colliding with targets are dropped first,
  to keep top-k slots honest), stacks the embeddings, and builds a single
  `HNSWIndex` over both. Returns `(index, db_seqs, is_decoy_db)`; the boolean
  vector lets every downstream FDR routine count decoy hits.

### 6. Reports
- `RetrievalReport` — `name, n_queries, recall, stage1_candidates,
  stage1_ranks, rescored_candidates, rescored_ranks, spec_emb, elapsed_s`.
- `FDRReport` — `name, top1_fdr, tda_fdr` (the `tda_fdr` dict carries both
  internal-decoy and `"external"` keys when external decoys were used).

### 7. Entry points (orchestrators called by the notebook)
- `run_retrieval_benchmark(bench_cfg, model_spec, model_pep, residue_set,
  hnsw_cfg, device)` — runs everything from "read CSVs" through "Recall@k",
  also adds PSM-derived modified sequences that aren't in the digest CSV so
  the recall ceiling is well-defined. Returns the report **plus** the
  artefacts (`index, db_seqs, is_decoy_db, kept_df, dataset`) so the FDR /
  rescore / Percolator steps don't have to redo the heavy work.
- `run_fdr_benchmark(bench_cfg, model_spec, model_pep, dataset,
  modified_sequences, device, use_external_decoys, db_seqs, is_decoy_db)` —
  runs:
  1. `compute_fdr` — top-1 precision FDR over a unique-modified-sequence DB.
  2. `compute_tda_fdr` — Target-Decoy with **internal token-level** decoys
     (`reverse-inner` keeps terminal residues fixed, reverses the rest).
  3. `compute_external_tda_fdr` — Target-Decoy with **Gabriel's external**
     reversed-protein FASTA. Exact dot-product against the full DB
     (no HNSW approximation), because TDA needs an honest top-1.
- `compute_external_tda_fdr(...)` — re-encodes spectra **and** the DB, picks
  top-1 per spectrum, then both a sweep over score thresholds and per-row
  q-values via the standard decreasing-score cumulative formula
  (`min` over the suffix to enforce monotonicity).
- `rescore_benchmark(bench_cfg, retrieval, rescorer, dataset, kept_df)` —
  runs `rescore_and_rerank` on the top-N Stage-1 candidates per spectrum
  using the InstaNovo decoder under teacher forcing, prints the
  Stage-1 → rescored Δ for each k, and mutates the report
  (`rescored_candidates`, `rescored_ranks`).
- `export_percolator_tsv(retrieval, kept_df, db_seqs, is_decoy_db,
  peptide_to_proteins, out_tsv)` — writes a TSV in the exact format Gabriel's
  `df_percolator_sbrodae.tsv` uses (`id, is_decoy, scan_number, norm_preds,
  Protein_list, proteinId1..N`). `is_decoy = -1` for decoys, `1` for targets
  (Percolator convention). Top-1 per spectrum is taken from rescored
  candidates if rescoring was run, else from Stage-1.
- `run_percolator(perc_tsv, percolator_exe, out_dir)` — shells out to
  `percolator.exe` and captures its summary.

### Critical / non-obvious details

- **Precursor tensor convention.** `join_psm_with_spectra` builds
  `pres = (mass, charge, mz)` — the `(B, 3)` layout the
  `InstaSearchSpectrumEncoder` expects. `rescore_benchmark` then collapses it
  to the `(B, 2) = (mz, charge)` layout that `rescore_and_rerank` needs.
- **Recall ceiling.** Sequest can label PTMs the digest CSV doesn't enumerate.
  `run_retrieval_benchmark` therefore adds the GT modified sequences to the
  target DB before encoding. Without this, the achievable Recall@k caps below
  100 % for reasons unrelated to the model.
- **Why `dropna` + `astype(str)` together** in `load_target_peptides`: the
  `astype(str)` alone is unreliable for object-dtype Series carrying real
  `np.nan` — NaN can pass through unchanged and break the `c in raw_seq`
  check. The `dropna` up front is the real fix; the `astype(str)` + `.strip()`
  afterward is defence in depth for stray numeric IDs or whitespace cells.
- **Internal vs external decoys.** The two TDA flavours answer different
  questions: internal token-decoys test whether the model exploits the
  ordering of residues vs. their mere composition; external reversed-protein
  decoys mirror real database-search FDR control and are the right number to
  publish.

---

## `notebooks/retrieval_benchmark_hela_brodae.ipynb` — cell-by-cell

| § | Cell | What it does |
|---|------|--------------|
| 1 | Environment | Puts repo root and `InstaNovo/` on `sys.path`; selects CUDA if available. |
| 2 | Pick dataset / paths | Sets `DATA_ROOT`, `DATASET ∈ {"hela","brodae"}`, gathers `MS_FILES` via glob, instantiates `bench_cfg = HELA_PRESET(...)` or `BRODAE_PRESET(...)`. **The only block you edit when moving local ↔ GCP.** |
| 3 | InstaNovo + dual encoders | `InstaNovo.from_pretrained("instanovo-v1.2.0")` (with a `weights_only=False` shim around `torch.load`), builds `ResidueSet` from `InstaNovo/instanovo/configs/residues/default.yaml`, constructs `InstaSearchSpectrumEncoder` (frozen InstaNovo backbone + projection) and `PeptideEncoder` (transformer + CLS pool + projection), then loads checkpoints from `checkpoints/model_spec.pt` and `checkpoints/model_pep.pt` if present. |
| 4 | Stage-1 retrieval | Builds an `HNSWConfig(embed_dim=EMBED_DIM, M=32, ef_construction=200, ef_search=128, k_retrieve=100)` and calls `run_retrieval_benchmark`. Prints a Recall@k bar chart. |
| 5 | FDR experiments | Calls `run_fdr_benchmark` with `use_external_decoys=True`, reusing `db_seqs` and `is_decoy_db` from §4 so the external-decoy TDA reuses the already-built DB. |
| 6 | Neural rescoring (optional) | Updates the InstaNovo decoder's residue remapping to `LEGACY_PTM_TO_UNIMOD` and runs `rescore_benchmark`. Mutates `report` in place with `rescored_candidates` / `rescored_ranks`. |
| 7 | Percolator export (optional) | Re-derives `peptide_to_proteins`, calls `export_percolator_tsv`, and runs `percolator.exe` if it's configured and present. |
| 8 | Plots | Saves `recall_curve.png` (Stage-1 vs rescored Recall@k) and `tda_fdr.png` (top-1 cosine score vs q-value) under `bench_cfg.out_dir`. |

### Running it locally vs on GCP

The benchmark is path-agnostic — same code on both. To run on the VM:

1. Clone this repo into the VM.
2. Stage `Data - Copy/*.csv`, `*.fasta` and the `.mzML` / `.mgf` files in a
   GCS bucket; `gsutil cp` them onto the VM.
3. Edit §2:
   ```python
   DATA_ROOT = Path("/home/jupyter/Data")
   MS_FILES  = [Path("/home/jupyter/mzml/<filename>.mzML")]
   ```
4. Run the notebook top-to-bottom; outputs land in
   `Path("/home/jupyter/Data/benchmark_out/<dataset>/")`.

### Common failure modes

| Symptom | Cause | Fix |
|--------|-------|-----|
| `No PSM rows could be joined with the supplied MS files` | `Spectrum File` basenames in the PSM CSV don't match the `.mzML` / `.mgf` basenames. | Re-stem your MS files, or rename in the PSM CSV. |
| `KeyError: None of [...] present in columns: [...]` | The CSV uses a header `_normalise_col` doesn't know. | Add the variant to the relevant `_normalise_col(...)` candidate list. |
| Recall@k caps well below the trained-model expectation | The digest CSV is missing PTM forms Sequest assigned. | Already handled — `run_retrieval_benchmark` adds GT modified sequences to the DB. If you still see this, check that `format_modified_sequence` is mapping all of your `var_mod_map` entries (unknown PTM names get silently skipped). |
| `Argument of type 'float' is not iterable` in `load_target_peptides` | Empty rows in the digest CSV. | Already fixed via `dropna` up front; if it recurs, your CSV has empty *strings* — extend the strip-and-skip check. |
