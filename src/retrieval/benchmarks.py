"""Retrieval evaluation + FDR benchmarks on HeLa and Brodae datasets.

Reproduces the workflow in ``Data - Copy/`` (Gabriel's analysis):

    1. Load PSM CSV          → ground-truth (modified peptide, scan, charge, m/z).
    2. Load target peptides  ← trypsin-digest CSV.
    3. Load decoy peptides   ← Gabriel's FASTA / CSV (reversed protein digest).
    4. Load spectra          ← mzML / MGF, join by (Spectrum File, scan number).
    5. Encode + build HNSW   over target ∪ decoy peptide DB.
    6. Retrieve top-k        → Recall@k by modified_sequence string match.
    7. Top-1 FDR (precision) and Target-Decoy FDR (q-values).
    8. Optional neural rescoring (InstaNovo decoder, Casanovo-DB).
    9. Optional Percolator-format TSV export and ``percolator.exe`` invocation.

A single :class:`BenchmarkConfig` carries every path so the same notebook
runs locally and on a GCP VM (only ``data_root`` changes).
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import SpectraPeptideDataset
from src.data.preprocess import (
    PROTON_MASS_AMU,
    preprocess_peptide_residueset,
    preprocess_spectrum,
)
from src.retrieval.index import HNSWConfig, HNSWIndex
from src.retrieval.search import (
    compute_fdr,
    compute_tda_fdr,
    evaluate_recall,
    extract_embeddings,
    retrieve_batch,
)
from src.utils.config import MAX_PEAKS, MAX_PEPTIDE_LEN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """All paths and knobs for a retrieval-evaluation run on one dataset.

    Use :func:`HELA_PRESET` or :func:`BRODAE_PRESET` to fill the defaults
    from a single ``data_root`` directory; override fields as needed.
    """
    name:                str
    data_root:           Path
    psm_csv:             Path
    digest_csv:          Path
    decoy_source:        Path                # CSV (one peptide per row) or FASTA
    ms_files:            List[Path] = field(default_factory=list)
    spectra_csv:         Optional[Path] = None   # alt to ms_files: pre-dumped InstaNovo decoder CSV
    fixed_mods:          str        = "C[UNIMOD:4]"
    var_mod_map:         dict       = field(default_factory=lambda: {
        "Oxidation":         "[UNIMOD:35]",
        "Carbamidomethyl":   "[UNIMOD:4]",
        "Phospho":           "[UNIMOD:21]",
        "Acetyl":            "[UNIMOD:1]",
        "Deamidated":        "[UNIMOD:7]",
    })
    psm_qvalue_cutoff:   Optional[float] = 0.01
    k_eval:              List[int]       = field(default_factory=lambda: [1, 5, 10, 50, 100])
    rescore_top_n:       int             = 50
    rescore_batch:       int             = 32
    out_dir:             Path            = Path("./benchmark_out")
    percolator_exe:      Optional[Path]  = None

    def __post_init__(self) -> None:
        self.data_root    = Path(self.data_root)
        self.psm_csv      = Path(self.psm_csv)
        self.digest_csv   = Path(self.digest_csv)
        self.decoy_source = Path(self.decoy_source)
        self.ms_files     = [Path(p) for p in self.ms_files]
        self.out_dir      = Path(self.out_dir)
        if self.percolator_exe is not None:
            self.percolator_exe = Path(self.percolator_exe)
        if self.spectra_csv is not None:
            self.spectra_csv = Path(self.spectra_csv)


def HELA_PRESET(data_root: str | Path, ms_files: Sequence[str | Path] = ()) -> BenchmarkConfig:
    """HeLa preset using Gabriel's helaqc_* files in ``data_root``.

    Spectra come from ``helaqc_decoder_search_labelled.csv`` by default (each
    row already carries the full mz/intensity arrays); pass ``ms_files`` to
    override with raw mzML/MGF.
    """
    root = Path(data_root)
    return BenchmarkConfig(
        name         = "hela",
        data_root    = root,
        psm_csv      = root / "helaqc_PSM.csv",
        digest_csv   = root / "Hela trypsin digest.csv",
        decoy_source = root / "human_extended_normalized.fasta",
        ms_files     = list(ms_files),
        spectra_csv  = root / "helaqc_decoder_search_labelled.csv",
        out_dir      = root / "benchmark_out" / "hela",
        percolator_exe = root / "percolator.exe",
    )


def BRODAE_PRESET(data_root: str | Path, ms_files: Sequence[str | Path] = ()) -> BenchmarkConfig:
    """Brodae preset using Gabriel's sbrodae_* files in ``data_root``."""
    root = Path(data_root)
    return BenchmarkConfig(
        name         = "brodae",
        data_root    = root,
        psm_csv      = root / "sbrodae_PSM.csv",
        digest_csv   = root / "brodae trypsin digest.csv",
        decoy_source = root / "proteins_decoy_brodae.fasta",
        ms_files     = list(ms_files),
        out_dir      = root / "benchmark_out" / "brodae",
        percolator_exe = root / "percolator.exe",
    )


# ---------------------------------------------------------------------------
# Modification formatting (Sequest → UNIMOD)
# ---------------------------------------------------------------------------

_MOD_TOKEN_RE = re.compile(r"\s*([A-Z])(\d+)\(([^)]+)\)\s*")
_TERM_MOD_RE  = re.compile(r"\s*(N-Term|C-Term)\(([^)]+)\)\s*", re.IGNORECASE)


def format_modified_sequence(
    sequence:    str,
    modifications: str | float | None,
    fixed_mods:  str = "C[UNIMOD:4]",
    var_mod_map: Optional[dict] = None,
) -> str:
    """Convert Sequest ``Sequence + Modifications`` into a UNIMOD-tagged string.

    Example:
        ``("DMHGVTSHLSNQELQDLVEFMK", "M2(Oxidation)")`` →
        ``"DM[UNIMOD:35]HGVTSHLSNQELQDLVEFMK"``.

    The optional ``fixed_mods`` argument is applied to every residue whose
    letter appears in the rule (currently only Carbamidomethyl-C is special-
    cased — supplied as the literal token to inject, e.g. ``"C[UNIMOD:4]"``).
    """
    if not isinstance(sequence, str) or not sequence:
        return ""
    var_mod_map = var_mod_map or {}

    aas: List[str] = [aa for aa in sequence if aa.isalpha()]
    inserts: dict[int, str] = {}

    if isinstance(modifications, str) and modifications.strip():
        for m in _MOD_TOKEN_RE.finditer(modifications):
            _aa, pos_str, name = m.group(1), m.group(2), m.group(3)
            unimod = var_mod_map.get(name)
            if unimod is None:
                # Skip unknown PTMs silently — keeps unsupported tokens from
                # poisoning the whole sequence.
                continue
            pos = int(pos_str) - 1
            if 0 <= pos < len(aas):
                inserts[pos] = inserts.get(pos, "") + unimod

    # Fixed modification: rule like "C[UNIMOD:4]" → inject [UNIMOD:4] after every C.
    fixed_aa, fixed_tag = None, None
    if fixed_mods and "[" in fixed_mods:
        fixed_aa  = fixed_mods[0]
        fixed_tag = fixed_mods[1:]

    out: List[str] = []
    for i, aa in enumerate(aas):
        out.append(aa)
        if fixed_aa and aa == fixed_aa:
            out.append(fixed_tag)
        if i in inserts:
            out.append(inserts[i])
    return "".join(out)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _normalise_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first column in ``df`` matching any of ``candidates`` (case- and
    whitespace-insensitive). Raises if none found."""
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        hit = norm.get(cand.lower().strip())
        if hit is not None:
            return hit
    raise KeyError(f"None of {list(candidates)} present in columns: {list(df.columns)}")


def load_psm_table(
    psm_csv:         Path,
    qvalue_cutoff:   Optional[float] = 0.01,
    fixed_mods:      str = "C[UNIMOD:4]",
    var_mod_map:     Optional[dict] = None,
) -> pd.DataFrame:
    """Parse a Proteome Discoverer / Sequest PSM CSV.

    Returns a DataFrame with normalised columns:
    ``[scan_id, ms_file, sequence, modified_sequence, charge, mz, qvalue]``.

    If ``qvalue_cutoff`` is given, rows with ``Percolator q-Value > cutoff``
    are dropped (so the remaining rows are confident GT PSMs).
    """
    df = pd.read_csv(psm_csv, low_memory=False)

    col_seq    = _normalise_col(df, ["Sequence"])
    col_mods   = _normalise_col(df, ["Modifications"])
    col_charge = _normalise_col(df, ["Charge"])
    col_mz     = _normalise_col(df, ["m/z [Da]", "m/z", "Precursor m/z"])
    col_scan   = _normalise_col(df, ["First Scan", "Scan", "Scan Number"])
    col_file   = _normalise_col(df, ["Spectrum File", "RawFile", "File Name"])
    try:
        col_q  = _normalise_col(df, ["Percolator q-Value", "q-Value", "qvalue"])
    except KeyError:
        col_q  = None

    out = pd.DataFrame({
        "sequence":  df[col_seq].astype(str),
        "_mods_raw": df[col_mods].fillna(""),
        "charge":    pd.to_numeric(df[col_charge], errors="coerce").fillna(0).astype(int),
        "mz":        pd.to_numeric(df[col_mz],     errors="coerce").astype(float),
        "scan_id":   pd.to_numeric(df[col_scan],   errors="coerce").astype("Int64"),
        "ms_file":   df[col_file].astype(str).map(lambda s: Path(s).stem),
        "qvalue":    (pd.to_numeric(df[col_q], errors="coerce") if col_q else np.nan),
    })

    out = out.dropna(subset=["scan_id", "mz"]).reset_index(drop=True)
    out["scan_id"] = out["scan_id"].astype(int)

    out["modified_sequence"] = [
        format_modified_sequence(seq, mods, fixed_mods=fixed_mods, var_mod_map=var_mod_map)
        for seq, mods in zip(out["sequence"], out["_mods_raw"])
    ]
    out = out.drop(columns=["_mods_raw"])

    if qvalue_cutoff is not None and out["qvalue"].notna().any():
        before = len(out)
        out = out[out["qvalue"] <= qvalue_cutoff].reset_index(drop=True)
        print(f"  PSM q-value filter ({qvalue_cutoff}): {before:,} → {len(out):,}")

    print(f"  Loaded {len(out):,} PSM rows from {psm_csv.name}")
    return out


def load_target_peptides(
    digest_csv:  Path,
    fixed_mods:  str = "C[UNIMOD:4]",
) -> Tuple[List[str], dict[str, List[str]]]:
    """Load tryptic peptides + their protein origin from a digest CSV.

    Mirrors ``Data - Copy/Decoys.py``: drop peptides containing U/O/X, apply
    fixed Carbamidomethyl-C, deduplicate.

    Returns:
        unique_peptides:   List of modified peptide strings (UNIMOD format).
        peptide_to_proteins:  ``{peptide: [protein_id, ...]}`` map for the
                              Percolator-export step.
    """
    df = pd.read_csv(digest_csv, encoding="utf-8-sig")
    col_pep = _normalise_col(df, ["Trypsin Digest Sequences", "Peptide", "Sequence"])
    col_prot = _normalise_col(df, ["Protein Origin", "Protein", "ProteinId"])

    df = df.dropna(subset=[col_pep, col_prot]).reset_index(drop=True)

    peptide_to_proteins: dict[str, List[str]] = {}
    for raw_seq, prot in zip(df[col_pep].astype(str), df[col_prot].astype(str)):
        raw_seq = raw_seq.strip()
        if not raw_seq or any(c in raw_seq for c in "UOX"):
            continue
        mod_seq = format_modified_sequence(raw_seq, "", fixed_mods=fixed_mods)
        peptide_to_proteins.setdefault(mod_seq, []).append(prot)

    unique = list(peptide_to_proteins.keys())
    print(f"  Loaded {len(unique):,} unique target peptides from {digest_csv.name}")
    return unique, peptide_to_proteins


def load_decoy_peptides(
    source:        Path,
    fixed_mods:    str = "C[UNIMOD:4]",
    min_length:    int = 5,
    max_length:    int = 40,
) -> List[str]:
    """Load decoy peptides from a CSV or by re-digesting a decoy FASTA.

    For CSVs the function uses the first column whose name contains
    ``"pep"`` or ``"seq"`` (case-insensitive).

    For FASTAs the protein sequences are tryptic-digested (cleavage after
    K/R) and filtered by length, matching ``Data - Copy/Decoys.py``.
    """
    src = Path(source)
    if src.suffix.lower() in {".fasta", ".fa", ".faa"}:
        from Bio import SeqIO

        peps: set[str] = set()
        for record in SeqIO.parse(str(src), "fasta"):
            seq = str(record.seq)
            cur: List[str] = []
            for aa in seq:
                cur.append(aa)
                if aa in ("K", "R"):
                    peps.add("".join(cur))
                    cur = []
            if cur:
                peps.add("".join(cur))
        peps = {p for p in peps if min_length <= len(p) <= max_length
                and not any(c in p for c in "UOX")}
    else:
        df = pd.read_csv(src)
        col = next(
            (c for c in df.columns if "pep" in c.lower() or "seq" in c.lower()),
            df.columns[0],
        )
        peps = set(df[col].dropna().astype(str).tolist())
        peps = {p for p in peps if p and not any(c in p for c in "UOX")}

    out = [format_modified_sequence(p, "", fixed_mods=fixed_mods) for p in peps]
    print(f"  Loaded {len(out):,} decoy peptides from {src.name}")
    return out


def load_ms_spectra(ms_files: Sequence[Path]) -> dict[tuple[str, int], dict]:
    """Index MS/MS spectra from a list of mzML / MGF files.

    Keyed by ``(filename_stem, scan_number)``. Each value is
    ``{"mz_array": ndarray, "intensity_array": ndarray,
       "precursor_mz": float, "charge": int}``.
    """
    out: dict[tuple[str, int], dict] = {}
    if not ms_files:
        print("  load_ms_spectra: no MS files supplied — returning empty index")
        return out

    from pyteomics import mgf, mzml  # local import keeps optional dep optional

    for path in ms_files:
        path = Path(path)
        stem = path.stem
        ext = path.suffix.lower()
        if ext == ".mzml":
            it = mzml.read(str(path))
            for spec in tqdm(it, desc=f"Indexing {path.name}"):
                if int(spec.get("ms level", 0)) != 2:
                    continue
                scan = int(re.search(r"scan=(\d+)", spec["id"]).group(1))
                prec = spec["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
                charge = int(prec.get("charge state", 0))
                pre_mz = float(prec["selected ion m/z"])
                out[(stem, scan)] = {
                    "mz_array":        np.asarray(spec["m/z array"],         dtype=np.float64),
                    "intensity_array": np.asarray(spec["intensity array"],   dtype=np.float64),
                    "precursor_mz":    pre_mz,
                    "charge":          charge,
                }
        elif ext == ".mgf":
            for spec in tqdm(mgf.read(str(path)), desc=f"Indexing {path.name}"):
                params = spec.get("params", {})
                scan_str = str(params.get("scans") or params.get("title", ""))
                m = re.search(r"scan[=:_ ](\d+)", scan_str)
                if not m:
                    continue
                scan = int(m.group(1))
                pepmass = params.get("pepmass", (0.0,))[0]
                charge = params.get("charge", [0])
                charge = int(charge[0]) if charge else 0
                out[(stem, scan)] = {
                    "mz_array":        np.asarray(spec["m/z array"],       dtype=np.float64),
                    "intensity_array": np.asarray(spec["intensity array"], dtype=np.float64),
                    "precursor_mz":    float(pepmass or 0.0),
                    "charge":          charge,
                }
        else:
            raise ValueError(f"Unsupported MS file extension: {ext} ({path})")

    print(f"  Indexed {len(out):,} MS2 spectra from {len(ms_files)} file(s)")
    return out


def load_ms_spectra_from_csv(csv_path: Path) -> dict[tuple[str, int], dict]:
    """Build the (stem, scan) -> spectrum index from an InstaNovo decoder CSV.

    The decoder export stores one row per MS2 scan with ``mz_array`` and
    ``intensity_array`` already serialised as numpy-style bracketed strings,
    plus ``precursor_mz``, ``precursor_charge``, ``experiment_name``,
    ``scan_number``. Use this when you have the decoder dump but not the
    original .raw/.mzML.

    ``experiment_name`` is normalised: a trailing ``_PSM`` is stripped so the
    stem matches the ``Spectrum File`` column in the PSM CSV (``<basename>.raw``
    -> stem ``<basename>``).
    """
    csv_path = Path(csv_path)
    cols = [
        "scan_number", "mz_array", "intensity_array",
        "precursor_mz", "precursor_charge", "experiment_name",
    ]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    out: dict[tuple[str, int], dict] = {}
    for row in tqdm(df.itertuples(index=False), total=len(df),
                    desc=f"Parsing {csv_path.name}"):
        stem = str(row.experiment_name)
        if stem.endswith("_PSM"):
            stem = stem[:-4]
        scan = int(row.scan_number)
        mz   = np.fromstring(str(row.mz_array).strip("[]"),        sep=" ", dtype=np.float64)
        inten= np.fromstring(str(row.intensity_array).strip("[]"), sep=" ", dtype=np.float64)
        out[(stem, scan)] = {
            "mz_array":        mz,
            "intensity_array": inten,
            "precursor_mz":    float(row.precursor_mz),
            "charge":          int(row.precursor_charge),
        }
    print(f"  Indexed {len(out):,} MS2 spectra from {csv_path.name}")
    return out


def join_psm_with_spectra(
    psm_df:         pd.DataFrame,
    spectra_index:  dict[tuple[str, int], dict],
    residue_set,
    max_peaks:      int = MAX_PEAKS,
    max_len:        int = MAX_PEPTIDE_LEN,
) -> Tuple[SpectraPeptideDataset, pd.DataFrame]:
    """Join PSM rows to indexed spectra; return a model-ready dataset.

    Returns:
        dataset:    SpectraPeptideDataset of length M (≤ len(psm_df)).
        kept_df:    The PSM rows that successfully joined, in dataset order.
                    Carries ``modified_sequence``, ``scan_id``, ``ms_file``,
                    ``charge``, ``mz``, ``qvalue`` for downstream FDR/export.
    """
    keep_specs:   List[np.ndarray] = []
    keep_peps:    List[np.ndarray] = []
    keep_pres:    List[np.ndarray] = []
    keep_rows:    List[int]        = []

    missed = 0
    for idx, row in psm_df.iterrows():
        key = (row["ms_file"], int(row["scan_id"]))
        spec = spectra_index.get(key)
        if spec is None:
            missed += 1
            continue
        mz = float(row["mz"] or spec["precursor_mz"])
        charge = int(row["charge"] or spec["charge"] or 1)
        if charge <= 0:
            charge = max(int(spec["charge"]), 1)

        spec_arr = preprocess_spectrum(spec["mz_array"], spec["intensity_array"],
                                       max_peaks=max_peaks)
        pep_arr  = preprocess_peptide_residueset(row["modified_sequence"], residue_set,
                                                 max_len=max_len)
        mass = mz * charge - charge * PROTON_MASS_AMU
        pre_arr = np.array([mass, charge, mz], dtype=np.float32)

        keep_specs.append(spec_arr.astype(np.float32))
        keep_peps.append(pep_arr)
        keep_pres.append(pre_arr)
        keep_rows.append(idx)

    if not keep_rows:
        raise RuntimeError(
            "No PSM rows could be joined with the supplied MS files. "
            "Check that BenchmarkConfig.ms_files matches the 'Spectrum File' "
            "column in the PSM CSV (basename without extension)."
        )

    print(f"  Joined {len(keep_rows):,} PSMs to spectra  "
          f"({missed:,} missed — no matching scan in ms_files)")

    specs = np.stack(keep_specs)
    peps  = np.stack(keep_peps)
    pres  = np.stack(keep_pres)
    dataset = SpectraPeptideDataset(specs, peps, pres)
    kept_df = psm_df.loc[keep_rows].reset_index(drop=True)
    return dataset, kept_df


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_database_index(
    model_pep:    nn.Module,
    target_seqs:  List[str],
    decoy_seqs:   Optional[List[str]],
    hnsw_cfg:     HNSWConfig,
    residue_set,
    device:       torch.device,
    batch_size:   int = 1024,
    max_len:      int = MAX_PEPTIDE_LEN,
) -> Tuple[HNSWIndex, List[str], np.ndarray]:
    """Encode target ∪ decoy peptides and build a single HNSW index over both.

    Returns:
        index:        Fully-built :class:`HNSWIndex`.
        db_seqs:      Peptide string for every DB row (length T + D).
        is_decoy_db:  ``bool (T + D,)`` — True for decoys.
    """
    decoy_seqs = decoy_seqs or []
    # Decoys can collide with targets; drop those decoys first.
    target_set = set(target_seqs)
    decoy_seqs = [d for d in decoy_seqs if d not in target_set]

    db_seqs     = list(target_seqs) + list(decoy_seqs)
    is_decoy_db = np.concatenate([
        np.zeros(len(target_seqs), dtype=bool),
        np.ones(len(decoy_seqs),   dtype=bool),
    ])

    print(f"  Encoding {len(db_seqs):,} DB peptides  "
          f"({len(target_seqs):,} targets + {len(decoy_seqs):,} decoys) ...")
    model_pep.eval()
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(db_seqs), batch_size), desc="Encoding DB"):
            chunk = db_seqs[start:start + batch_size]
            tokens = np.stack([
                preprocess_peptide_residueset(s, residue_set, max_len=max_len)
                for s in chunk
            ])
            z = model_pep(torch.as_tensor(tokens, dtype=torch.int64, device=device))
            embeddings.append(z.cpu().float().numpy())
    db_emb = np.vstack(embeddings).astype(np.float32, copy=False)

    index = HNSWIndex(hnsw_cfg).build(db_emb)
    return index, db_seqs, is_decoy_db


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass
class RetrievalReport:
    name:                str
    n_queries:           int
    recall:              dict[int, float]
    stage1_candidates:   List[List[Tuple[str, float]]]
    stage1_ranks:        np.ndarray
    rescored_candidates: Optional[List[List[Tuple[str, float]]]] = None
    rescored_ranks:      Optional[np.ndarray] = None
    spec_emb:            Optional[np.ndarray] = None
    elapsed_s:           float = 0.0


@dataclass
class FDRReport:
    name:           str
    top1_fdr:       dict
    tda_fdr:        dict


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_retrieval_benchmark(
    bench_cfg:    BenchmarkConfig,
    model_spec:   nn.Module,
    model_pep:    nn.Module,
    residue_set,
    hnsw_cfg:     HNSWConfig,
    device:       torch.device,
) -> Tuple[RetrievalReport, HNSWIndex, List[str], np.ndarray, pd.DataFrame, SpectraPeptideDataset]:
    """End-to-end Stage-1 retrieval evaluation on a benchmark dataset.

    Returns the report plus the artefacts (index, db_seqs, is_decoy_db,
    joined PSM dataframe, dataset) so callers can chain rescoring / FDR /
    Percolator export without re-running expensive steps.
    """
    bench_cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n══ Retrieval benchmark: {bench_cfg.name} ══")
    psm_df = load_psm_table(
        bench_cfg.psm_csv,
        qvalue_cutoff=bench_cfg.psm_qvalue_cutoff,
        fixed_mods=bench_cfg.fixed_mods,
        var_mod_map=bench_cfg.var_mod_map,
    )
    target_seqs, _pep_to_prots = load_target_peptides(
        bench_cfg.digest_csv, fixed_mods=bench_cfg.fixed_mods,
    )
    decoy_seqs = load_decoy_peptides(bench_cfg.decoy_source, fixed_mods=bench_cfg.fixed_mods)

    # Add GT modified_sequences to the target DB so the recall ceiling is well-defined.
    # (Sequest can return PTMs the digest CSV doesn't enumerate.)
    gt_seqs = sorted(set(psm_df["modified_sequence"].tolist()))
    target_set = set(target_seqs)
    extra = [s for s in gt_seqs if s and s not in target_set]
    if extra:
        print(f"  Adding {len(extra):,} PSM-derived targets not in the digest")
        target_seqs = target_seqs + extra

    if bench_cfg.spectra_csv is not None and bench_cfg.spectra_csv.exists():
        spectra_index = load_ms_spectra_from_csv(bench_cfg.spectra_csv)
    else:
        spectra_index = load_ms_spectra(bench_cfg.ms_files)
    dataset, kept_df = join_psm_with_spectra(psm_df, spectra_index, residue_set)

    index, db_seqs, is_decoy_db = build_database_index(
        model_pep, target_seqs, decoy_seqs, hnsw_cfg, residue_set, device,
    )

    spec_emb = extract_embeddings(
        model_spec, dataset, mode="spectrum",
        batch_size=512, device=device, desc="Encoding query spectra",
    )

    t0 = time.time()
    candidates, ranks = retrieve_batch(
        spec_emb.astype(np.float32),
        kept_df["modified_sequence"].tolist(),
        index,
        db_seqs,
        k=hnsw_cfg.k_retrieve,
    )
    elapsed = time.time() - t0

    recall = {
        k: float((ranks <= k).mean())
        for k in bench_cfg.k_eval if k <= hnsw_cfg.k_retrieve
    }
    print("\n── Recall@k ─────────────────────────────────────────")
    for k, r in recall.items():
        bar = "█" * int(r * 40)
        print(f"  Recall@{k:<4d}: {r:.4f}  {bar}")
    print(f"  ({elapsed:.1f}s, {elapsed/len(ranks)*1000:.2f} ms/query)")

    report = RetrievalReport(
        name              = bench_cfg.name,
        n_queries         = len(ranks),
        recall            = recall,
        stage1_candidates = candidates,
        stage1_ranks      = ranks,
        spec_emb          = spec_emb,
        elapsed_s         = elapsed,
    )
    return report, index, db_seqs, is_decoy_db, kept_df, dataset


def run_fdr_benchmark(
    bench_cfg:    BenchmarkConfig,
    model_spec:   nn.Module,
    model_pep:    nn.Module,
    dataset:      SpectraPeptideDataset,
    modified_sequences: List[str],
    device:       torch.device,
    use_external_decoys: bool = True,
    db_seqs:      Optional[List[str]] = None,
    is_decoy_db:  Optional[np.ndarray] = None,
) -> FDRReport:
    """Run both FDR estimates the way ``src.retrieval.search`` defines them.

    - ``compute_fdr``       : top-1 precision FDR over unique-modified-sequence DB.
    - ``compute_tda_fdr``   : target-decoy competition with internal token-level decoys.

    If ``use_external_decoys=True`` and ``db_seqs/is_decoy_db`` are provided
    (from :func:`build_database_index`), an additional TDA estimate using
    Gabriel's external decoy DB is computed via
    :func:`compute_external_tda_fdr` (sequence-level competition over the
    target+decoy peptide DB already in memory).
    """
    bench_cfg.out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=512, num_workers=0)

    print(f"\n══ FDR (top-1 precision) — {bench_cfg.name} ══")
    top1 = compute_fdr(
        model_spec=model_spec, model_pep=model_pep,
        loader=loader, device=device,
        modified_sequences=modified_sequences,
    )

    print(f"\n══ FDR (Target-Decoy, internal token decoys) — {bench_cfg.name} ══")
    tda = compute_tda_fdr(
        model_spec=model_spec, model_pep=model_pep,
        loader=loader, device=device,
        modified_sequences=modified_sequences,
        decoy_strategy="reverse-inner",
    )

    if use_external_decoys and db_seqs is not None and is_decoy_db is not None:
        print(f"\n══ FDR (Target-Decoy, external decoy DB — Gabriel-style) — {bench_cfg.name} ══")
        tda_ext = compute_external_tda_fdr(
            model_spec=model_spec,
            dataset=dataset,
            modified_sequences=modified_sequences,
            db_seqs=db_seqs,
            is_decoy_db=is_decoy_db,
            model_pep=model_pep,
            residue_set=None,    # already encoded in db
            device=device,
        )
        tda["external"] = tda_ext

    return FDRReport(name=bench_cfg.name, top1_fdr=top1, tda_fdr=tda)


@torch.no_grad()
def compute_external_tda_fdr(
    model_spec:         nn.Module,
    dataset:            SpectraPeptideDataset,
    modified_sequences: List[str],
    db_seqs:            List[str],
    is_decoy_db:        np.ndarray,
    model_pep:          Optional[nn.Module] = None,
    residue_set=None,
    device:             Optional[torch.device] = None,
    thresholds:         Optional[np.ndarray] = None,
    chunk_size:         int = 1024,
) -> dict:
    """Target-decoy FDR using an externally-supplied decoy peptide DB.

    Mirrors ``compute_tda_fdr`` but searches against the (target+decoy) DB
    you've already built and encoded via :func:`build_database_index`. This
    is the "Gabriel-style" variant: decoys come from the reversed protein
    FASTA, not from token-level reversal of each target.

    The function expects either (a) precomputed DB embeddings already in
    the HNSW index — but we re-encode here to do an exact dot-product
    against the full DB, which is what target-decoy competition needs at
    top-1.

    Returns the same dict shape as :func:`compute_tda_fdr` (top-1 hits,
    threshold table, q-values).
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.05)
    if device is None:
        device = next(model_spec.parameters()).device

    loader = DataLoader(dataset, batch_size=512, num_workers=0)
    spec_chunks: List[np.ndarray] = []
    model_spec.eval()
    with torch.no_grad():
        for batch in loader:
            specs = batch[0].to(device)
            pres  = batch[2].to(device)
            z = model_spec(specs, pres)
            spec_chunks.append(F.normalize(z, dim=-1).cpu().float().numpy())
    spec_emb = np.vstack(spec_chunks).astype(np.float32, copy=False)

    if model_pep is None or residue_set is None:
        raise ValueError(
            "compute_external_tda_fdr needs model_pep+residue_set to re-encode "
            "the DB (or pass a precomputed db_emb)."
        )
    db_emb_chunks: List[np.ndarray] = []
    model_pep.eval()
    bs = 1024
    with torch.no_grad():
        for start in range(0, len(db_seqs), bs):
            chunk = db_seqs[start:start + bs]
            toks = np.stack([
                preprocess_peptide_residueset(s, residue_set, max_len=MAX_PEPTIDE_LEN)
                for s in chunk
            ])
            z = model_pep(torch.as_tensor(toks, dtype=torch.int64, device=device))
            db_emb_chunks.append(F.normalize(z, dim=-1).cpu().float().numpy())
    db_emb = np.vstack(db_emb_chunks).astype(np.float32, copy=False)

    n_rows = spec_emb.shape[0]
    top1_idx     = np.empty(n_rows, dtype=np.int64)
    top1_scores  = np.empty(n_rows, dtype=np.float32)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        sims = spec_emb[start:end] @ db_emb.T
        local = np.argmax(sims, axis=1)
        top1_idx[start:end]    = local
        top1_scores[start:end] = sims[np.arange(end - start), local]

    top1_is_decoy = is_decoy_db[top1_idx]
    top1_seq      = [db_seqs[int(i)] for i in top1_idx]
    n_targets = int((~top1_is_decoy).sum())
    n_decoys  = int(top1_is_decoy.sum())
    fdr_top1  = min(1.0, n_decoys / n_targets) if n_targets else 1.0

    rows = []
    for t in thresholds:
        accepted = top1_scores >= t
        t_hits = int((accepted & ~top1_is_decoy).sum())
        d_hits = int((accepted & top1_is_decoy).sum())
        fdr    = min(1.0, d_hits / t_hits) if t_hits else (1.0 if d_hits else 0.0)
        rows.append({
            "threshold": float(t),
            "n_accepted": int(accepted.sum()),
            "target_hits": t_hits,
            "decoy_hits":  d_hits,
            "fdr":         float(fdr),
            "fraction_accepted": float(accepted.mean()),
        })

    order = np.argsort(-top1_scores)
    cum_d = np.cumsum(top1_is_decoy[order])
    cum_t = np.cumsum(~top1_is_decoy[order])
    fdr_by_rank = np.minimum(cum_d / np.maximum(cum_t, 1), 1.0)
    qsorted = np.minimum.accumulate(fdr_by_rank[::-1])[::-1]
    qvalues = np.empty(n_rows, dtype=np.float64)
    qvalues[order] = qsorted

    out = {
        "n_total":         int(n_rows),
        "n_target_top1":   n_targets,
        "n_decoy_top1":    n_decoys,
        "fdr_top1":        float(fdr_top1),
        "top1_scores":     top1_scores,
        "top1_is_decoy":   top1_is_decoy,
        "top1_sequence":   top1_seq,
        "qvalues":         qvalues,
        "threshold_results": rows,
    }

    print("=" * 60)
    print(f"  Spectra:        {n_rows}")
    print(f"  Target top-1:   {n_targets}")
    print(f"  Decoy top-1:    {n_decoys}")
    print(f"  FDR top-1:      {fdr_top1:.4f}")
    print(f"  {'Threshold':>10} {'Accepted':>10} {'Targets':>10} {'Decoys':>10} {'FDR':>10}")
    for r in rows:
        if r["n_accepted"] > 0:
            print(f"  {r['threshold']:>10.2f} {r['n_accepted']:>10} "
                  f"{r['target_hits']:>10} {r['decoy_hits']:>10} {r['fdr']:>10.4f}")
    print("=" * 60)
    return out


# ---------------------------------------------------------------------------
# Rescoring + Percolator export
# ---------------------------------------------------------------------------

def rescore_benchmark(
    bench_cfg:   BenchmarkConfig,
    retrieval:   RetrievalReport,
    rescorer,                                  # InstaNovo model
    dataset:     SpectraPeptideDataset,
    kept_df:     pd.DataFrame,
) -> RetrievalReport:
    """Neural-rescore the top-N Stage-1 candidates and update the report."""
    from src.retrieval.rerank import rescore_and_rerank

    # rescore_and_rerank expects pre_tensor = (Q, 2) [mz, charge].
    pres3 = dataset.pres.float()                       # (Q, 3) [mass, charge, mz]
    pres2 = torch.stack([pres3[:, 2], pres3[:, 1]], dim=1)

    reranked, ranks_rs = rescore_and_rerank(
        model              = rescorer,
        stage1_candidates  = retrieval.stage1_candidates,
        spec_tensor        = dataset.specs,
        pre_tensor         = pres2,
        ground_truth_seqs  = kept_df["modified_sequence"].tolist(),
        rescore_top_n      = bench_cfg.rescore_top_n,
        rescore_batch      = bench_cfg.rescore_batch,
        reduction          = "mean",
    )

    retrieval.rescored_candidates = reranked
    retrieval.rescored_ranks      = ranks_rs
    print("\n── Recall@k after rescoring ─────────────────────────")
    for k in bench_cfg.k_eval:
        rs = float((ranks_rs <= k).mean())
        s1 = retrieval.recall.get(k, float("nan"))
        delta = rs - s1
        print(f"  Recall@{k:<4d}: {rs:.4f}   (Δ vs Stage-1: {delta:+.4f})")
    return retrieval


def export_percolator_tsv(
    retrieval:          RetrievalReport,
    kept_df:            pd.DataFrame,
    db_seqs:            List[str],
    is_decoy_db:        np.ndarray,
    peptide_to_proteins: dict[str, List[str]],
    out_tsv:            Path,
    n_protein_slots:    int = 7,
) -> Path:
    """Emit a Percolator-format TSV matching ``df_percolator_sbrodae.tsv``.

    Columns: ``id, is_decoy, scan_number, norm_preds, Protein_list,
    proteinId1..proteinIdN``.

    - ``is_decoy = -1`` for decoy top-1, ``1`` for target (Percolator convention).
    - ``id`` follows Thermo PD style: ``controllerType=0 controllerNumber=1 scan=<scan>``.
    - The top-1 candidate per spectrum is the candidate after rescoring
      (if rescoring was run) or after Stage-1.
    """
    cands = retrieval.rescored_candidates or retrieval.stage1_candidates
    db_seq_to_uid = {s: i for i, s in enumerate(db_seqs)}

    rows = []
    for i, (_, row) in enumerate(kept_df.iterrows()):
        if not cands[i]:
            continue
        top1_seq = cands[i][0][0]
        uid = db_seq_to_uid.get(top1_seq, -1)
        is_decoy = -1 if (uid >= 0 and is_decoy_db[uid]) else 1
        proteins = peptide_to_proteins.get(top1_seq, [])
        if is_decoy == -1:
            proteins = [f"DECOY_{p}" for p in proteins] or ["DECOY_unknown"]
        proteins = proteins[:n_protein_slots] or ["unknown"]
        protein_slots = proteins + [""] * (n_protein_slots - len(proteins))
        rows.append({
            "id":            f"controllerType=0 controllerNumber=1 scan={int(row['scan_id'])}",
            "is_decoy":      is_decoy,
            "scan_number":   int(row["scan_id"]),
            "norm_preds":    top1_seq,
            "Protein_list":  str(proteins),
            **{f"proteinId{j+1}": protein_slots[j] for j in range(n_protein_slots)},
        })

    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"  Wrote Percolator TSV: {out_tsv}  ({len(rows):,} rows)")
    return out_tsv


def run_percolator(
    perc_tsv:        Path,
    percolator_exe:  Path,
    out_dir:         Path,
    extra_args:      Sequence[str] = (),
) -> dict[str, Path]:
    """Invoke ``percolator.exe`` on a Percolator-format TSV.

    Returns paths to the PSM and peptide result files. Caller is expected
    to load and post-process them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psm_out = out_dir / "percolator_psms.tsv"
    pep_out = out_dir / "percolator_peptides.tsv"

    cmd = [str(percolator_exe), "-r", str(pep_out), "-m", str(psm_out),
           *list(extra_args), str(perc_tsv)]
    print("  Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"percolator exited with code {res.returncode}")
    print(f"  Percolator outputs: {psm_out.name}, {pep_out.name}")
    return {"psms": psm_out, "peptides": pep_out}


__all__ = [
    "BenchmarkConfig",
    "HELA_PRESET",
    "BRODAE_PRESET",
    "FDRReport",
    "RetrievalReport",
    "format_modified_sequence",
    "load_psm_table",
    "load_target_peptides",
    "load_decoy_peptides",
    "load_ms_spectra",
    "join_psm_with_spectra",
    "build_database_index",
    "run_retrieval_benchmark",
    "run_fdr_benchmark",
    "compute_external_tda_fdr",
    "rescore_benchmark",
    "export_percolator_tsv",
    "run_percolator",
]
