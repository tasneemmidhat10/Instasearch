"""Two-stage retrieval and neural rescoring pipeline.

Submodules
----------
index
    :class:`~src.retrieval.index.HNSWIndex` — build, search, save, load.
    :func:`~src.retrieval.index.build_exact_index` — brute-force baseline.
    :func:`~src.retrieval.index.sweep_ef_search` — latency/recall sweep.
    :func:`~src.retrieval.index.sweep_build_params` — M/ef_construction grid search.

search
    :func:`~src.retrieval.search.extract_embeddings` — batched dual-encoder encoding.
    :func:`~src.retrieval.search.build_unique_peptide_db` — deduplicate peptide DB.
    :func:`~src.retrieval.search.retrieve_batch` — top-k retrieval over a query set.
    :func:`~src.retrieval.search.retrieve_single` — single-spectrum interactive helper.
    :func:`~src.retrieval.search.evaluate_recall` — Recall@k (sequence-string GT).
    :func:`~src.retrieval.search.candidate_recall` — Candidate Recall@k over unique peptides.

rerank
    :func:`~src.retrieval.rerank.instanovo_score` — Casanovo-DB neural PSM score.
    :func:`~src.retrieval.rerank.rescore_and_rerank` — two-stage reranking driver.
    :func:`~src.retrieval.rerank.encode_candidates` — tokenize + reverse + pad.
    :func:`~src.retrieval.rerank.precursor_features` — build InstaNovo precursor tensor.
    :func:`~src.retrieval.rerank.rescore_spectrum` — single-spectrum convenience wrapper.
"""
from .index import HNSWConfig, HNSWIndex, build_exact_index, sweep_ef_search, sweep_build_params
from .search import (
    extract_embeddings,
    build_unique_peptide_db,
    retrieve_batch,
    retrieve_single,
    evaluate_recall,
    candidate_recall,
    compute_fdr,
    compute_tda_fdr,
    evaluate_top_k_retrieval,
)
from .rerank import (
    load_instanovo_rescorer,
    encode_candidates,
    instanovo_score,
    rescore_and_rerank,
    compute_neural_tda_fdr,
    precursor_features,
    rescore_spectrum,
    rescore_psms_from_embeddings,
)
from .benchmarks import (
    BenchmarkConfig,
    HELA_PRESET,
    BRODAE_PRESET,
    FDRReport,
    RetrievalReport,
    format_modified_sequence,
    load_psm_table,
    load_target_peptides,
    load_decoy_peptides,
    load_ms_spectra,
    join_psm_with_spectra,
    build_database_index,
    run_retrieval_benchmark,
    run_fdr_benchmark,
    compute_external_tda_fdr,
    rescore_benchmark,
    export_percolator_tsv,
    run_percolator,
)

__all__ = [
    # index
    "HNSWConfig",
    "HNSWIndex",
    "build_exact_index",
    "sweep_ef_search",
    "sweep_build_params",
    # search
    "extract_embeddings",
    "build_unique_peptide_db",
    "retrieve_batch",
    "retrieve_single",
    "evaluate_recall",
    "candidate_recall",
    "compute_fdr",
    "compute_tda_fdr",
    "evaluate_top_k_retrieval",
    # rerank
    "load_instanovo_rescorer",
    "encode_candidates",
    "instanovo_score",
    "rescore_and_rerank",
    "compute_neural_tda_fdr",
    "precursor_features",
    "rescore_spectrum",
    "rescore_psms_from_embeddings",
    # benchmarks
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
