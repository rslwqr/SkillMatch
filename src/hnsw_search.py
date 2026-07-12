from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/e5-small-v2"

DEFAULT_CORPUS_PATH = Path("data/processed/corpus_500k.parquet")
DEFAULT_QUERIES_PATH = Path("qrels/queries.csv")
DEFAULT_EMBEDDINGS_PATH = Path(
    "data/embeddings/e5_small_v2_embeddings.npy"
)
DEFAULT_FLAT_INDEX_PATH = Path(
    "data/embeddings/e5_small_v2_flat.index"
)
DEFAULT_HNSW_INDEX_PATH = Path(
    "data/embeddings/e5_small_v2_hnsw_m32_efc200.index"
)
DEFAULT_RESULTS_DIR = Path("results/hnsw")
DEFAULT_METRICS_PATH = Path("results/hnsw_metrics.csv")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def choose_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def get_process_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing columns: {sorted(missing)}"
        )


def load_corpus(corpus_path: Path) -> pd.DataFrame:
    columns = [
        "doc_id",
        "job_title",
        "company",
        "job_location",
        "job_level",
        "job_type",
        "job_skills",
    ]

    LOGGER.info("Loading corpus from %s", corpus_path)

    corpus = pd.read_parquet(
        corpus_path,
        columns=columns,
    ).fillna("")

    validate_columns(
        corpus,
        set(columns),
        "Corpus",
    )

    LOGGER.info("Loaded %s documents", f"{len(corpus):,}")

    return corpus


def load_queries(queries_path: Path) -> pd.DataFrame:
    LOGGER.info("Loading queries from %s", queries_path)

    queries = pd.read_csv(queries_path)

    validate_columns(
        queries,
        {"query_id", "query", "category"},
        "Queries",
    )

    return queries


def load_model(
    model_name: str,
    device: str,
) -> SentenceTransformer:
    LOGGER.info("Loading model %s on %s", model_name, device)

    return SentenceTransformer(
        model_name,
        device=device,
    )


def encode_queries(
    queries: list[str],
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    prefixed_queries = [
        f"query: {query}"
        for query in queries
    ]

    embeddings = model.encode(
        prefixed_queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )


def validate_embeddings(
    embeddings_path: Path,
    expected_documents: int,
) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}"
        )

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected float32 embeddings, got {embeddings.dtype}"
        )

    if embeddings.shape[0] != expected_documents:
        raise ValueError(
            "Corpus and embeddings have different sizes: "
            f"{expected_documents:,} != {embeddings.shape[0]:,}"
        )

    LOGGER.info(
        "Embeddings shape: %s",
        embeddings.shape,
    )

    return embeddings


def build_hnsw_index(
    embeddings: np.ndarray,
    index_path: Path,
    m: int,
    ef_construction: int,
    add_batch_size: int,
    rebuild: bool,
) -> tuple[faiss.Index, dict[str, float | int]]:
    if index_path.exists() and not rebuild:
        LOGGER.info(
            "Loading existing HNSW index from %s",
            index_path,
        )

        index = faiss.read_index(str(index_path))

        if index.ntotal != embeddings.shape[0]:
            raise ValueError(
                "Existing HNSW index has a different number "
                "of vectors. Run with --rebuild-index."
            )

        metrics = {
            "index_build_time_seconds": 0.0,
            "index_size_mb": (
                index_path.stat().st_size / (1024**2)
            ),
            "index_vectors": int(index.ntotal),
        }

        return index, metrics

    if index_path.exists():
        index_path.unlink()

    dimension = int(embeddings.shape[1])

    LOGGER.info(
        "Building HNSW: documents=%s, dimension=%s, "
        "M=%s, efConstruction=%s",
        f"{embeddings.shape[0]:,}",
        dimension,
        m,
        ef_construction,
    )

    index = faiss.IndexHNSWFlat(
        dimension,
        m,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = ef_construction

    start_time = time.perf_counter()

    for start in range(
        0,
        embeddings.shape[0],
        add_batch_size,
    ):
        end = min(
            start + add_batch_size,
            embeddings.shape[0],
        )

        batch = np.ascontiguousarray(
            embeddings[start:end],
            dtype=np.float32,
        )

        index.add(batch)

        LOGGER.info(
            "Added %s / %s vectors",
            f"{end:,}",
            f"{embeddings.shape[0]:,}",
        )

    build_time = time.perf_counter() - start_time

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    index_size_mb = (
        index_path.stat().st_size / (1024**2)
    )

    LOGGER.info(
        "HNSW build completed in %.2f minutes",
        build_time / 60,
    )
    LOGGER.info(
        "HNSW index size: %.2f MB",
        index_size_mb,
    )

    metrics = {
        "index_build_time_seconds": build_time,
        "index_size_mb": index_size_mb,
        "index_vectors": int(index.ntotal),
    }

    return index, metrics


def exact_flat_results(
    flat_index_path: Path,
    query_embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if not flat_index_path.exists():
        raise FileNotFoundError(
            f"Flat index not found: {flat_index_path}"
        )

    LOGGER.info(
        "Loading exact Flat index from %s",
        flat_index_path,
    )

    flat_index = faiss.read_index(
        str(flat_index_path)
    )

    _, exact_indices = flat_index.search(
        query_embeddings,
        top_k,
    )

    return exact_indices


def calculate_ann_recall_at_k(
    exact_indices: np.ndarray,
    approximate_indices: np.ndarray,
    k: int,
) -> float:
    recalls: list[float] = []

    for exact_row, approximate_row in zip(
        exact_indices,
        approximate_indices,
    ):
        exact_set = set(
            int(value)
            for value in exact_row[:k]
            if value >= 0
        )

        approximate_set = set(
            int(value)
            for value in approximate_row[:k]
            if value >= 0
        )

        overlap = len(
            exact_set.intersection(approximate_set)
        )

        recalls.append(overlap / k)

    return float(np.mean(recalls))


def measure_hnsw_search(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int,
    warmup_runs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Warm-up is excluded from benchmark statistics.
    for _ in range(warmup_runs):
        index.search(
            query_embeddings[:1],
            top_k,
        )

    scores_list: list[np.ndarray] = []
    indices_list: list[np.ndarray] = []
    latencies_ms: list[float] = []

    for query_vector in query_embeddings:
        query_vector = query_vector.reshape(1, -1)

        start_time = time.perf_counter()

        scores, indices = index.search(
            query_vector,
            top_k,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        scores_list.append(scores[0])
        indices_list.append(indices[0])
        latencies_ms.append(latency_ms)

    return (
        np.asarray(scores_list),
        np.asarray(indices_list),
        np.asarray(latencies_ms, dtype=np.float64),
    )


def build_results_dataframe(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    latencies_ms: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    system_name = (
        f"hnsw_m{m}_efc{ef_construction}_efs{ef_search}"
    )

    for query_number, query_row in queries.iterrows():
        for rank, (document_index, score) in enumerate(
            zip(
                indices[query_number],
                scores[query_number],
            ),
            start=1,
        ):
            if document_index < 0:
                continue

            document = corpus.iloc[int(document_index)]

            rows.append(
                {
                    "query_id": query_row["query_id"],
                    "query": query_row["query"],
                    "category": query_row["category"],
                    "system": system_name,
                    "rank": rank,
                    "doc_id": document["doc_id"],
                    "score": float(score),
                    "job_title": document["job_title"],
                    "company": document["company"],
                    "job_location": document["job_location"],
                    "job_level": document["job_level"],
                    "job_type": document["job_type"],
                    "job_skills": document["job_skills"],
                    "latency_ms": float(
                        latencies_ms[query_number]
                    ),
                }
            )

    return pd.DataFrame(rows)


def run_ef_search_sweep(
    index: faiss.Index,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    query_embeddings: np.ndarray,
    exact_indices: np.ndarray,
    results_dir: Path,
    top_k: int,
    recall_k: int,
    m: int,
    ef_construction: int,
    ef_search_values: list[int],
    index_metrics: dict[str, float | int],
    warmup_runs: int,
) -> pd.DataFrame:
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_rows: list[dict[str, object]] = []

    for ef_search in ef_search_values:
        LOGGER.info(
            "Testing efSearch=%s",
            ef_search,
        )

        index.hnsw.efSearch = ef_search

        scores, indices, latencies_ms = (
            measure_hnsw_search(
                index=index,
                query_embeddings=query_embeddings,
                top_k=top_k,
                warmup_runs=warmup_runs,
            )
        )

        ann_recall = calculate_ann_recall_at_k(
            exact_indices=exact_indices,
            approximate_indices=indices,
            k=recall_k,
        )

        mean_latency = float(
            latencies_ms.mean()
        )
        p95_latency = float(
            np.quantile(latencies_ms, 0.95)
        )

        system_name = (
            f"hnsw_m{m}_efc{ef_construction}"
            f"_efs{ef_search}"
        )

        results = build_results_dataframe(
            corpus=corpus,
            queries=queries,
            scores=scores,
            indices=indices,
            latencies_ms=latencies_ms,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

        output_path = (
            results_dir / f"{system_name}_top20.csv"
        )

        results.to_csv(
            output_path,
            index=False,
        )

        LOGGER.info(
            "%s | Recall@%s=%.4f | mean=%.3f ms "
            "| p95=%.3f ms",
            system_name,
            recall_k,
            ann_recall,
            mean_latency,
            p95_latency,
        )

        metrics_rows.append(
            {
                "system": system_name,
                "documents": len(corpus),
                "M": m,
                "efConstruction": ef_construction,
                "efSearch": ef_search,
                f"ann_recall_at_{recall_k}": ann_recall,
                "mean_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "min_latency_ms": float(
                    latencies_ms.min()
                ),
                "max_latency_ms": float(
                    latencies_ms.max()
                ),
                **index_metrics,
                "process_ram_mb": get_process_memory_mb(),
                "results_path": str(output_path),
            }
        )

    return pd.DataFrame(metrics_rows)


def save_json_summary(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = metrics.to_dict(
        orient="records"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate a FAISS HNSW index "
            "using existing dense embeddings."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
    )

    parser.add_argument(
        "--flat-index",
        type=Path,
        default=DEFAULT_FLAT_INDEX_PATH,
    )

    parser.add_argument(
        "--hnsw-index",
        type=Path,
        default=DEFAULT_HNSW_INDEX_PATH,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )

    parser.add_argument(
        "--m",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help=(
            "One or more efSearch values, for example "
            "--ef-search 32 64 128"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--recall-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    device = choose_device(
        args.device
    )

    LOGGER.info(
        "Selected device: %s",
        device,
    )
    LOGGER.info(
        "Initial RAM: %.2f MB",
        get_process_memory_mb(),
    )

    corpus = load_corpus(
        args.corpus
    )

    queries = load_queries(
        args.queries
    )

    embeddings = validate_embeddings(
        embeddings_path=args.embeddings,
        expected_documents=len(corpus),
    )

    model = load_model(
        model_name=args.model,
        device=device,
    )

    query_embeddings = encode_queries(
        queries=queries["query"].tolist(),
        model=model,
        batch_size=args.query_batch_size,
    )

    exact_indices = exact_flat_results(
        flat_index_path=args.flat_index,
        query_embeddings=query_embeddings,
        top_k=args.top_k,
    )

    index, index_metrics = build_hnsw_index(
        embeddings=embeddings,
        index_path=args.hnsw_index,
        m=args.m,
        ef_construction=args.ef_construction,
        add_batch_size=args.add_batch_size,
        rebuild=args.rebuild_index,
    )

    metrics = run_ef_search_sweep(
        index=index,
        corpus=corpus,
        queries=queries,
        query_embeddings=query_embeddings,
        exact_indices=exact_indices,
        results_dir=args.results_dir,
        top_k=args.top_k,
        recall_k=args.recall_k,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search_values=args.ef_search,
        index_metrics=index_metrics,
        warmup_runs=args.warmup_runs,
    )

    args.metrics.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        args.metrics,
        index=False,
    )

    save_json_summary(
        metrics=metrics,
        output_path=args.metrics.with_suffix(".json"),
    )

    LOGGER.info(
        "Saved metrics to %s",
        args.metrics,
    )

    print("\nHNSW experiment summary:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()