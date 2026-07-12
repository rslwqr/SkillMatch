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
DEFAULT_INDEX_PATH = Path(
    "data/embeddings/e5_small_v2_flat.index"
)
DEFAULT_RESULTS_PATH = Path(
    "results/dense_flat_top20.csv"
)
DEFAULT_METRICS_PATH = Path(
    "results/dense_flat_metrics.json"
)


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


def get_process_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def load_corpus(
    corpus_path: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    LOGGER.info("Loading corpus from %s", corpus_path)

    columns = [
        "doc_id",
        "job_title",
        "company",
        "job_location",
        "job_level",
        "job_type",
        "job_skills",
        "search_text",
    ]

    corpus = pd.read_parquet(
        corpus_path,
        columns=columns,
    )

    validate_columns(
        corpus,
        set(columns),
        "Corpus",
    )

    if limit is not None:
        corpus = corpus.head(limit).copy()

    corpus = corpus.fillna("")

    LOGGER.info("Loaded %s documents", f"{len(corpus):,}")

    return corpus


def load_model(
    model_name: str,
    device: str,
    max_seq_length: int,
) -> SentenceTransformer:
    LOGGER.info("Loading model %s on %s", model_name, device)

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    model.max_seq_length = min(
        max_seq_length,
        model.max_seq_length,
    )

    LOGGER.info(
        "Model embedding dimension: %s",
        model.get_sentence_embedding_dimension(),
    )
    LOGGER.info(
        "Maximum sequence length: %s",
        model.max_seq_length,
    )

    return model


def encode_corpus(
    corpus: pd.DataFrame,
    model: SentenceTransformer,
    embeddings_path: Path,
    batch_size: int,
    rebuild: bool,
) -> dict[str, float | int]:
    if embeddings_path.exists() and not rebuild:
        existing = np.load(
            embeddings_path,
            mmap_mode="r",
        )

        if existing.shape[0] != len(corpus):
            raise ValueError(
                "Existing embeddings do not match the corpus size: "
                f"{existing.shape[0]:,} != {len(corpus):,}. "
                "Run with --rebuild-embeddings."
            )

        LOGGER.info(
            "Using existing embeddings: %s, shape=%s",
            embeddings_path,
            existing.shape,
        )

        return {
            "embedding_time_seconds": 0.0,
            "documents_per_second": 0.0,
            "embedding_dimension": int(existing.shape[1]),
        }

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dimension = model.get_sentence_embedding_dimension()

    if dimension is None:
        raise ValueError("Could not determine embedding dimension.")

    LOGGER.info(
        "Encoding %s documents in batches of %s",
        f"{len(corpus):,}",
        batch_size,
    )

    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(corpus), dimension),
    )

    start_time = time.perf_counter()

    for start in range(0, len(corpus), batch_size):
        end = min(start + batch_size, len(corpus))

        texts = [
            f"passage: {text}"
            for text in corpus["search_text"].iloc[start:end].tolist()
        ]

        batch_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        embeddings[start:end] = batch_embeddings
        embeddings.flush()

        if (
            end % 10_000 == 0
            or end == len(corpus)
        ):
            elapsed = time.perf_counter() - start_time
            throughput = end / elapsed

            LOGGER.info(
                "Encoded %s / %s documents | %.2f docs/s",
                f"{end:,}",
                f"{len(corpus):,}",
                throughput,
            )

    elapsed = time.perf_counter() - start_time

    del embeddings

    LOGGER.info(
        "Encoding completed in %.2f minutes",
        elapsed / 60,
    )

    return {
        "embedding_time_seconds": elapsed,
        "documents_per_second": len(corpus) / elapsed,
        "embedding_dimension": int(dimension),
    }


def build_flat_index(
    embeddings_path: Path,
    index_path: Path,
    rebuild: bool,
    add_batch_size: int = 50_000,
) -> dict[str, float | int]:
    if index_path.exists() and not rebuild:
        index = faiss.read_index(str(index_path))

        LOGGER.info(
            "Using existing FAISS index: %s vectors",
            f"{index.ntotal:,}",
        )

        return {
            "index_build_time_seconds": 0.0,
            "index_vectors": int(index.ntotal),
            "index_size_mb": index_path.stat().st_size / (1024**2),
        }

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"FAISS requires float32 vectors, got {embeddings.dtype}"
        )

    dimension = embeddings.shape[1]

    LOGGER.info(
        "Building IndexFlatIP for %s vectors",
        f"{len(embeddings):,}",
    )

    index = faiss.IndexFlatIP(dimension)

    start_time = time.perf_counter()

    for start in range(0, len(embeddings), add_batch_size):
        end = min(start + add_batch_size, len(embeddings))

        batch = np.ascontiguousarray(
            embeddings[start:end],
            dtype=np.float32,
        )

        index.add(batch)

        LOGGER.info(
            "Added %s / %s vectors",
            f"{end:,}",
            f"{len(embeddings):,}",
        )

    elapsed = time.perf_counter() - start_time

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    index_size_mb = index_path.stat().st_size / (1024**2)

    LOGGER.info(
        "Index built in %.2f seconds",
        elapsed,
    )
    LOGGER.info(
        "Index size: %.2f MB",
        index_size_mb,
    )

    return {
        "index_build_time_seconds": elapsed,
        "index_vectors": int(index.ntotal),
        "index_size_mb": index_size_mb,
    }


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


def run_dense_search(
    corpus: pd.DataFrame,
    queries_path: Path,
    model: SentenceTransformer,
    index_path: Path,
    output_path: Path,
    top_k: int,
    query_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    LOGGER.info("Loading queries from %s", queries_path)

    queries = pd.read_csv(queries_path)

    validate_columns(
        queries,
        {"query_id", "query", "category"},
        "Queries",
    )

    index = faiss.read_index(str(index_path))

    if index.ntotal != len(corpus):
        raise ValueError(
            "FAISS index and corpus have different sizes: "
            f"{index.ntotal:,} != {len(corpus):,}"
        )

    query_embeddings = encode_queries(
        queries=queries["query"].tolist(),
        model=model,
        batch_size=query_batch_size,
    )

    all_results: list[dict[str, object]] = []
    latencies: list[float] = []

    # Measure each query separately so that latency statistics
    # represent one user request.
    for query_number, query_row in queries.iterrows():
        query_vector = query_embeddings[
            query_number : query_number + 1
        ]

        start_time = time.perf_counter()

        scores, indices = index.search(
            query_vector,
            top_k,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1_000

        latencies.append(latency_ms)

        LOGGER.info(
            "%s | %s results | %.2f ms",
            query_row["query_id"],
            top_k,
            latency_ms,
        )

        for rank, (document_index, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            if document_index < 0:
                continue

            document = corpus.iloc[int(document_index)]

            all_results.append(
                {
                    "query_id": query_row["query_id"],
                    "query": query_row["query"],
                    "category": query_row["category"],
                    "system": "dense_flat",
                    "rank": rank,
                    "doc_id": document["doc_id"],
                    "score": float(score),
                    "job_title": document["job_title"],
                    "company": document["company"],
                    "job_location": document["job_location"],
                    "job_level": document["job_level"],
                    "job_type": document["job_type"],
                    "job_skills": document["job_skills"],
                    "latency_ms": latency_ms,
                }
            )

    results = pd.DataFrame(all_results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    latency_array = np.asarray(
        latencies,
        dtype=np.float64,
    )

    metrics = {
        "query_count": len(queries),
        "top_k": top_k,
        "mean_latency_ms": float(latency_array.mean()),
        "p95_latency_ms": float(
            np.quantile(latency_array, 0.95)
        ),
        "min_latency_ms": float(latency_array.min()),
        "max_latency_ms": float(latency_array.max()),
    }

    LOGGER.info("Saved results to %s", output_path)
    LOGGER.info(
        "Mean search latency: %.2f ms",
        metrics["mean_latency_ms"],
    )
    LOGGER.info(
        "P95 search latency: %.2f ms",
        metrics["p95_latency_ms"],
    )

    return results, metrics


def save_metrics(
    metrics_path: Path,
    metrics: dict[str, object],
) -> None:
    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("Saved metrics to %s", metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact dense retrieval with E5 and FAISS."
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
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use the first N documents for a quick test.",
    )

    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    device = choose_device(args.device)

    LOGGER.info("Selected device: %s", device)
    LOGGER.info(
        "Initial process RAM: %.2f MB",
        get_process_memory_mb(),
    )

    corpus = load_corpus(
        corpus_path=args.corpus,
        limit=args.limit,
    )

    model = load_model(
        model_name=args.model,
        device=device,
        max_seq_length=args.max_seq_length,
    )

    encoding_metrics = encode_corpus(
        corpus=corpus,
        model=model,
        embeddings_path=args.embeddings,
        batch_size=args.batch_size,
        rebuild=args.rebuild_embeddings,
    )

    index_metrics = build_flat_index(
        embeddings_path=args.embeddings,
        index_path=args.index,
        rebuild=args.rebuild_index,
    )

    _, search_metrics = run_dense_search(
        corpus=corpus,
        queries_path=args.queries,
        model=model,
        index_path=args.index,
        output_path=args.output,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
    )

    final_metrics = {
        "system": "dense_flat",
        "model": args.model,
        "device": device,
        "documents": len(corpus),
        "max_seq_length": model.max_seq_length,
        **encoding_metrics,
        **index_metrics,
        **search_metrics,
        "final_process_ram_mb": get_process_memory_mb(),
    }

    save_metrics(
        metrics_path=args.metrics,
        metrics=final_metrics,
    )


if __name__ == "__main__":
    main()