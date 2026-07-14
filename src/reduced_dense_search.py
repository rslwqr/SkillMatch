from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/e5-small-v2"
DEFAULT_CORPUS_PATH = Path("data/processed/corpus_500k.parquet")
DEFAULT_QUERIES_PATH = Path("qrels/queries.csv")
DEFAULT_EMBEDDINGS_DIR = Path("data/embeddings")
DEFAULT_PCA_MODEL_PATH = Path(
    "data/embeddings/e5_small_v2_pca.joblib"
)
DEFAULT_RESULTS_DIR = Path(
    "results/dimensionality_reduction"
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

    return model


def load_pca_model(
    pca_model_path: Path,
):
    if not pca_model_path.exists():
        raise FileNotFoundError(
            f"PCA model not found: {pca_model_path}"
        )

    LOGGER.info("Loading PCA model from %s", pca_model_path)

    return joblib.load(pca_model_path)


def load_reduced_embeddings(
    embeddings_path: Path,
    dimension: int,
    expected_documents: int,
) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Reduced embeddings not found: {embeddings_path}"
        )

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got shape {embeddings.shape}"
        )

    if embeddings.shape[0] != expected_documents:
        raise ValueError(
            "Reduced embeddings and corpus have different sizes: "
            f"{embeddings.shape[0]:,} != {expected_documents:,}"
        )

    if embeddings.shape[1] != dimension:
        raise ValueError(
            "Reduced embeddings have incorrect dimension: "
            f"{embeddings.shape[1]} != {dimension}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected float32 embeddings, got {embeddings.dtype}"
        )

    LOGGER.info(
        "Loaded reduced embeddings with shape %s",
        embeddings.shape,
    )

    return embeddings


def build_flat_index(
    embeddings: np.ndarray,
    index_path: Path,
    rebuild: bool,
    add_batch_size: int,
) -> dict[str, float | int]:
    if index_path.exists() and not rebuild:
        index = faiss.read_index(str(index_path))

        if index.ntotal != len(embeddings):
            raise ValueError(
                "Existing FAISS index and embeddings have different sizes: "
                f"{index.ntotal:,} != {len(embeddings):,}"
            )

        if index.d != embeddings.shape[1]:
            raise ValueError(
                "Existing FAISS index has incorrect dimension: "
                f"{index.d} != {embeddings.shape[1]}"
            )

        LOGGER.info(
            "Using existing FAISS index: %s vectors",
            f"{index.ntotal:,}",
        )

        return {
            "index_build_time_seconds": 0.0,
            "index_vectors": int(index.ntotal),
            "index_size_mb": index_path.stat().st_size / (1024**2),
        }

    index = faiss.IndexFlatIP(embeddings.shape[1])

    start_time = time.perf_counter()

    for start in range(0, len(embeddings), add_batch_size):
        end = min(
            start + add_batch_size,
            len(embeddings),
        )

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


def encode_and_reduce_queries(
    queries: list[str],
    model: SentenceTransformer,
    pca,
    dimension: int,
    batch_size: int,
) -> np.ndarray:
    prefixed_queries = [
        f"query: {query}"
        for query in queries
    ]

    query_embeddings = model.encode(
        prefixed_queries,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    if query_embeddings.shape[1] != pca.n_features_in_:
        raise ValueError(
            "Query embedding dimension does not match PCA input dimension: "
            f"{query_embeddings.shape[1]} != {pca.n_features_in_}"
        )

    reduced_queries = pca.transform(
        query_embeddings
    )[:, :dimension].astype(np.float32)

    norms = np.linalg.norm(
        reduced_queries,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    reduced_queries = reduced_queries / norms

    return np.ascontiguousarray(
        reduced_queries,
        dtype=np.float32,
    )


def run_dense_search(
    corpus: pd.DataFrame,
    queries_path: Path,
    model: SentenceTransformer,
    pca,
    dimension: int,
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

    if index.d != dimension:
        raise ValueError(
            "FAISS index has incorrect dimension: "
            f"{index.d} != {dimension}"
        )

    query_encoding_start = time.perf_counter()

    query_embeddings = encode_and_reduce_queries(
        queries=queries["query"].tolist(),
        model=model,
        pca=pca,
        dimension=dimension,
        batch_size=query_batch_size,
    )

    query_encoding_time_seconds = (
        time.perf_counter() - query_encoding_start
    )

    all_results: list[dict[str, object]] = []
    latencies: list[float] = []

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
                    "system": f"dense_pca{dimension}",
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
        "query_encoding_time_seconds": query_encoding_time_seconds,
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
        description="Exact dense retrieval with PCA-reduced E5 embeddings."
    )

    parser.add_argument(
        "--dimension",
        type=int,
        choices=[128, 256],
        required=True,
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
        "--pca-model",
        type=Path,
        default=DEFAULT_PCA_MODEL_PATH,
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=50_000,
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
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    dimension = args.dimension

    embeddings_path = (
        args.embeddings_dir
        / f"e5_small_v2_pca{dimension}_embeddings.npy"
    )

    index_path = (
        args.embeddings_dir
        / f"e5_small_v2_pca{dimension}_flat.index"
    )

    output_path = (
        args.results_dir
        / f"dense_pca{dimension}_top20.csv"
    )

    metrics_path = (
        args.results_dir
        / f"dense_pca{dimension}_metrics.json"
    )

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

    reduced_embeddings = load_reduced_embeddings(
        embeddings_path=embeddings_path,
        dimension=dimension,
        expected_documents=len(corpus),
    )

    model = load_model(
        model_name=args.model,
        device=device,
        max_seq_length=args.max_seq_length,
    )

    pca = load_pca_model(
        pca_model_path=args.pca_model,
    )

    if dimension > pca.n_components_:
        raise ValueError(
            "Requested dimension is larger than the number "
            f"of PCA components: {dimension} > {pca.n_components_}"
        )

    index_metrics = build_flat_index(
        embeddings=reduced_embeddings,
        index_path=index_path,
        rebuild=args.rebuild_index,
        add_batch_size=args.index_batch_size,
    )

    _, search_metrics = run_dense_search(
        corpus=corpus,
        queries_path=args.queries,
        model=model,
        pca=pca,
        dimension=dimension,
        index_path=index_path,
        output_path=output_path,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
    )

    final_metrics = {
        "system": f"dense_pca{dimension}",
        "model": args.model,
        "device": device,
        "documents": len(corpus),
        "original_embedding_dimension": int(pca.n_features_in_),
        "embedding_dimension": dimension,
        "max_seq_length": model.max_seq_length,
        "embeddings_path": str(embeddings_path),
        "pca_model_path": str(args.pca_model),
        **index_metrics,
        **search_metrics,
        "final_process_ram_mb": get_process_memory_mb(),
    }

    save_metrics(
        metrics_path=metrics_path,
        metrics=final_metrics,
    )


if __name__ == "__main__":
    main()
