from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_BM25_PATH = Path("results/bm25_top20.csv")
DEFAULT_DENSE_PATH = Path("results/dense_flat_top20.csv")
DEFAULT_HNSW_PATH = Path(
    "results/hnsw/hnsw_m32_efc200_efs64_top20.csv"
)
DEFAULT_CORPUS_PATH = Path(
    "data/processed/corpus_500k.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "qrels/annotation_pool.csv"
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_top_k_results(
    path: Path,
    system_name: str,
    top_k: int,
) -> pd.DataFrame:
    LOGGER.info("Loading %s results from %s", system_name, path)

    dataframe = pd.read_csv(path)

    required_columns = {
        "query_id",
        "query",
        "category",
        "doc_id",
        "rank",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    dataframe = dataframe[
        dataframe["rank"] <= top_k
    ].copy()

    dataframe["source_system"] = system_name

    return dataframe


def select_max_candidates_per_query(
    candidates: pd.DataFrame,
    max_candidates: int,
) -> pd.DataFrame:
    """
    Keep at most max_candidates per query.

    Priority:
    1. documents returned by more systems;
    2. lower best rank;
    3. stable doc_id order.
    """
    candidates = candidates.sort_values(
        [
            "query_id",
            "system_count",
            "best_rank",
            "doc_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    return (
        candidates
        .groupby("query_id", group_keys=False)
        .head(max_candidates)
        .reset_index(drop=True)
    )


def build_annotation_pool(
    bm25_path: Path,
    dense_path: Path,
    hnsw_path: Path,
    corpus_path: Path,
    output_path: Path,
    source_top_k: int,
    max_candidates: int,
) -> pd.DataFrame:
    bm25 = load_top_k_results(
        path=bm25_path,
        system_name="bm25",
        top_k=source_top_k,
    )

    dense = load_top_k_results(
        path=dense_path,
        system_name="dense_flat",
        top_k=source_top_k,
    )

    hnsw = load_top_k_results(
        path=hnsw_path,
        system_name="hnsw",
        top_k=source_top_k,
    )

    combined = pd.concat(
        [bm25, dense, hnsw],
        ignore_index=True,
    )

    candidate_info = (
        combined
        .groupby(["query_id", "doc_id"], as_index=False)
        .agg(
            query=("query", "first"),
            category=("category", "first"),
            source_systems=(
                "source_system",
                lambda values: ", ".join(
                    sorted(set(values))
                ),
            ),
            system_count=(
                "source_system",
                lambda values: len(set(values)),
            ),
            best_rank=("rank", "min"),
        )
    )

    candidate_info = select_max_candidates_per_query(
        candidates=candidate_info,
        max_candidates=max_candidates,
    )

    corpus_columns = [
        "doc_id",
        "job_title",
        "company",
        "job_location",
        "job_level",
        "job_type",
        "job_summary",
        "job_skills",
    ]

    LOGGER.info("Loading corpus metadata from %s", corpus_path)

    corpus = pd.read_parquet(
        corpus_path,
        columns=corpus_columns,
    ).fillna("")

    pool = candidate_info.merge(
        corpus,
        on="doc_id",
        how="left",
        validate="many_to_one",
    )

    pool = pool[
        [
            "query_id",
            "query",
            "category",
            "doc_id",
            "job_title",
            "company",
            "job_location",
            "job_level",
            "job_type",
            "job_summary",
            "job_skills",
            "source_systems",
            "system_count",
            "best_rank",
        ]
    ]

    pool = pool.sort_values(
        ["query_id", "best_rank", "doc_id"]
    ).reset_index(drop=True)

    pool["relevance"] = ""
    pool["annotator"] = ""
    pool["notes"] = ""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info("Saved annotation pool to %s", output_path)
    LOGGER.info("Total pairs: %s", f"{len(pool):,}")

    counts = pool.groupby("query_id").size()

    LOGGER.info(
        "Candidates per query: min=%s, mean=%.1f, max=%s",
        int(counts.min()),
        float(counts.mean()),
        int(counts.max()),
    )

    return pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact pooled annotation set "
            "from BM25, Dense Flat and HNSW."
        )
    )

    parser.add_argument(
        "--bm25",
        type=Path,
        default=DEFAULT_BM25_PATH,
    )

    parser.add_argument(
        "--dense",
        type=Path,
        default=DEFAULT_DENSE_PATH,
    )

    parser.add_argument(
        "--hnsw",
        type=Path,
        default=DEFAULT_HNSW_PATH,
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--source-top-k",
        type=int,
        default=10,
        help="How many top documents to take from each system.",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Maximum number of pooled documents per query.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    build_annotation_pool(
        bm25_path=args.bm25,
        dense_path=args.dense,
        hnsw_path=args.hnsw,
        corpus_path=args.corpus,
        output_path=args.output,
        source_top_k=args.source_top_k,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()