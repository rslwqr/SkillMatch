from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path("data/processed/corpus_500k.parquet")
DEFAULT_QUERIES_PATH = Path("qrels/queries.csv")
DEFAULT_INDEX_PATH = Path("data/processed/bm25_index.sqlite")
DEFAULT_RESULTS_PATH = Path("results/bm25_top20.csv")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def connect_database(index_path: Path) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(index_path)

    # Faster index construction for a local experiment.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -200000")

    return connection


def create_bm25_index(
    corpus_path: Path,
    index_path: Path,
    rebuild: bool = False,
    batch_size: int = 5_000,
    limit: int | None = None,
) -> None:
    if index_path.exists() and not rebuild:
        LOGGER.info(
            "BM25 index already exists: %s",
            index_path,
        )
        return

    if index_path.exists():
        LOGGER.info("Removing old index: %s", index_path)
        index_path.unlink()

    LOGGER.info("Loading corpus from %s", corpus_path)

    columns = [
        "doc_id",
        "job_title",
        "company",
        "job_location",
        "job_level",
        "job_type",
        "job_summary",
        "job_skills",
    ]

    corpus = pd.read_parquet(
        corpus_path,
        columns=columns,
    )

    validate_columns(
        corpus,
        required_columns=set(columns),
        dataframe_name="Corpus",
    )

    if limit is not None:
        corpus = corpus.head(limit).copy()

    corpus = corpus.fillna("")

    LOGGER.info(
        "Documents to index: %s",
        f"{len(corpus):,}",
    )

    connection = connect_database(index_path)

    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE jobs_fts USING fts5(
                doc_id UNINDEXED,
                job_title,
                job_skills,
                job_summary,
                company UNINDEXED,
                job_location UNINDEXED,
                job_level UNINDEXED,
                job_type UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )

        insert_query = """
            INSERT INTO jobs_fts (
                doc_id,
                job_title,
                job_skills,
                job_summary,
                company,
                job_location,
                job_level,
                job_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        start_time = time.perf_counter()

        for start in range(0, len(corpus), batch_size):
            end = min(start + batch_size, len(corpus))
            batch = corpus.iloc[start:end]

            rows = list(
                batch[
                    [
                        "doc_id",
                        "job_title",
                        "job_skills",
                        "job_summary",
                        "company",
                        "job_location",
                        "job_level",
                        "job_type",
                    ]
                ].itertuples(index=False, name=None)
            )

            connection.executemany(insert_query, rows)
            connection.commit()

            if end % 50_000 == 0 or end == len(corpus):
                LOGGER.info(
                    "Indexed %s / %s documents",
                    f"{end:,}",
                    f"{len(corpus):,}",
                )

        elapsed = time.perf_counter() - start_time

        LOGGER.info(
            "BM25 index built in %.2f minutes",
            elapsed / 60,
        )

    finally:
        connection.close()


def prepare_fts_query(query: str) -> str:
    """
    Convert a natural-language query into a safe FTS5 OR query.

    Example:
        "Java developer with Spring Boot"
        -> "Java OR developer OR with OR Spring OR Boot"
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query)

    if not tokens:
        raise ValueError(f"Query has no searchable tokens: {query!r}")

    escaped_tokens = [
        f'"{token.replace(chr(34), chr(34) * 2)}"'
        for token in tokens
    ]

    return " OR ".join(escaped_tokens)


def search_bm25(
    connection: sqlite3.Connection,
    query: str,
    top_k: int,
) -> tuple[list[dict[str, object]], float]:
    fts_query = prepare_fts_query(query)

    sql = """
        SELECT
            doc_id,
            job_title,
            company,
            job_location,
            job_level,
            job_type,
            job_skills,
            bm25(
                jobs_fts,
                0.0,
                3.0,
                2.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0
            ) AS bm25_raw_score
        FROM jobs_fts
        WHERE jobs_fts MATCH ?
        ORDER BY bm25_raw_score ASC
        LIMIT ?
    """

    start_time = time.perf_counter()

    rows = connection.execute(
        sql,
        (fts_query, top_k),
    ).fetchall()

    latency_ms = (time.perf_counter() - start_time) * 1_000

    results = []

    for rank, row in enumerate(rows, start=1):
        (
            doc_id,
            job_title,
            company,
            job_location,
            job_level,
            job_type,
            job_skills,
            raw_score,
        ) = row

        # SQLite FTS5 usually returns negative BM25 scores.
        # We invert them only to make larger displayed scores better.
        score = -float(raw_score)

        results.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "job_title": job_title,
                "company": company,
                "job_location": job_location,
                "job_level": job_level,
                "job_type": job_type,
                "job_skills": job_skills,
            }
        )

    return results, latency_ms


def run_queries(
    queries_path: Path,
    index_path: Path,
    output_path: Path,
    top_k: int,
) -> pd.DataFrame:
    LOGGER.info("Loading queries from %s", queries_path)

    queries = pd.read_csv(queries_path)

    validate_columns(
        queries,
        required_columns={"query_id", "query", "category"},
        dataframe_name="Queries",
    )

    connection = sqlite3.connect(index_path)

    all_results: list[dict[str, object]] = []
    latencies: list[float] = []

    try:
        for row in queries.itertuples(index=False):
            results, latency_ms = search_bm25(
                connection=connection,
                query=row.query,
                top_k=top_k,
            )

            latencies.append(latency_ms)

            LOGGER.info(
                "%s | %s results | %.2f ms",
                row.query_id,
                len(results),
                latency_ms,
            )

            for result in results:
                all_results.append(
                    {
                        "query_id": row.query_id,
                        "query": row.query,
                        "category": row.category,
                        "system": "bm25",
                        "latency_ms": latency_ms,
                        **result,
                    }
                )

    finally:
        connection.close()

    results_df = pd.DataFrame(all_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    latency_series = pd.Series(latencies)

    LOGGER.info("Saved results to %s", output_path)
    LOGGER.info(
        "Mean latency: %.2f ms",
        latency_series.mean(),
    )
    LOGGER.info(
        "P95 latency: %.2f ms",
        latency_series.quantile(0.95),
    )

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query the SkillMatch BM25 index."
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
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N documents for a quick test.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the existing BM25 index.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    create_bm25_index(
        corpus_path=args.corpus,
        index_path=args.index,
        rebuild=args.rebuild,
        limit=args.limit,
    )

    run_queries(
        queries_path=args.queries,
        index_path=args.index,
        output_path=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()