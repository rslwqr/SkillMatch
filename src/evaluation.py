from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_QRELS_PATH = Path("qrels/annotation_pool.csv")
DEFAULT_OUTPUT_DIR = Path("results/evaluation")
ALLOWED_RELEVANCE_GRADES = {0.0, 1.0, 2.0, 3.0}
DEFAULT_RUNS = {
    "bm25": Path("results/bm25_top20.csv"),
    "dense_flat": Path("results/dense_flat_top20.csv"),
    "hnsw": Path(
        "results/hnsw/hnsw_m32_efc200_efs64_top20.csv"
    ),
}


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
            f"{dataframe_name} is missing columns: {sorted(missing)}"
        )


def load_qrels(
    path: Path,
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    """Load expert judgements and validate one grade per query-document pair."""
    LOGGER.info("Loading expert judgements from %s", path)

    qrels = pd.read_csv(path, dtype={"query_id": "string", "doc_id": "string"})
    validate_columns(
        qrels,
        {"query_id", "doc_id", "relevance"},
        "Qrels",
    )

    raw_relevance = qrels["relevance"]
    incomplete = (
        raw_relevance.isna()
        | raw_relevance.astype("string").str.strip().eq("")
    )
    numeric_relevance = pd.to_numeric(
        raw_relevance,
        errors="coerce",
    )

    malformed = numeric_relevance.isna() & ~incomplete
    if malformed.any():
        invalid_values = sorted(
            raw_relevance.loc[malformed].astype(str).unique().tolist()
        )
        raise ValueError(
            "Relevance grades must be numeric. "
            f"Invalid values: {invalid_values}"
        )

    qrels["relevance"] = numeric_relevance
    if incomplete.any():
        message = (
            f"Qrels contain {int(incomplete.sum()):,} unlabelled pairs "
            f"out of {len(qrels):,}."
        )
        if not allow_incomplete:
            raise ValueError(
                message
                + " Finish the annotation or pass --allow-incomplete "
                "for a provisional evaluation."
            )
        LOGGER.warning("%s They will be treated as relevance 0.", message)
        qrels.loc[incomplete, "relevance"] = 0.0

    invalid_grades = qrels.loc[
        ~qrels["relevance"].isin(ALLOWED_RELEVANCE_GRADES),
        "relevance",
    ]
    if not invalid_grades.empty:
        invalid_values = sorted(invalid_grades.unique().tolist())
        raise ValueError(
            "Relevance grades must be one of 0, 1, 2 or 3. "
            f"Invalid values: {invalid_values}"
        )

    pair_columns = ["query_id", "doc_id"]
    duplicates = qrels.duplicated(pair_columns, keep=False)
    if duplicates.any():
        conflicting = (
            qrels.loc[duplicates]
            .groupby(pair_columns)["relevance"]
            .nunique()
        )
        if (conflicting > 1).any():
            raise ValueError(
                "Qrels contain conflicting relevance grades for the same "
                "query-document pair."
            )
        qrels = qrels.drop_duplicates(pair_columns, keep="first")

    if qrels.empty:
        raise ValueError("Qrels do not contain any query-document pairs.")

    qrels["relevance"] = qrels["relevance"].astype(float)
    return qrels


def load_run(path: Path, system_name: str) -> pd.DataFrame:
    LOGGER.info("Loading %s results from %s", system_name, path)

    run = pd.read_csv(path, dtype={"query_id": "string", "doc_id": "string"})
    validate_columns(
        run,
        {"query_id", "doc_id", "rank"},
        f"Run {system_name!r}",
    )

    run["rank"] = pd.to_numeric(run["rank"], errors="coerce")
    if run["rank"].isna().any() or (run["rank"] < 1).any():
        raise ValueError(
            f"Run {system_name!r} contains invalid ranks. "
            "Ranks must be positive numbers."
        )

    duplicate_docs = run.duplicated(["query_id", "doc_id"], keep=False)
    if duplicate_docs.any():
        raise ValueError(
            f"Run {system_name!r} contains duplicate documents within a query."
        )

    duplicate_ranks = run.duplicated(["query_id", "rank"], keep=False)
    if duplicate_ranks.any():
        raise ValueError(
            f"Run {system_name!r} contains duplicate ranks within a query."
        )

    run = run.copy()
    run["system"] = system_name
    return run.sort_values(["query_id", "rank"]).reset_index(drop=True)


def dcg(relevances: np.ndarray) -> float:
    """Discounted cumulative gain with exponential gain."""
    if relevances.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, relevances.size + 2))
    gains = np.power(2.0, relevances) - 1.0
    return float(np.sum(gains / discounts))


def metrics_at_k(
    retrieved_relevance: np.ndarray,
    judged_relevance: np.ndarray,
    k: int,
    relevance_threshold: float,
) -> dict[str, float]:
    top_relevance = retrieved_relevance[:k]

    ideal_relevance = np.sort(judged_relevance)[::-1][:k]
    ideal_dcg = dcg(ideal_relevance)
    ndcg = dcg(top_relevance) / ideal_dcg if ideal_dcg > 0 else 0.0

    relevant_positions = np.flatnonzero(
        top_relevance >= relevance_threshold
    )
    reciprocal_rank = (
        1.0 / float(relevant_positions[0] + 1)
        if relevant_positions.size
        else 0.0
    )

    # The denominator remains k by the standard Precision@k definition.
    precision = float(
        np.sum(top_relevance >= relevance_threshold) / k
    )
    relevant_judged = int(
        np.sum(judged_relevance >= relevance_threshold)
    )
    recall = (
        float(
            np.sum(top_relevance >= relevance_threshold)
            / relevant_judged
        )
        if relevant_judged > 0
        else 0.0
    )

    return {
        f"ndcg_at_{k}": ndcg,
        f"mrr_at_{k}": reciprocal_rank,
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
    }


def evaluate_run(
    run: pd.DataFrame,
    qrels: pd.DataFrame,
    system_name: str,
    k: int = 10,
    relevance_threshold: float = 2.0,
) -> pd.DataFrame:
    """Calculate query-level metrics for one retrieval run."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not 0 <= relevance_threshold <= 3:
        raise ValueError("relevance_threshold must be between 0 and 3.")

    relevance_lookup = qrels.set_index(
        ["query_id", "doc_id"]
    )["relevance"]
    query_metadata_columns = [
        column for column in ["query", "category"] if column in qrels.columns
    ]
    query_metadata = (
        qrels.groupby("query_id", as_index=True)[query_metadata_columns].first()
        if query_metadata_columns
        else pd.DataFrame(index=qrels["query_id"].drop_duplicates())
    )

    rows: list[dict[str, object]] = []

    for query_id, query_qrels in qrels.groupby("query_id", sort=False):
        query_run = run[run["query_id"] == query_id].sort_values("rank")
        top_run = query_run.head(k)

        retrieved_relevance = np.asarray(
            [
                relevance_lookup.get((query_id, doc_id), 0.0)
                for doc_id in top_run["doc_id"]
            ],
            dtype=np.float64,
        )
        judged_pairs = set(
            zip(query_qrels["query_id"], query_qrels["doc_id"])
        )
        judged_at_k = sum(
            (query_id, doc_id) in judged_pairs
            for doc_id in top_run["doc_id"]
        )
        judged_relevance = query_qrels["relevance"].to_numpy(
            dtype=np.float64
        )

        row: dict[str, object] = {
            "system": system_name,
            "query_id": query_id,
            "retrieved_at_k": len(top_run),
            "judged_at_k": judged_at_k,
            "judgement_coverage_at_k": (
                judged_at_k / len(top_run) if len(top_run) > 0 else 0.0
            ),
            "judged_documents": len(query_qrels),
            "relevant_judged_documents": int(
                np.sum(judged_relevance >= relevance_threshold)
            ),
            **metrics_at_k(
                retrieved_relevance,
                judged_relevance,
                k,
                relevance_threshold,
            ),
        }

        if query_id in query_metadata.index:
            for column in query_metadata_columns:
                row[column] = query_metadata.at[query_id, column]

        if "latency_ms" in query_run.columns and not query_run.empty:
            latency = pd.to_numeric(
                query_run["latency_ms"], errors="coerce"
            ).dropna()
            row["latency_ms"] = (
                float(latency.iloc[0]) if not latency.empty else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_metrics(
    per_query: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    metric_columns = [
        f"ndcg_at_{k}",
        f"mrr_at_{k}",
        f"precision_at_{k}",
        f"recall_at_{k}",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "query_count": ("query_id", "nunique"),
        "mean_judgement_coverage_at_k": (
            "judgement_coverage_at_k",
            "mean",
        ),
        **{
            column: (column, "mean")
            for column in metric_columns
        },
    }

    if "latency_ms" in per_query.columns:
        aggregations.update(
            {
                "mean_latency_ms": ("latency_ms", "mean"),
                "p95_latency_ms": (
                    "latency_ms",
                    lambda values: values.quantile(0.95),
                ),
            }
        )

    return (
        per_query.groupby("system", as_index=False)
        .agg(**aggregations)
        .sort_values(f"ndcg_at_{k}", ascending=False)
        .reset_index(drop=True)
    )


def summarise_by_category(
    per_query: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    if "category" not in per_query.columns:
        return pd.DataFrame()

    metric_columns = [
        f"ndcg_at_{k}",
        f"mrr_at_{k}",
        f"precision_at_{k}",
        f"recall_at_{k}",
    ]
    return (
        per_query.groupby(["system", "category"], as_index=False)
        .agg(
            query_count=("query_id", "nunique"),
            **{column: (column, "mean") for column in metric_columns},
        )
        .sort_values(["category", f"ndcg_at_{k}"], ascending=[True, False])
        .reset_index(drop=True)
    )


def evaluate_all(
    qrels_path: Path,
    runs: dict[str, Path],
    output_dir: Path,
    k: int,
    relevance_threshold: float,
    allow_incomplete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qrels = load_qrels(qrels_path, allow_incomplete=allow_incomplete)
    evaluations = []

    for system_name, run_path in runs.items():
        run = load_run(run_path, system_name)
        evaluations.append(
            evaluate_run(
                run=run,
                qrels=qrels,
                system_name=system_name,
                k=k,
                relevance_threshold=relevance_threshold,
            )
        )

    per_query = pd.concat(evaluations, ignore_index=True)
    incomplete_coverage = per_query["judgement_coverage_at_k"] < 1.0
    if incomplete_coverage.any():
        LOGGER.warning(
            "%s of %s system-query results have less than complete "
            "judgement coverage at %s. Unjudged documents are treated "
            "as non-relevant; inspect per_query_metrics.csv before "
            "interpreting the comparison.",
            int(incomplete_coverage.sum()),
            len(per_query),
            k,
        )
    summary = summarise_metrics(per_query, k)
    by_category = summarise_by_category(per_query, k)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(output_dir / "per_query_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    if not by_category.empty:
        by_category.to_csv(
            output_dir / "category_metrics.csv",
            index=False,
        )

    LOGGER.info("Saved evaluation results to %s", output_dir)
    return summary, per_query, by_category


def parse_run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Run must have the form SYSTEM=PATH."
        )

    system_name, raw_path = value.split("=", maxsplit=1)
    if not system_name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Run must have a non-empty system name and path."
        )

    return system_name.strip(), Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SkillMatch retrieval runs using expert relevance "
            "judgements."
        )
    )
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_argument,
        help=(
            "Retrieval run as SYSTEM=PATH. Repeat for multiple systems. "
            "If omitted, the standard BM25, Dense Flat and HNSW files are used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=2.0,
        help=(
            "Minimum expert grade considered relevant for MRR, Precision "
            "and Recall. Defaults to 2 on the 0-3 relevance scale."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Run a provisional evaluation by treating unlabelled pooled "
            "documents as non-relevant."
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    runs = dict(args.run) if args.run else DEFAULT_RUNS
    if not runs:
        raise ValueError("At least one retrieval run is required.")

    summary, _, _ = evaluate_all(
        qrels_path=args.qrels,
        runs=runs,
        output_dir=args.output_dir,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        allow_incomplete=args.allow_incomplete,
    )

    print("\nSkillMatch evaluation summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
