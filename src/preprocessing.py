from __future__ import annotations

import argparse
import html
import logging
import re
import time
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

RANDOM_STATE = 42

JOBS_COLUMNS = [
    "job_link",
    "job_title",
    "company",
    "job_location",
    "search_city",
    "search_country",
    "search_position",
    "job_level",
    "job_type",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def clean_text(value: object) -> str:
    """Clean text while preserving technical terms such as C++, C# and .NET."""
    if pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_group_value(value: object) -> str:
    text = clean_text(value).lower()
    return text if text else "unknown"


def load_jobs(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading job postings from %s", path)

    jobs = pd.read_csv(
        path,
        usecols=JOBS_COLUMNS,
        dtype="string",
        low_memory=False,
    )

    LOGGER.info("Loaded %s job posting rows", f"{len(jobs):,}")

    jobs = jobs.drop_duplicates()
    jobs = jobs.drop_duplicates(subset=["job_link"], keep="first")

    jobs["job_title"] = jobs["job_title"].map(clean_text)

    jobs = jobs[
        jobs["job_link"].notna()
        & jobs["job_title"].ne("")
    ].copy()

    metadata_columns = [
        "company",
        "job_location",
        "search_city",
        "search_country",
        "search_position",
        "job_level",
        "job_type",
    ]

    for column in metadata_columns:
        jobs[column] = jobs[column].map(clean_text)

    LOGGER.info(
        "Remaining rows after initial cleaning: %s",
        f"{len(jobs):,}",
    )

    return jobs


def build_sampling_stratum(jobs: pd.DataFrame) -> pd.Series:
    """
    Create broad strata preserving country, seniority and working arrangement.

    Very rare strata are later merged into a shared 'rare' group.
    """
    country = jobs["search_country"].map(normalize_group_value)
    level = jobs["job_level"].map(normalize_group_value)
    job_type = jobs["job_type"].map(normalize_group_value)

    return country + " | " + level + " | " + job_type


def proportional_stratified_sample(
    jobs: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    if sample_size >= len(jobs):
        LOGGER.warning(
            "Requested sample size is not smaller than the dataset. "
            "Using the complete dataset."
        )
        return jobs.copy()

    jobs = jobs.copy()
    jobs["_stratum"] = build_sampling_stratum(jobs)

    stratum_counts = jobs["_stratum"].value_counts()
    rare_strata = stratum_counts[stratum_counts < 20].index

    jobs.loc[jobs["_stratum"].isin(rare_strata), "_stratum"] = "rare"

    group_sizes = jobs["_stratum"].value_counts()

    allocations = (
        group_sizes / group_sizes.sum() * sample_size
    ).round().astype(int)

    allocations = allocations.clip(lower=1)
    allocations = allocations.combine(group_sizes, min)

    difference = sample_size - int(allocations.sum())

    if difference != 0:
        ordered_strata = group_sizes.sort_values(
            ascending=difference > 0
        ).index.tolist()

        index = 0

        while difference != 0:
            stratum = ordered_strata[index % len(ordered_strata)]

            if difference > 0:
                if allocations[stratum] < group_sizes[stratum]:
                    allocations[stratum] += 1
                    difference -= 1
            else:
                if allocations[stratum] > 1:
                    allocations[stratum] -= 1
                    difference += 1

            index += 1

    sampled_parts: list[pd.DataFrame] = []

    for stratum, group in jobs.groupby("_stratum", sort=False):
        n = int(allocations[stratum])

        sampled_parts.append(
            group.sample(
                n=n,
                random_state=RANDOM_STATE,
            )
        )

    sample = pd.concat(sampled_parts, ignore_index=True)

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    sample = sample.drop(columns="_stratum")

    LOGGER.info(
        "Created representative sample with %s rows",
        f"{len(sample):,}",
    )

    return sample


def load_filtered_text_table(
    path: Path,
    text_column: str,
    selected_links: set[str],
    chunksize: int = 100_000,
) -> pd.DataFrame:
    LOGGER.info("Reading %s in chunks", path.name)

    selected_chunks: list[pd.DataFrame] = []
    total_matches = 0

    reader = pd.read_csv(
        path,
        usecols=["job_link", text_column],
        dtype="string",
        chunksize=chunksize,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        filtered = chunk[chunk["job_link"].isin(selected_links)].copy()

        if not filtered.empty:
            filtered[text_column] = filtered[text_column].map(clean_text)
            selected_chunks.append(filtered)
            total_matches += len(filtered)

        if chunk_number % 10 == 0:
            LOGGER.info(
                "%s: processed %s chunks, retained %s rows",
                path.name,
                chunk_number,
                f"{total_matches:,}",
            )

    if not selected_chunks:
        return pd.DataFrame(columns=["job_link", text_column])

    result = pd.concat(selected_chunks, ignore_index=True)

    result = result.drop_duplicates(
        subset=["job_link"],
        keep="first",
    )

    LOGGER.info(
        "%s: retained %s unique rows",
        path.name,
        f"{len(result):,}",
    )

    return result


def build_search_text(row: pd.Series) -> str:
    parts = [f"Title: {row['job_title']}."]

    if row["job_summary"]:
        parts.append(f"Summary: {row['job_summary']}.")

    if row["job_skills"]:
        parts.append(f"Skills: {row['job_skills']}.")

    return " ".join(parts)


def build_corpus(
    jobs_path: Path,
    summaries_path: Path,
    skills_path: Path,
    output_path: Path,
    sample_size: int,
    final_size: int | None = None,
) -> pd.DataFrame:
    start_time = time.perf_counter()

    jobs = load_jobs(jobs_path)
    jobs = proportional_stratified_sample(jobs, sample_size)

    selected_links = set(jobs["job_link"].dropna().astype(str))

    summaries = load_filtered_text_table(
        path=summaries_path,
        text_column="job_summary",
        selected_links=selected_links,
    )

    skills = load_filtered_text_table(
        path=skills_path,
        text_column="job_skills",
        selected_links=selected_links,
    )

    corpus = jobs.merge(
        summaries,
        on="job_link",
        how="left",
        validate="one_to_one",
    )

    corpus = corpus.merge(
        skills,
        on="job_link",
        how="left",
        validate="one_to_one",
    )

    corpus["job_summary"] = corpus["job_summary"].fillna("")
    corpus["job_skills"] = corpus["job_skills"].fillna("")

    corpus = corpus[
        corpus["job_summary"].ne("")
        | corpus["job_skills"].ne("")
    ].copy()

    corpus = corpus.drop_duplicates(
        subset=[
            "job_title",
            "company",
            "job_location",
            "job_summary",
        ],
        keep="first",
    )

    if final_size is not None:
        if len(corpus) < final_size:
            raise ValueError(
                f"Not enough valid documents after cleaning: {len(corpus):,}. "
                f"Required: {final_size:,}. Increase --sample-size."
            )

        corpus = corpus.sample(
            n=final_size,
            random_state=RANDOM_STATE,
        )

    corpus = corpus.reset_index(drop=True)

    corpus.insert(
        0,
        "doc_id",
        [f"JOB_{index:07d}" for index in range(1, len(corpus) + 1)],
    )

    corpus["search_text"] = corpus.apply(
        build_search_text,
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    elapsed = time.perf_counter() - start_time

    LOGGER.info("Saved corpus to %s", output_path)
    LOGGER.info("Final corpus size: %s", f"{len(corpus):,}")
    LOGGER.info("Total preprocessing time: %.2f minutes", elapsed / 60)

    return corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SkillMatch search corpus."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/corpus_500k.parquet"),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--final-size",
        type=int,
        default=None,
        help="Exact number of documents to keep after cleaning.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    build_corpus(
        jobs_path=args.raw_dir / "linkedin_job_postings.csv",
        summaries_path=args.raw_dir / "job_summary.csv",
        skills_path=args.raw_dir / "job_skills.csv",
        output_path=args.output,
        sample_size=args.sample_size,
        final_size=args.final_size,
    )


if __name__ == "__main__":
    main()