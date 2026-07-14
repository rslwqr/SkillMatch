from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_INPUT_PATH = Path(
    "data/embeddings/e5_small_v2_embeddings.npy"
)
DEFAULT_OUTPUT_DIR = Path("data/embeddings")
DEFAULT_MODEL_PATH = Path(
    "data/embeddings/e5_small_v2_pca.joblib"
)
DEFAULT_METRICS_DIR = Path(
    "results/dimensionality_reduction"
)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}")

    embeddings = np.load(path, mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got shape {embeddings.shape}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected float32 embeddings, got {embeddings.dtype}"
        )

    return embeddings


def fit_pca(
    embeddings: np.ndarray,
    max_dimension: int,
    sample_size: int,
    random_state: int,
) -> tuple[PCA, float, int]:
    if max_dimension >= embeddings.shape[1]:
        raise ValueError(
            "The reduced dimension must be smaller than the input dimension."
        )

    actual_sample_size = min(sample_size, embeddings.shape[0])
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(
        embeddings.shape[0],
        size=actual_sample_size,
        replace=False,
    )
    sample = np.asarray(
        embeddings[sample_indices],
        dtype=np.float32,
    )

    pca = PCA(
        n_components=max_dimension,
        svd_solver="randomized",
        random_state=random_state,
    )

    start_time = time.perf_counter()
    pca.fit(sample)
    fit_time_seconds = time.perf_counter() - start_time

    return pca, fit_time_seconds, actual_sample_size


def reduce_embeddings(
    embeddings: np.ndarray,
    pca: PCA,
    dimension: int,
    output_path: Path,
    batch_size: int,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reduced = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(embeddings.shape[0], dimension),
    )

    start_time = time.perf_counter()

    for start in range(0, embeddings.shape[0], batch_size):
        end = min(start + batch_size, embeddings.shape[0])
        batch = np.asarray(embeddings[start:end], dtype=np.float32)
        projected = pca.transform(batch)[:, :dimension]

        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        reduced[start:end] = projected / norms

    reduced.flush()
    transform_time_seconds = time.perf_counter() - start_time
    del reduced

    return transform_time_seconds


def run_reduction(
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    metrics_dir: Path,
    dimensions: list[int],
    sample_size: int,
    batch_size: int,
    random_state: int,
) -> pd.DataFrame:
    embeddings = load_embeddings(input_path)
    dimensions = sorted(set(dimensions))

    if not dimensions or min(dimensions) <= 0:
        raise ValueError("Dimensions must contain positive integers.")

    max_dimension = max(dimensions)
    pca, fit_time_seconds, actual_sample_size = fit_pca(
        embeddings=embeddings,
        max_dimension=max_dimension,
        sample_size=sample_size,
        random_state=random_state,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, model_path)

    metrics_dir.mkdir(parents=True, exist_ok=True)

    explained_variance = pd.DataFrame(
        {
            "component": np.arange(1, max_dimension + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    explained_variance.to_csv(
        metrics_dir / "pca_explained_variance.csv",
        index=False,
    )

    input_dimension = int(embeddings.shape[1])
    input_bytes_per_vector = input_dimension * np.dtype(np.float32).itemsize
    rows: list[dict[str, object]] = []

    for dimension in dimensions:
        output_path = (
            output_dir
            / f"e5_small_v2_pca{dimension}_embeddings.npy"
        )
        transform_time_seconds = reduce_embeddings(
            embeddings=embeddings,
            pca=pca,
            dimension=dimension,
            output_path=output_path,
            batch_size=batch_size,
        )

        output_bytes_per_vector = (
            dimension * np.dtype(np.float32).itemsize
        )

        rows.append(
            {
                "documents": int(embeddings.shape[0]),
                "input_dimension": input_dimension,
                "output_dimension": dimension,
                "pca_sample_size": actual_sample_size,
                "cumulative_explained_variance_ratio": float(
                    np.sum(pca.explained_variance_ratio_[:dimension])
                ),
                "input_bytes_per_vector": input_bytes_per_vector,
                "output_bytes_per_vector": output_bytes_per_vector,
                "compression_ratio": (
                    input_bytes_per_vector / output_bytes_per_vector
                ),
                "pca_fit_time_seconds": fit_time_seconds,
                "transform_time_seconds": transform_time_seconds,
                "embeddings_path": str(output_path),
                "pca_model_path": str(model_path),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        metrics_dir / "pca_reduction_metrics.csv",
        index=False,
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce E5 embedding dimensions with PCA."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[128, 256],
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_reduction(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model_output,
        metrics_dir=args.metrics_dir,
        dimensions=args.dimensions,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
