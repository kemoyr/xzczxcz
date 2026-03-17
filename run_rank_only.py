"""Standalone script: rank existing candidates and produce submission.csv.

Usage:
    uv run python run_rank_only.py

Skips prepare_data / build_features / generate_candidates.
Reads artifacts/candidates.parquet and data/ CSVs, runs the improved
CatBoostRanker, writes artifacts/predictions.parquet and artifacts/submission.csv.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition.ranking import rank_predictions
from src.platform.core.artifacts import atomic_write_dataframe
from src.platform.core.dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_rank_only")


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    candidates_path = artifacts_dir / "candidates.parquet"
    predictions_path = artifacts_dir / "predictions.parquet"
    submission_path = artifacts_dir / "submission.csv"

    # ── load dataset ────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s …", data_dir)
    t0 = time.perf_counter()
    dataset = Dataset.load(data_dir)
    logger.info(
        "Dataset loaded in %.1fs  |  interactions=%d  targets=%d  editions=%d",
        time.perf_counter() - t0,
        len(dataset.interactions_df),
        len(dataset.targets_df),
        len(dataset.catalog_df),
    )

    # ── load candidates ──────────────────────────────────────────────────────
    if not candidates_path.exists():
        logger.error("candidates.parquet not found at %s", candidates_path)
        sys.exit(1)

    logger.info("Loading candidates from %s …", candidates_path)
    candidates = pd.read_parquet(candidates_path)
    logger.info(
        "Candidates loaded: %d rows, %d users, sources=%s",
        len(candidates),
        candidates["user_id"].nunique(),
        sorted(candidates["source"].unique().tolist()),
    )

    # ── rank ─────────────────────────────────────────────────────────────────
    k = 20
    source_weights: dict[str, float] = {
        "global_popularity": 0.6,
        "global_temporal_popularity": 1.4,
        "user_genre": 1.2,
        "user_author": 1.4,
        "user_language": 1.0,
        "user_publisher": 0.8,
        "user_genre_recent": 1.5,
        "user_author_recent": 1.6,
        "post_incident_genre_profile": 1.7,
        "post_incident_author_profile": 1.8,
        "post_incident_language_profile": 1.3,
        "post_incident_publisher_profile": 1.2,
        "item_cooccurrence": 1.8,
        "svd_cf": 2.0,
    }

    logger.info("Running rank_predictions (k=%d) …", k)
    t1 = time.perf_counter()
    predictions = rank_predictions(
        dataset=dataset,
        candidates=candidates,
        source_weights=source_weights,
        k=k,
    )
    elapsed = time.perf_counter() - t1
    logger.info(
        "Ranking done in %.1fs  |  rows=%d  users=%d",
        elapsed,
        len(predictions),
        predictions["user_id"].nunique() if not predictions.empty else 0,
    )

    # ── validate output contract ─────────────────────────────────────────────
    target_users = set(dataset.targets_df["user_id"].tolist())
    pred_users = set(predictions["user_id"].tolist())
    missing_users = target_users - pred_users
    if missing_users:
        logger.warning(
            "%d target users have no predictions: %s …",
            len(missing_users),
            list(missing_users)[:5],
        )

    per_user_counts = predictions.groupby("user_id")["edition_id"].count()
    bad = per_user_counts[per_user_counts != k]
    if not bad.empty:
        logger.warning("%d users do not have exactly %d predictions.", len(bad), k)

    dupes = predictions.groupby("user_id")["edition_id"].apply(lambda x: x.duplicated().any())
    if dupes.any():
        logger.warning("Duplicate edition_ids found for some users!")

    # ── save artifacts ────────────────────────────────────────────────────────
    logger.info("Saving predictions to %s …", predictions_path)
    atomic_write_dataframe(predictions, predictions_path)

    logger.info("Saving submission to %s …", submission_path)
    submission = predictions[["user_id", "edition_id", "rank"]].sort_values(
        ["user_id", "rank"]
    )
    atomic_write_dataframe(submission, submission_path)

    logger.info(
        "Done.  submission.csv: %d rows for %d users.",
        len(submission),
        submission["user_id"].nunique(),
    )


if __name__ == "__main__":
    main()
