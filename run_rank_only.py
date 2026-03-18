"""Standalone script: rank existing candidates and produce submission.csv.

Usage:
    uv run python run_rank_only.py

Skips prepare_data / build_features / generate_candidates.
Reads artifacts/candidates.parquet and data/ CSVs, runs the improved
CatBoostRanker, writes artifacts/predictions.parquet and artifacts/submission.csv.
"""

from __future__ import annotations

import argparse
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
from src.platform.cli.config_loader import load_config
from src.platform.core.artifacts import atomic_write_dataframe
from src.platform.core.dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_rank_only")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank existing candidates only")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/high_recall.yaml"),
        help="Path to experiment config",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=None,
        help="Optional path to candidates parquet; defaults to best available",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("artifacts/predictions.parquet"),
        help="Output predictions path",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path("artifacts/submission.csv"),
        help="Output submission path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = PROJECT_ROOT / "data"
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    default_candidates_v2 = artifacts_dir / "candidates_v2.parquet"
    default_candidates = artifacts_dir / "candidates.parquet"
    if args.candidates_path is not None:
        candidates_path = (PROJECT_ROOT / args.candidates_path).resolve()
    else:
        candidates_path = (
            default_candidates_v2 if default_candidates_v2.exists() else default_candidates
        )

    predictions_path = (PROJECT_ROOT / args.predictions_path).resolve()
    submission_path = (PROJECT_ROOT / args.submission_path).resolve()

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
    k = int(config["pipeline"]["k"])
    source_weights: dict[str, float] = {
        str(k_): float(v_) for k_, v_ in config["ranking"]["source_weights"].items()
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
