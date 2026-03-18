"""Generate new candidates (v2) and produce a fresh submission without
touching any existing artifacts.

Outputs:
    artifacts/candidates_v2.parquet   — new candidate pool (all 14 generators)
    artifacts/predictions_v2.parquet  — ranked predictions
    artifacts/submission_v2.csv       — submission ready for upload

Usage:
    uv run python run_full_v2.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition.features import build_features_frame
from src.competition.generators import run_generators
from src.competition.ranking import rank_predictions
from src.platform.cli.config_loader import load_config
from src.platform.core.artifacts import atomic_write_dataframe
from src.platform.core.dataset import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_full_v2")

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "high_recall.yaml"

CANDIDATES_V2_PATH = PROJECT_ROOT / "artifacts" / "candidates_v2.parquet"
PREDICTIONS_V2_PATH = PROJECT_ROOT / "artifacts" / "predictions_v2.parquet"
SUBMISSION_V2_PATH = PROJECT_ROOT / "artifacts" / "submission_v2.csv"


def main() -> None:
    config = load_config(CONFIG_PATH)
    recent_days = int(config["pipeline"]["recent_days"])
    per_generator_k = int(config["candidates"]["per_generator_k"])
    generators_cfg = list(config["candidates"]["generators"])
    seed = int(config["pipeline"]["seed"])
    k = int(config["pipeline"]["k"])
    source_weights = {
        str(key): float(val)
        for key, val in config["ranking"]["source_weights"].items()
    }

    # ── load dataset ─────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    t0 = time.perf_counter()
    dataset = Dataset.load(PROJECT_ROOT / "data")
    logger.info(
        "Dataset loaded in %.1fs  |  interactions=%d  targets=%d  editions=%d",
        time.perf_counter() - t0,
        len(dataset.interactions_df),
        len(dataset.targets_df),
        len(dataset.catalog_df),
    )

    # ── build features ───────────────────────────────────────────────────────
    logger.info("Building features (recent_days=%d) …", recent_days)
    t1 = time.perf_counter()
    features = build_features_frame(dataset=dataset, recent_days=recent_days)
    logger.info(
        "Features built in %.1fs  |  rows=%d  types=%d",
        time.perf_counter() - t1,
        len(features),
        features["feature_type"].nunique(),
    )
    logger.info(
        "Feature types: %s",
        sorted(features["feature_type"].unique().tolist()),
    )

    # ── generate candidates → candidates_v2.parquet ──────────────────────────
    logger.info(
        "Generating candidates (per_generator_k=%d, %d generators) …",
        per_generator_k,
        len(generators_cfg),
    )
    t2 = time.perf_counter()
    user_ids = dataset.targets_df["user_id"].astype("int64")
    candidates = run_generators(
        dataset=dataset,
        features=features,
        user_ids=user_ids,
        generators_cfg=generators_cfg,
        per_generator_k=per_generator_k,
        seed=seed,
        tqdm_enabled=True,
    )
    logger.info(
        "Candidates generated in %.1fs  |  rows=%d  users=%d  sources=%s",
        time.perf_counter() - t2,
        len(candidates),
        candidates["user_id"].nunique(),
        sorted(candidates["source"].unique().tolist()),
    )

    logger.info("Saving candidates to %s …", CANDIDATES_V2_PATH)
    atomic_write_dataframe(candidates, CANDIDATES_V2_PATH)

    # ── rank → predictions_v2 + submission_v2 ────────────────────────────────
    logger.info("Running rank_predictions (k=%d) …", k)
    t3 = time.perf_counter()
    predictions = rank_predictions(
        dataset=dataset,
        candidates=candidates,
        source_weights=source_weights,
        k=k,
    )
    logger.info(
        "Ranking done in %.1fs  |  rows=%d  users=%d",
        time.perf_counter() - t3,
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

    # ── save ─────────────────────────────────────────────────────────────────
    logger.info("Saving predictions to %s …", PREDICTIONS_V2_PATH)
    atomic_write_dataframe(predictions, PREDICTIONS_V2_PATH)

    submission = predictions[["user_id", "edition_id", "rank"]].sort_values(
        ["user_id", "rank"]
    )
    logger.info("Saving submission to %s …", SUBMISSION_V2_PATH)
    atomic_write_dataframe(submission, SUBMISSION_V2_PATH)

    logger.info(
        "Done.  submission_v2.csv: %d rows for %d users.",
        len(submission),
        submission["user_id"].nunique(),
    )


if __name__ == "__main__":
    main()
