"""Fast strict local validation without candidate leakage.

Builds a pseudo-incident split, masks a fraction of positive pairs,
rebuilds features and candidates from the masked observation,
then reports candidate recall and NDCG@20.

Usage:
    uv run python validate_fast.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.competition.features import build_features_frame
from src.competition.generators import run_generators
from src.competition.ranking import rank_predictions
from src.platform.cli.config_loader import load_config
from src.platform.core.dataset import Dataset
from src.platform.core.metrics import ndcg_at_k, summarize_ndcg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("validate_fast")


# ── config ────────────────────────────────────────────────────────────────────
PSEUDO_INCIDENT_DAYS = 14
MASK_FRACTION = 0.20
SEED = 42
K = 20
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "high_recall.yaml"


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    config = load_config(CONFIG_PATH)
    recent_days = int(config["pipeline"]["recent_days"])
    per_generator_k = int(config["candidates"]["per_generator_k"])
    generators_cfg = list(config["candidates"]["generators"])
    source_weights = {
        str(k): float(v) for k, v in config["ranking"]["source_weights"].items()
    }

    # ── load dataset ─────────────────────────────────────────────────────────
    logger.info("Loading full dataset …")
    t0 = time.perf_counter()
    dataset = Dataset.load(data_dir)
    logger.info("Dataset loaded in %.1fs", time.perf_counter() - t0)

    # ── build pseudo-incident ────────────────────────────────────────────────
    positives = dataset.interactions_df[
        dataset.interactions_df["event_type"].isin([1, 2])
    ]
    min_ts = positives["event_ts"].min()
    clean_end = min_ts + pd.Timedelta(days=150)
    clean_df = positives[positives["event_ts"] < clean_end].copy()

    incident_end = clean_df["event_ts"].max()
    incident_start = incident_end - pd.Timedelta(days=PSEUDO_INCIDENT_DAYS)
    pseudo_window = clean_df[clean_df["event_ts"] >= incident_start]
    pseudo_pairs = pseudo_window[["user_id", "edition_id"]].drop_duplicates()

    rng = np.random.default_rng(SEED)
    mask_count = max(1, int(len(pseudo_pairs) * MASK_FRACTION))
    mask_idx = rng.choice(len(pseudo_pairs), size=mask_count, replace=False)
    masked_pairs = pseudo_pairs.iloc[mask_idx].copy()

    logger.info(
        "Pseudo-incident: window=[%s, %s], total_pairs=%d, masked=%d",
        incident_start.date(),
        incident_end.date(),
        len(pseudo_pairs),
        len(masked_pairs),
    )

    # ── build masked dataset ─────────────────────────────────────────────────
    observed = clean_df.merge(
        masked_pairs.assign(_m=1), on=["user_id", "edition_id"], how="left"
    )
    observed = observed[observed["_m"].isna()].drop(columns=["_m"])

    targets = pseudo_window[["user_id"]].drop_duplicates().astype({"user_id": "int64"})
    val_seen_df = observed[["user_id", "edition_id"]].drop_duplicates()

    val_dataset = Dataset(
        interactions_df=observed,
        targets_df=targets,
        catalog_df=dataset.catalog_df,
        authors_df=dataset.authors_df,
        book_genres_df=dataset.book_genres_df,
        genres_df=dataset.genres_df,
        users_df=dataset.users_df,
        seen_positive_df=val_seen_df,
    )

    # ── rebuild features + candidates from masked observation ─────────────────
    logger.info("Building validation features …")
    val_features = build_features_frame(dataset=val_dataset, recent_days=recent_days)

    logger.info("Generating validation candidates (strict, no injection) …")
    val_candidates = run_generators(
        dataset=val_dataset,
        features=val_features,
        user_ids=targets["user_id"].astype("int64"),
        generators_cfg=generators_cfg,
        per_generator_k=per_generator_k,
        seed=SEED,
        tqdm_enabled=False,
    )
    logger.info(
        "Validation candidates: %d rows for %d users, sources=%s",
        len(val_candidates),
        val_candidates["user_id"].nunique(),
        sorted(val_candidates["source"].unique().tolist()) if not val_candidates.empty else [],
    )

    # Candidate recall@M where M is merged pool size per user.
    masked_pairs_int = masked_pairs.astype({"user_id": "int64", "edition_id": "int64"})
    cand_pairs = val_candidates[["user_id", "edition_id"]].drop_duplicates()
    coverage = masked_pairs_int.merge(
        cand_pairs.assign(in_candidates=1),
        on=["user_id", "edition_id"],
        how="left",
    )
    pair_recall = float(coverage["in_candidates"].fillna(0).mean())
    user_recall = (
        coverage.groupby("user_id", as_index=False)["in_candidates"]
        .mean()
        .rename(columns={"in_candidates": "candidate_recall"})
    )
    mean_user_recall = float(user_recall["candidate_recall"].mean()) if not user_recall.empty else 0.0
    logger.info(
        "Candidate recall: pair_recall@M=%.6f, mean_user_recall@M=%.6f",
        pair_recall,
        mean_user_recall,
    )

    # ── rank ─────────────────────────────────────────────────────────────────
    logger.info("Running rank_predictions …")
    t1 = time.perf_counter()
    predictions = rank_predictions(
        dataset=val_dataset,
        candidates=val_candidates,
        source_weights=source_weights,
        k=K,
    )
    logger.info("Ranking done in %.1fs", time.perf_counter() - t1)

    # ── compute NDCG@20 ──────────────────────────────────────────────────────
    relevant_by_user: dict[int, set[int]] = {}
    for row in masked_pairs.itertuples(index=False):
        relevant_by_user.setdefault(int(row.user_id), set()).add(int(row.edition_id))

    rows: list[dict] = []
    for user_id in targets["user_id"].tolist():
        user_pred = (
            predictions[predictions["user_id"] == int(user_id)]
            .sort_values("rank")["edition_id"]
            .astype("int64")
            .tolist()
        )
        relevant = relevant_by_user.get(int(user_id), set())
        ndcg = ndcg_at_k(predicted=user_pred, relevant=relevant, k=K)
        rows.append({"user_id": int(user_id), f"ndcg@{K}": ndcg})

    per_user = pd.DataFrame(rows)
    summary = summarize_ndcg(per_user, score_column=f"ndcg@{K}")

    logger.info(
        "=== Validation result ===\n"
        "  mean NDCG@%d : %.6f\n"
        "  users         : %d\n"
        "  quantiles     : %s",
        K,
        summary.mean_ndcg,
        len(per_user),
        summary.quantiles,
    )

    # ── breakdown by user activity bucket ────────────────────────────────────
    user_activity = (
        positives.groupby("user_id")["edition_id"].nunique().rename("n_positives")
    )
    per_user = per_user.join(user_activity, on="user_id")
    per_user["bucket"] = pd.cut(
        per_user["n_positives"].fillna(0),
        bins=[0, 5, 20, 50, float("inf")],
        labels=["cold (≤5)", "light (6-20)", "medium (21-50)", "heavy (>50)"],
    )
    bucket_ndcg = per_user.groupby("bucket", observed=True)[f"ndcg@{K}"].mean()
    logger.info("NDCG@%d by activity bucket:\n%s", K, bucket_ndcg.to_string())

    print(
        f'\n{{"mean_ndcg@{K}": {summary.mean_ndcg:.6f}, '
        f'"pair_candidate_recall@M": {pair_recall:.6f}, '
        f'"mean_user_candidate_recall@M": {mean_user_recall:.6f}, '
        f'"users": {len(per_user)}, '
        f'"quantiles": {summary.quantiles}}}'
    )


if __name__ == "__main__":
    main()
