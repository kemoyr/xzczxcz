"""Global temporal popularity generator combining multiple time-window signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset

_SIGNAL_ORDER = [
    "edition_popularity_incident",
    "edition_popularity_post_incident",
    "edition_popularity_w30",
    "edition_popularity_w14",
    "edition_popularity_trend_short_long",
    "edition_popularity_weighted",
    "edition_popularity_w90",
]


class GlobalTemporalPopularityGenerator:
    """Blend multiple temporal popularity signals for global candidate generation.

    Targets the incident-window recovery task by combining incident-period,
    post-incident, recent-window, and trending popularity signals into a single
    composite score. Each signal is min-max normalised before weighting so that
    different raw scales do not distort the blend. Items that were popular
    specifically around the incident window are boosted relative to all-time
    head items.

    All signals are normalised to [0, 1] per feature before weighting, then
    summed. Seen positives are filtered before the top-k broadcast to avoid
    wasting budget on already-known items.
    """

    name = "global_temporal_popularity"

    def __init__(
        self,
        w_incident: float = 2.0,
        w_post_incident: float = 1.5,
        w_w30: float = 1.0,
        w_w14: float = 0.8,
        w_trend: float = 0.6,
        w_weighted: float = 0.5,
        w_w90: float = 0.4,
        show_progress: bool = False,
    ) -> None:
        """Configure per-signal blend weights.

        Args:
            w_incident: Weight for incident-window popularity signal.
            w_post_incident: Weight for post-incident popularity signal.
            w_w30: Weight for 30-day rolling popularity.
            w_w14: Weight for 14-day rolling popularity.
            w_trend: Weight for short/long popularity trend ratio.
            w_weighted: Weight for read+wishlist weighted popularity.
            w_w90: Weight for 90-day rolling popularity.
            show_progress: Retained for API compatibility.
        """
        self.signal_weights = {
            "edition_popularity_incident": w_incident,
            "edition_popularity_post_incident": w_post_incident,
            "edition_popularity_w30": w_w30,
            "edition_popularity_w14": w_w14,
            "edition_popularity_trend_short_long": w_trend,
            "edition_popularity_weighted": w_weighted,
            "edition_popularity_w90": w_w90,
        }
        self.show_progress = show_progress

    @staticmethod
    def _normalize_min_max(values: pd.Series) -> pd.Series:
        """Scale a numeric series to [0, 1] with safe constant handling."""
        min_val = float(values.min())
        max_val = float(values.max())
        if max_val <= min_val:
            return pd.Series(np.ones(len(values), dtype=float), index=values.index)
        return (values - min_val) / (max_val - min_val)

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Generate candidates by blending temporal popularity signals.

        Args:
            dataset: Runtime dataset providing seen-positive pairs for filtering.
            user_ids: Users for whom candidates must be emitted.
            features: Long feature table containing temporal popularity features.
            k: Maximum candidate count per user after seen-filtering.
            seed: Pipeline seed (unused by this deterministic generator).

        Returns:
            Candidate DataFrame with `user_id`, `edition_id`, `score`, `source`.
        """
        del seed
        parts: list[pd.DataFrame] = []
        for feature_type in _SIGNAL_ORDER:
            weight = float(self.signal_weights.get(feature_type, 0.0))
            if weight <= 0.0:
                continue
            mask = features["feature_type"] == feature_type
            if not mask.any():
                continue
            pop_df = features[mask][["edition_id", "value"]].copy()
            if pop_df.empty:
                continue
            pop_df["score"] = self._normalize_min_max(pop_df["value"]).astype(float) * weight
            parts.append(pop_df[["edition_id", "score"]])

        if not parts or len(user_ids) == 0:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        combined = (
            pd.concat(parts, ignore_index=True)
            .groupby("edition_id", as_index=False)["score"]
            .sum()
        )
        if combined.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        pool_size = min(len(combined), max(k * 20, 1000))
        combined = combined.sort_values(
            ["score", "edition_id"], ascending=[False, True]
        ).head(pool_size)

        n_users = len(user_ids)
        n_items = len(combined)
        result = pd.DataFrame(
            {
                "user_id": np.repeat(user_ids, n_items),
                "edition_id": np.tile(combined["edition_id"].to_numpy(), n_users),
                "score": np.tile(combined["score"].to_numpy(), n_users),
                "source": self.name,
            }
        )

        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        result = result.merge(seen.assign(_seen=1), on=["user_id", "edition_id"], how="left")
        result = result[result["_seen"].isna()].drop(columns=["_seen"])

        result = result.sort_values(
            ["user_id", "score", "edition_id"], ascending=[True, False, True]
        )
        result = result.groupby("user_id", group_keys=False).head(k)
        return result[["user_id", "edition_id", "score", "source"]].reset_index(drop=True)
