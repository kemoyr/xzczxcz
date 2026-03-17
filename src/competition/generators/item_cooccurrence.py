"""Item-to-item co-occurrence collaborative filtering generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class ItemCooccurrenceGenerator:
    """Retrieve candidates via item-to-item co-occurrence collaborative filtering.

    Builds a co-occurrence table from a recent interaction window: two editions
    co-occur whenever the same user interacted with both within the window.
    For each target user the generator seeds from that user's own recent
    interactions, looks up the top co-occurring editions per seed, aggregates
    co-occurrence counts across all seeds, and returns the highest-scoring
    novel candidates.

    This source captures collaborative signals absent from profile-only
    generators and is particularly effective for incident recovery because
    it surfaces items that similar users interacted with during the same period.

    Scale safeguards:
    - Co-occurrence is built from a bounded time window (``cooccurrence_days``).
    - Each user contributes at most ``max_items_per_user`` items to the matrix
      (capped by recency to keep the signal fresh and limit join size).
    - Only the top ``top_per_seed`` co-occurring items per seed are retained.
    """

    name = "item_cooccurrence"

    def __init__(
        self,
        cooccurrence_days: int = 90,
        seed_days: int = 45,
        max_items_per_user: int = 30,
        top_per_seed: int = 200,
        show_progress: bool = False,
    ) -> None:
        """Configure co-occurrence retrieval parameters.

        Args:
            cooccurrence_days: Rolling window (in days from dataset max_ts) used
                to build the item co-occurrence table.
            seed_days: Rolling window (in days from dataset max_ts) used to
                extract seed items for each target user.
            max_items_per_user: Maximum items a single user contributes to the
                co-occurrence matrix (most-recent items are kept first).
            top_per_seed: Maximum co-occurring candidates stored per seed item.
            show_progress: Retained for API compatibility.
        """
        self.cooccurrence_days = cooccurrence_days
        self.seed_days = seed_days
        self.max_items_per_user = max_items_per_user
        self.top_per_seed = top_per_seed
        self.show_progress = show_progress

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_cooccurrence(self, positives: pd.DataFrame, max_ts: pd.Timestamp) -> pd.DataFrame:
        """Build top-N co-occurrence table from a recent interaction window.

        Returns DataFrame with columns ``edition_id_seed``, ``edition_id_cand``,
        ``co_count``.
        """
        window = positives[
            positives["event_ts"] >= max_ts - pd.Timedelta(days=self.cooccurrence_days)
        ]
        # Take the most-recent `max_items_per_user` editions per user.
        user_items = (
            window.sort_values("event_ts", ascending=False)[["user_id", "edition_id"]]
            .drop_duplicates(subset=["user_id", "edition_id"])
            .groupby("user_id", group_keys=False)
            .head(self.max_items_per_user)
            .reset_index(drop=True)
        )
        if user_items.empty:
            return pd.DataFrame(columns=["edition_id_seed", "edition_id_cand", "co_count"])

        # Self-join on user to get all within-user item pairs.
        co = user_items.merge(user_items, on="user_id", suffixes=("_seed", "_cand"))
        co = co[co["edition_id_seed"] != co["edition_id_cand"]]
        if co.empty:
            return pd.DataFrame(columns=["edition_id_seed", "edition_id_cand", "co_count"])

        co_counts = (
            co.groupby(["edition_id_seed", "edition_id_cand"])
            .size()
            .reset_index(name="co_count")
        )
        # Keep only the top co-occurring candidates per seed to bound memory.
        top_co = (
            co_counts.sort_values(
                ["edition_id_seed", "co_count", "edition_id_cand"],
                ascending=[True, False, True],
            )
            .groupby("edition_id_seed", group_keys=False)
            .head(self.top_per_seed)
        )
        return top_co

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Emit top-k co-occurrence-based candidates for every target user.

        Args:
            dataset: Runtime dataset with raw interactions and seen-positive pairs.
            user_ids: Target users for candidate generation.
            features: Long feature table (unused; signals come from raw interactions).
            k: Maximum number of candidates generated per user.
            seed: Pipeline seed (unused by this deterministic generator).

        Returns:
            Candidate DataFrame with required schema and source name.
        """
        del features, seed
        positives = dataset.interactions_df[
            dataset.interactions_df["event_type"].isin([1, 2])
        ]
        if positives.empty or len(user_ids) == 0:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        max_ts = positives["event_ts"].max()

        # Build global co-occurrence table once.
        top_co = self._build_cooccurrence(positives, max_ts)
        if top_co.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        # Extract seed items for target users from a recent window.
        target_users_set = set(int(u) for u in user_ids)
        seed_window = positives[
            (positives["user_id"].isin(target_users_set))
            & (positives["event_ts"] >= max_ts - pd.Timedelta(days=self.seed_days))
        ]
        user_seeds = (
            seed_window[["user_id", "edition_id", "event_ts"]]
            .groupby(["user_id", "edition_id"], as_index=False)["event_ts"]
            .max()
            .reset_index(drop=True)
        )
        if user_seeds.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])
        days_since_seed = (max_ts - user_seeds["event_ts"]).dt.days.clip(lower=0)
        user_seeds["seed_weight"] = 1.0 / (1.0 + days_since_seed.astype(float))

        # For each seed item a user has, retrieve its top co-occurring items.
        candidates = user_seeds.merge(
            top_co, left_on="edition_id", right_on="edition_id_seed", how="inner"
        )
        if candidates.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        # Aggregate weighted co-occurrence counts across a user's seed items.
        candidates["score_part"] = candidates["co_count"] * candidates["seed_weight"]
        result = (
            candidates.groupby(["user_id", "edition_id_cand"], as_index=False)["score_part"]
            .sum()
            .rename(columns={"edition_id_cand": "edition_id", "score_part": "score"})
        )
        result["score"] = result["score"].astype(float)

        # Filter already-seen items so the budget is spent on novel candidates.
        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        result = result.merge(seen.assign(_seen=1), on=["user_id", "edition_id"], how="left")
        result = result[result["_seen"].isna()].drop(columns=["_seen"])

        result = result.sort_values(
            ["user_id", "score", "edition_id"], ascending=[True, False, True]
        )
        result = result.groupby("user_id", group_keys=False).head(k)
        result["source"] = self.name
        return result[["user_id", "edition_id", "score", "source"]].reset_index(drop=True)
