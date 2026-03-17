"""User recent author affinity generator using short-term interaction profile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class UserAuthorRecentGenerator:
    """Generate candidates from a user's recent (short-window) author preferences.

    Uses the `user_author_profile_recent` feature (built over the last
    `recent_days` of observed history, default 30 days) paired with a
    temporal popularity prior. Recent author affinity captures short-term
    reading patterns more relevant to incident-window recovery than an all-time
    profile. Seen positives are filtered before top-k truncation.
    """

    name = "user_author_recent"

    def __init__(self, author_smoothing: float = 0.5, show_progress: bool = False) -> None:
        """Store hyperparameters controlling recent author scoring.

        Args:
            author_smoothing: Additive prior applied to per-edition popularity.
            show_progress: Retained for API compatibility.
        """
        self.author_smoothing = author_smoothing
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Emit top-k recent-author-driven candidates for every target user.

        Args:
            dataset: Runtime dataset containing edition-to-author mapping.
            user_ids: Target users for candidate generation.
            features: Long feature table with `user_author_profile_recent`.
            k: Maximum number of candidates generated per user.
            seed: Pipeline seed (unused by this deterministic generator).

        Returns:
            Candidate DataFrame with required schema and source name.
        """
        del seed
        user_profile = features[features["feature_type"] == "user_author_profile_recent"][
            ["user_id", "author_id", "value"]
        ].rename(columns={"value": "weight"})
        if user_profile.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        pop_df = pd.DataFrame()
        for pop_type in (
            "edition_popularity_w30",
            "edition_popularity_weighted",
            "edition_popularity_all",
        ):
            pop_df = features[features["feature_type"] == pop_type][
                ["edition_id", "value"]
            ].rename(columns={"value": "pop"})
            if not pop_df.empty:
                break

        author_editions = dataset.catalog_df[["edition_id", "author_id"]].copy()
        if not pop_df.empty:
            author_editions = author_editions.merge(pop_df, on="edition_id", how="left")
            author_editions["pop"] = author_editions["pop"].fillna(0.0)
        else:
            author_editions["pop"] = 0.0

        top_per_author = max(k * 5, 200)
        top_author_editions = (
            author_editions.sort_values(
                ["author_id", "pop", "edition_id"], ascending=[True, False, True]
            )
            .groupby("author_id", group_keys=False)
            .head(top_per_author)[["author_id", "edition_id", "pop"]]
        )

        user_profile = user_profile[user_profile["user_id"].isin(user_ids)]
        if user_profile.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        merged = user_profile.merge(top_author_editions, on="author_id", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        merged["score"] = merged["weight"] * (merged["pop"] + self.author_smoothing)
        result = merged.groupby(["user_id", "edition_id"], as_index=False)["score"].sum()

        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        result = result.merge(seen.assign(_seen=1), on=["user_id", "edition_id"], how="left")
        result = result[result["_seen"].isna()].drop(columns=["_seen"])

        result = result.sort_values(
            ["user_id", "score", "edition_id"], ascending=[True, False, True]
        )
        result = result.groupby("user_id", group_keys=False).head(k)
        result["source"] = self.name
        return result[["user_id", "edition_id", "score", "source"]].reset_index(drop=True)
