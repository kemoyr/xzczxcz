"""Post-incident publisher-profile generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class PostIncidentPublisherProfileGenerator:
    """Generate candidates from user publisher profile built on post-incident window."""

    name = "post_incident_publisher_profile"

    def __init__(
        self,
        publisher_smoothing: float = 0.3,
        show_progress: bool = False,
    ) -> None:
        self.publisher_smoothing = publisher_smoothing
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        del seed
        user_profile = features[
            features["feature_type"] == "user_publisher_profile_post_incident"
        ][["user_id", "publisher_id", "value"]].rename(columns={"value": "weight"})
        if user_profile.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        pop_df = pd.DataFrame()
        for pop_type in (
            "edition_popularity_incident",
            "edition_popularity_w30",
            "edition_popularity_weighted",
            "edition_popularity_all",
        ):
            pop_df = features[features["feature_type"] == pop_type][
                ["edition_id", "value"]
            ].rename(columns={"value": "pop"})
            if not pop_df.empty:
                break

        publisher_editions = dataset.catalog_df[["edition_id", "publisher_id"]].copy()
        if not pop_df.empty:
            publisher_editions = publisher_editions.merge(pop_df, on="edition_id", how="left")
            publisher_editions["pop"] = publisher_editions["pop"].fillna(0.0)
        else:
            publisher_editions["pop"] = 0.0

        top_per_publisher = max(k * 6, 240)
        top_publisher_editions = (
            publisher_editions.sort_values(
                ["publisher_id", "pop", "edition_id"], ascending=[True, False, True]
            )
            .groupby("publisher_id", group_keys=False)
            .head(top_per_publisher)[["publisher_id", "edition_id", "pop"]]
        )

        user_profile = user_profile[user_profile["user_id"].isin(user_ids)]
        if user_profile.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        merged = user_profile.merge(top_publisher_editions, on="publisher_id", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        merged["score"] = merged["weight"] * (merged["pop"] + self.publisher_smoothing)
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
