"""Global popularity generator for participant solution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class GlobalPopularityGenerator:
    """Recommend globally popular editions for each target user.

    The generator serves as a robust baseline source and is used as fallback
    recall in blended ranking. It scores items by unique-user popularity from
    precomputed feature tables and broadcasts top-k popular editions to all
    requested users. Seen positives are filtered before truncation to k so the
    per-generator budget is spent entirely on novel candidate items.
    """

    name = "global_popularity"

    def __init__(self, show_progress: bool = False) -> None:
        """Initialize progress behavior for user-level iteration.

        Args:
            show_progress: Whether to render tqdm bars in interactive sessions.
        """
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Generate candidate rows from global popularity statistics.

        Broadcasts top globally popular editions to all users. Seen positives
        are filtered per-user before the final top-k truncation to maximise
        candidate diversity within the budget.

        Args:
            dataset: Runtime dataset providing seen-positive pairs for filtering.
            user_ids: Users for whom candidates must be emitted.
            features: Long feature table containing `edition_popularity_all`.
            k: Maximum candidate count per user after seen-filtering.
            seed: Pipeline seed (unused by this deterministic generator).

        Returns:
            Candidate DataFrame with `user_id`, `edition_id`, `score`, `source`.
        """
        del seed
        popularity = features[features["feature_type"] == "edition_popularity_all"][
            ["edition_id", "value"]
        ].copy()
        pool_size = max(k * 20, 1000)
        popularity = popularity.sort_values(
            ["value", "edition_id"], ascending=[False, True]
        ).head(pool_size)

        if popularity.empty or len(user_ids) == 0:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        n_users = len(user_ids)
        n_items = len(popularity)
        result = pd.DataFrame(
            {
                "user_id": np.repeat(user_ids, n_items),
                "edition_id": np.tile(popularity["edition_id"].to_numpy(), n_users),
                "score": np.tile(popularity["value"].to_numpy(), n_users),
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

