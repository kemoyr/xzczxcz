"""SVD-based collaborative filtering generator using truncated matrix factorisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.platform.core.dataset import Dataset

_BATCH_SIZE = 400  # users per scoring batch (tune down if OOM on low-RAM machines)


class SVDCollaborativeGenerator:
    """Generate candidates via truncated SVD on the user-item interaction matrix.

    Trains a latent-factor model on all observed positive interactions with
    confidence-weighted values (reads weighted higher than wishlists). For each
    target user the generator computes dot-product scores against every item
    that has at least one observed interaction, then returns the highest-scoring
    unseen items.

    This adds a collaborative filtering signal that is completely orthogonal to
    the profile-based generators: it captures "users with similar histories
    liked this item" rather than "user likes this author / genre".

    Scoring is done in batches to keep peak memory usage bounded. Seen positives
    are filtered before the top-k truncation.
    """

    name = "svd_cf"

    def __init__(
        self,
        n_factors: int = 64,
        read_weight: float = 2.0,
        wishlist_weight: float = 1.0,
        show_progress: bool = False,
    ) -> None:
        """Configure the SVD factorisation.

        Args:
            n_factors: Number of latent dimensions (singular vectors to retain).
                Higher values capture more signal but increase memory and compute.
            read_weight: Confidence weight assigned to ``event_type=2`` (read).
            wishlist_weight: Confidence weight for ``event_type=1`` (wishlist).
            show_progress: Retained for API compatibility.
        """
        self.n_factors = n_factors
        self.read_weight = read_weight
        self.wishlist_weight = wishlist_weight
        self.show_progress = show_progress

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_matrix(
        self, positives: pd.DataFrame
    ) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Build a confidence-weighted sparse user-item matrix.

        Returns:
            Tuple of (sparse_matrix, user_id_array, item_id_array) where
            ``user_id_array[i]`` is the user_id for row ``i`` of the matrix,
            and similarly for items / columns.
        """
        weight_map = {1: self.wishlist_weight, 2: self.read_weight}
        df = positives[["user_id", "edition_id", "event_type"]].copy()
        df["w"] = df["event_type"].map(weight_map).fillna(1.0)

        # Keep max weight per (user, item) pair if duplicated.
        df = df.groupby(["user_id", "edition_id"], as_index=False)["w"].max()

        user_ids_unique = df["user_id"].unique()
        item_ids_unique = df["edition_id"].unique()
        user_index = {uid: i for i, uid in enumerate(user_ids_unique)}
        item_index = {iid: i for i, iid in enumerate(item_ids_unique)}

        rows = df["user_id"].map(user_index).to_numpy()
        cols = df["edition_id"].map(item_index).to_numpy()
        vals = df["w"].to_numpy(dtype=np.float32)

        mat = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(user_ids_unique), len(item_ids_unique)),
            dtype=np.float32,
        )
        return mat, user_ids_unique, item_ids_unique

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
        """Generate top-k SVD-scored candidates for each target user.

        Args:
            dataset: Runtime dataset with raw interactions and seen-positive pairs.
            user_ids: Target users for candidate generation.
            features: Unused; signals come from raw interactions.
            k: Maximum candidates per user after seen-filtering.
            seed: Pipeline seed (unused; SVD is deterministic up to sign).

        Returns:
            Candidate DataFrame with required schema and source name.
        """
        del features, seed
        positives = dataset.interactions_df[
            dataset.interactions_df["event_type"].isin([1, 2])
        ]
        if positives.empty or len(user_ids) == 0:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        mat, all_user_ids, all_item_ids = self._build_matrix(positives)
        user_index = {int(uid): i for i, uid in enumerate(all_user_ids)}

        # Clamp n_factors to matrix rank – 1 (svds requirement).
        n_factors = min(self.n_factors, min(mat.shape) - 1)
        if n_factors < 1:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        # svds returns smallest singular values by default; which=LM gives largest.
        U, S, Vt = svds(mat, k=n_factors, which="LM", random_state=42)
        # Absorb singular values into user factors for fast per-user scoring.
        U = (U * S).astype(np.float32)       # (n_users, n_factors)
        V = Vt.T.astype(np.float32)          # (n_items, n_factors)

        # Map target user_ids to matrix row indices.
        target_row_idx = np.array(
            [user_index.get(int(uid), -1) for uid in user_ids], dtype=np.int64
        )
        valid_mask = target_row_idx >= 0
        valid_user_ids = user_ids[valid_mask]
        valid_row_idx = target_row_idx[valid_mask]

        if len(valid_user_ids) == 0:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        seen_set: set[tuple[int, int]] = set(
            (int(r.user_id), int(r.edition_id)) for r in seen.itertuples(index=False)
        )

        rows_out: list[dict[str, object]] = []
        fetch_k = max(k * 5, k + 200)  # absorb seen-item losses on active users

        for batch_start in range(0, len(valid_user_ids), _BATCH_SIZE):
            batch_end = min(batch_start + _BATCH_SIZE, len(valid_user_ids))
            batch_uids = valid_user_ids[batch_start:batch_end]
            batch_rows = valid_row_idx[batch_start:batch_end]

            # (batch, n_factors) @ (n_factors, n_items) → (batch, n_items)
            batch_scores = U[batch_rows] @ V.T

            for i, uid in enumerate(batch_uids):
                uid_int = int(uid)
                user_scores = batch_scores[i]

                # Efficient top-k via argpartition (avoids full sort).
                fetch = min(fetch_k, len(user_scores))
                if fetch == len(user_scores):
                    top_idx = np.argsort(user_scores)[::-1]
                else:
                    top_idx = np.argpartition(user_scores, -fetch)[-fetch:]
                    top_idx = top_idx[np.argsort(user_scores[top_idx])[::-1]]

                count = 0
                for idx in top_idx:
                    eid = int(all_item_ids[idx])
                    if (uid_int, eid) in seen_set:
                        continue
                    rows_out.append(
                        {
                            "user_id": uid_int,
                            "edition_id": eid,
                            "score": float(user_scores[idx]),
                            "source": self.name,
                        }
                    )
                    count += 1
                    if count >= k:
                        break

        if not rows_out:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        result = pd.DataFrame(rows_out)
        result = result.sort_values(
            ["user_id", "score", "edition_id"], ascending=[True, False, True]
        )
        return result[["user_id", "edition_id", "score", "source"]].reset_index(drop=True)
