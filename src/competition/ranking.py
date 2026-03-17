"""Ranking logic for participant solution outputs."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset

logger = logging.getLogger(__name__)

# Hardcoded incident boundary matching features.py constants – used to define
# the pseudo-incident training window for the CatBoost ranker.
_INCIDENT_START_TS = pd.Timestamp("2025-10-01 00:00:00")
_PSEUDO_INCIDENT_DAYS = 14  # days before incident start used as pseudo ground-truth


class SimpleBlendRanker:
    """Blend sources with weighted scores and enforce top-k per user.

    The ranker combines candidate sources by weighted max aggregation, filters
    already seen positives, and fills missing slots with global-popularity
    fallback to preserve a valid submission shape for every target user.
    """

    def __init__(self, source_weights: dict[str, float] | None = None) -> None:
        """Capture source-level blend multipliers from experiment config.

        Args:
            source_weights: Optional mapping from source name to multiplicative
                score weight. Missing sources default to weight 1.0.
        """
        self.source_weights = source_weights or {}

    def _apply_weights(self, candidates: pd.DataFrame) -> pd.DataFrame:
        weighted = candidates.copy()
        weighted["weight"] = weighted["source"].map(self.source_weights).fillna(1.0)
        weighted["final_score"] = weighted["score"] * weighted["weight"]
        return weighted

    def rank(self, dataset: Dataset, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """Rank candidates and produce exactly top-k rows per user.

        Args:
            dataset: Runtime dataset with targets and seen-positive pairs.
            candidates: Candidate frame merged across generator sources.
            k: Required output cutoff per user.

        Returns:
            DataFrame with `user_id`, `edition_id`, `rank`, `final_score`.
        """
        if candidates.empty:
            return self._fallback_only(dataset, k)

        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        filtered = candidates.merge(
            seen.assign(_seen=1),
            on=["user_id", "edition_id"],
            how="left",
        )
        filtered = filtered[filtered["_seen"].isna()].drop(columns=["_seen"])
        if filtered.empty:
            return self._fallback_only(dataset, k)

        filtered = self._apply_weights(filtered)
        blended = (
            filtered.groupby(["user_id", "edition_id"], as_index=False)["final_score"]
            .max()
            .sort_values(["user_id", "final_score", "edition_id"], ascending=[True, False, True])
        )

        selected = blended.groupby("user_id", group_keys=False).head(k).copy()
        selected["rank"] = selected.groupby("user_id").cumcount() + 1
        selected = selected[["user_id", "edition_id", "rank", "final_score"]]

        completed = self._apply_fallback(selected, dataset, k)
        return completed.sort_values(["user_id", "rank"]).reset_index(drop=True)

    def _fallback_only(self, dataset: Dataset, k: int) -> pd.DataFrame:
        rows: list[dict[str, int | float]] = []
        positives = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])]
        popularity = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "pop"})
            .sort_values(["pop", "edition_id"], ascending=[False, True])
        )
        ranked_editions = popularity["edition_id"].tolist()
        seen_pairs = set(
            tuple(x)
            for x in dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates().to_numpy()
        )
        for user_id in dataset.targets_df["user_id"].tolist():
            rank = 1
            for edition_id in ranked_editions:
                if (int(user_id), int(edition_id)) in seen_pairs:
                    continue
                rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(edition_id),
                        "rank": rank,
                        "final_score": 0.0,
                    }
                )
                rank += 1
                if rank > k:
                    break
        return pd.DataFrame(rows)

    def _apply_fallback(
        self,
        selected: pd.DataFrame,
        dataset: Dataset,
        k: int,
    ) -> pd.DataFrame:
        positives = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])]
        popularity = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "pop"})
            .sort_values(["pop", "edition_id"], ascending=[False, True])
        )
        popular_editions = popularity["edition_id"].tolist()
        seen_pairs = set(
            tuple(x)
            for x in dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates().to_numpy()
        )
        chosen_pairs = set(tuple(x) for x in selected[["user_id", "edition_id"]].to_numpy())
        missing_rows: list[dict[str, int | float]] = []
        by_user_counts = selected.groupby("user_id").size().to_dict()

        for user_id in dataset.targets_df["user_id"].tolist():
            count = int(by_user_counts.get(int(user_id), 0))
            rank = count + 1
            if count >= k:
                continue
            for edition_id in popular_editions:
                pair = (int(user_id), int(edition_id))
                if pair in chosen_pairs or pair in seen_pairs:
                    continue
                missing_rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(edition_id),
                        "rank": rank,
                        "final_score": 0.0,
                    }
                )
                chosen_pairs.add(pair)
                rank += 1
                if rank > k:
                    break
        if missing_rows:
            selected = pd.concat([selected, pd.DataFrame(missing_rows)], ignore_index=True)
        return selected


class CatBoostRanker:
    """Re-rank candidates with a CatBoost binary classifier trained on-the-fly.

    The model is trained using a pseudo-incident setup derived from the clean
    interaction history:
    - **Pseudo-positives**: (user, item) pairs observed in the last
      ``pseudo_incident_days`` of clean history (before the real incident
      window) that also appear in the candidate pool.
    - **Pseudo-negatives**: all other candidates for the same users.

    Feature matrix for each (user, item) candidate:
    1. **Source scores** – one column per generator source (0 when absent).
    2. **n_sources** – how many generators surfaced this item for this user.
    3. **User activity features** – total interactions, recent-window count,
       days since last event.
    4. **Item popularity features** – all-time, recent, and incident-window
       unique-user counts.
    5. **Item catalogue attributes** – publication year, age restriction,
       language (treated as categorical by CatBoost).
    6. **Cross (affinity) features** – user–author, user–language and
       user–genre overlap, computed from raw interaction history.

    The trained model is applied to all candidates; top-k per user are selected
    and the global-popularity fallback fills any gaps.
    """

    def __init__(
        self,
        pseudo_incident_days: int = _PSEUDO_INCIDENT_DAYS,
        catboost_iterations: int = 300,
        catboost_depth: int = 6,
        catboost_lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.pseudo_incident_days = pseudo_incident_days
        self.catboost_iterations = catboost_iterations
        self.catboost_depth = catboost_depth
        self.catboost_lr = catboost_lr
        self.seed = seed

    # ------------------------------------------------------------------
    # feature construction helpers
    # ------------------------------------------------------------------

    def _source_score_features(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Pivot source scores to wide format and count contributing sources."""
        pivoted = (
            candidates.pivot_table(
                index=["user_id", "edition_id"],
                columns="source",
                values="score",
                aggfunc="max",
            )
            .fillna(0.0)
            .reset_index()
        )
        # Rename columns to avoid spaces/special chars in CatBoost feature names.
        pivoted.columns = [
            f"src_{c}" if c not in ("user_id", "edition_id") else c
            for c in pivoted.columns
        ]
        src_cols = [c for c in pivoted.columns if c.startswith("src_")]
        pivoted["n_sources"] = (pivoted[src_cols] > 0).sum(axis=1).astype(float)
        return pivoted

    def _user_activity_features(self, positives: pd.DataFrame) -> pd.DataFrame:
        """Compute per-user activity statistics from observed interactions."""
        max_ts = positives["event_ts"].max()
        w30 = positives[positives["event_ts"] >= max_ts - pd.Timedelta(days=30)]

        total = (
            positives.groupby("user_id", as_index=False)["edition_id"]
            .nunique()
            .rename(columns={"edition_id": "user_n_positives"})
        )
        recent = (
            w30.groupby("user_id", as_index=False)["edition_id"]
            .nunique()
            .rename(columns={"edition_id": "user_n_positives_w30"})
        )
        last_ts = (
            positives.groupby("user_id", as_index=False)["event_ts"]
            .max()
            .rename(columns={"event_ts": "last_ts"})
        )
        last_ts["user_days_since_last"] = (max_ts - last_ts["last_ts"]).dt.days.astype(float)

        user_feats = (
            total.merge(recent, on="user_id", how="left")
            .merge(last_ts[["user_id", "user_days_since_last"]], on="user_id", how="left")
        )
        user_feats["user_n_positives_w30"] = user_feats["user_n_positives_w30"].fillna(0.0)
        user_feats["user_days_since_last"] = user_feats["user_days_since_last"].fillna(999.0)
        return user_feats

    def _item_popularity_features(self, positives: pd.DataFrame) -> pd.DataFrame:
        """Compute per-item popularity statistics across multiple time windows."""
        max_ts = positives["event_ts"].max()

        def pop(df: pd.DataFrame, col: str) -> pd.DataFrame:
            return (
                df.groupby("edition_id", as_index=False)["user_id"]
                .nunique()
                .rename(columns={"user_id": col})
            )

        all_pop = pop(positives, "item_pop_all")
        w30_pop = pop(
            positives[positives["event_ts"] >= max_ts - pd.Timedelta(days=30)],
            "item_pop_w30",
        )
        incident_pop = pop(
            positives[
                (positives["event_ts"] >= _INCIDENT_START_TS)
                & (positives["event_ts"] < _INCIDENT_START_TS + pd.Timedelta(days=31))
            ],
            "item_pop_incident",
        )
        reads = pop(positives[positives["event_type"] == 2], "item_n_reads")
        wishlists = pop(positives[positives["event_type"] == 1], "item_n_wishlists")

        item_feats = (
            all_pop.merge(w30_pop, on="edition_id", how="left")
            .merge(incident_pop, on="edition_id", how="left")
            .merge(reads, on="edition_id", how="left")
            .merge(wishlists, on="edition_id", how="left")
        )
        for col in ["item_pop_w30", "item_pop_incident", "item_n_reads", "item_n_wishlists"]:
            item_feats[col] = item_feats[col].fillna(0.0)
        return item_feats

    def _item_catalogue_features(self, dataset: Dataset) -> pd.DataFrame:
        """Extract item-level attributes from the edition catalogue."""
        cat = dataset.catalog_df[
            ["edition_id", "publication_year", "age_restriction", "language_id"]
        ].copy()
        cat["publication_year"] = cat["publication_year"].fillna(
            cat["publication_year"].median()
        )
        cat["age_restriction"] = cat["age_restriction"].fillna(0).astype(int)
        cat["language_id"] = cat["language_id"].fillna(-1).astype(int)
        return cat

    def _cross_features(
        self, pairs: pd.DataFrame, positives: pd.DataFrame, dataset: Dataset
    ) -> pd.DataFrame:
        """Compute user×item affinity signals from raw interaction history.

        Computes three cross features for each (user_id, edition_id) pair:
        - ``user_author_affinity``: share of user's interactions that share
          the item's author (0–1, higher = more loyal to this author).
        - ``user_language_affinity``: share of user's interactions in the
          item's language.
        - ``user_genre_affinity``: max fraction of user's genre interactions
          that belong to any one of the item's genres.
        """
        # user total positives (for normalisation)
        user_totals = (
            positives.groupby("user_id")["edition_id"].count().rename("user_total")
        )

        # --- author affinity ---
        item_author = dataset.catalog_df[["edition_id", "author_id"]].copy()
        author_counts = (
            positives.merge(item_author, on="edition_id", how="left")
            .groupby(["user_id", "author_id"])["edition_id"]
            .count()
            .rename("author_n")
            .reset_index()
        )
        cf_author = (
            pairs[["user_id", "edition_id"]]
            .merge(item_author, on="edition_id", how="left")
            .merge(author_counts, on=["user_id", "author_id"], how="left")
            .merge(user_totals, on="user_id", how="left")
        )
        cf_author["user_author_affinity"] = (
            cf_author["author_n"].fillna(0) / cf_author["user_total"].clip(lower=1)
        )
        cf_author = cf_author[["user_id", "edition_id", "user_author_affinity"]]

        # --- language affinity ---
        item_lang = dataset.catalog_df[["edition_id", "language_id"]].copy()
        lang_counts = (
            positives.merge(item_lang, on="edition_id", how="left")
            .groupby(["user_id", "language_id"])["edition_id"]
            .count()
            .rename("lang_n")
            .reset_index()
        )
        cf_lang = (
            pairs[["user_id", "edition_id"]]
            .merge(item_lang, on="edition_id", how="left")
            .merge(lang_counts, on=["user_id", "language_id"], how="left")
            .merge(user_totals, on="user_id", how="left")
        )
        cf_lang["user_language_affinity"] = (
            cf_lang["lang_n"].fillna(0) / cf_lang["user_total"].clip(lower=1)
        )
        cf_lang = cf_lang[["user_id", "edition_id", "user_language_affinity"]]

        # --- genre affinity (max over item's genres) ---
        item_genres = (
            dataset.catalog_df[["edition_id", "book_id"]]
            .merge(dataset.book_genres_df[["book_id", "genre_id"]], on="book_id", how="inner")
            .drop(columns=["book_id"])
        )
        genre_counts = (
            positives.merge(item_genres, on="edition_id", how="left")
            .groupby(["user_id", "genre_id"])["edition_id"]
            .count()
            .rename("genre_n")
            .reset_index()
        )
        cf_genre = (
            pairs[["user_id", "edition_id"]]
            .merge(item_genres, on="edition_id", how="left")
            .merge(genre_counts, on=["user_id", "genre_id"], how="left")
            .merge(user_totals, on="user_id", how="left")
        )
        cf_genre["genre_affinity"] = (
            cf_genre["genre_n"].fillna(0) / cf_genre["user_total"].clip(lower=1)
        )
        cf_genre = (
            cf_genre.groupby(["user_id", "edition_id"], as_index=False)["genre_affinity"]
            .max()
            .rename(columns={"genre_affinity": "user_genre_affinity"})
        )

        result = (
            pairs[["user_id", "edition_id"]]
            .merge(cf_author, on=["user_id", "edition_id"], how="left")
            .merge(cf_lang, on=["user_id", "edition_id"], how="left")
            .merge(cf_genre, on=["user_id", "edition_id"], how="left")
        )
        for col in ["user_author_affinity", "user_language_affinity", "user_genre_affinity"]:
            result[col] = result[col].fillna(0.0)
        return result

    def _build_feature_matrix(
        self, candidates: pd.DataFrame, positives: pd.DataFrame, dataset: Dataset
    ) -> pd.DataFrame:
        """Assemble the full feature matrix for the candidate (user, item) pairs."""
        src_feats = self._source_score_features(candidates)
        pairs = src_feats[["user_id", "edition_id"]].copy()

        user_feats = self._user_activity_features(positives)
        item_pop_feats = self._item_popularity_features(positives)
        item_cat_feats = self._item_catalogue_features(dataset)
        cross_feats = self._cross_features(pairs, positives, dataset)

        feat = (
            src_feats.merge(user_feats, on="user_id", how="left")
            .merge(item_pop_feats, on="edition_id", how="left")
            .merge(item_cat_feats, on="edition_id", how="left")
            .merge(cross_feats, on=["user_id", "edition_id"], how="left")
        )
        for col in ["user_n_positives", "user_n_positives_w30", "user_days_since_last"]:
            feat[col] = feat[col].fillna(0.0)
        for col in ["item_pop_all", "item_pop_w30", "item_pop_incident"]:
            feat[col] = feat[col].fillna(0.0)
        feat["publication_year"] = feat["publication_year"].fillna(
            feat["publication_year"].median()
        )
        feat["age_restriction"] = feat["age_restriction"].fillna(0).astype(int)
        feat["language_id"] = feat["language_id"].fillna(-1).astype(int)
        return feat

    def _build_training_labels(
        self, feat: pd.DataFrame, positives: pd.DataFrame
    ) -> pd.Series:
        """Label each candidate row using a pseudo-incident holdout window.

        Items observed in the ``pseudo_incident_days`` window immediately before
        the real incident start are treated as pseudo-lost ground-truth. All
        other candidates receive label 0.
        """
        pseudo_end = _INCIDENT_START_TS
        pseudo_start = pseudo_end - pd.Timedelta(days=self.pseudo_incident_days)
        pseudo_positives = (
            positives[
                (positives["event_ts"] >= pseudo_start)
                & (positives["event_ts"] < pseudo_end)
            ][["user_id", "edition_id"]]
            .drop_duplicates()
            .assign(label=1)
        )
        labels = feat[["user_id", "edition_id"]].merge(
            pseudo_positives, on=["user_id", "edition_id"], how="left"
        )["label"].fillna(0).astype(int)
        return labels

    # ------------------------------------------------------------------
    # training & inference
    # ------------------------------------------------------------------

    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cat_features: list[str],
    ) -> object:
        """Train a CatBoost binary classifier with automatic class balancing."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return None

        model = CatBoostClassifier(
            iterations=self.catboost_iterations,
            learning_rate=self.catboost_lr,
            depth=self.catboost_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self.seed,
            auto_class_weights="Balanced",
            early_stopping_rounds=30,
            use_best_model=True,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train, cat_features=cat_features)
        return model

    def rank(self, dataset: Dataset, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """Re-rank candidates with CatBoost; fall back to blend on failure.

        Args:
            dataset: Runtime dataset with raw interactions and catalogue.
            candidates: Candidate frame merged across all generator sources.
            k: Required top-k output per user.

        Returns:
            Ranked DataFrame with ``user_id``, ``edition_id``, ``rank``,
            ``final_score`` columns.
        """
        if candidates.empty:
            return self._fallback_only(dataset, k)

        positives = dataset.interactions_df[
            dataset.interactions_df["event_type"].isin([1, 2])
        ]

        try:
            feat = self._build_feature_matrix(candidates, positives, dataset)
            labels = self._build_training_labels(feat, positives)

            id_cols = ["user_id", "edition_id"]
            cat_features = ["age_restriction", "language_id"]
            feature_cols = [c for c in feat.columns if c not in id_cols]

            X = feat[feature_cols].copy()
            for col in cat_features:
                if col in X.columns:
                    X[col] = X[col].astype(str)

            n_positives = int((labels == 1).sum())
            model = None
            if n_positives >= 10:
                model = self._train_catboost(X, labels, cat_features)
                if model is not None:
                    logger.info(
                        "CatBoostRanker trained: %d positives / %d candidates",
                        n_positives,
                        len(X),
                    )

            if model is not None:
                final_scores = model.predict_proba(X)[:, 1]
            else:
                # Weighted blend fallback when not enough training signal.
                src_cols = [c for c in feature_cols if c.startswith("src_")]
                final_scores = X[src_cols].sum(axis=1).to_numpy()

            result = feat[["user_id", "edition_id"]].copy()
            result["final_score"] = final_scores

        except Exception as exc:
            logger.warning("CatBoostRanker failed (%s); falling back to blend.", exc)
            agg = (
                candidates.groupby(["user_id", "edition_id"], as_index=False)["score"]
                .max()
                .rename(columns={"score": "final_score"})
            )
            result = agg

        # Filter seen, top-k, fallback fill.
        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        result = result.merge(seen.assign(_seen=1), on=["user_id", "edition_id"], how="left")
        result = result[result["_seen"].isna()].drop(columns=["_seen"])

        result = result.sort_values(["user_id", "final_score"], ascending=[True, False])
        selected = result.groupby("user_id", group_keys=False).head(k).copy()
        selected["rank"] = selected.groupby("user_id").cumcount() + 1
        selected = selected[["user_id", "edition_id", "rank", "final_score"]]

        completed = self._apply_fallback(selected, dataset, k)
        return completed.sort_values(["user_id", "rank"]).reset_index(drop=True)

    def _fallback_only(self, dataset: Dataset, k: int) -> pd.DataFrame:
        return SimpleBlendRanker()._fallback_only(dataset, k)

    def _apply_fallback(
        self, selected: pd.DataFrame, dataset: Dataset, k: int
    ) -> pd.DataFrame:
        return SimpleBlendRanker()._apply_fallback(selected, dataset, k)


def rank_predictions(
    dataset: Dataset,
    candidates: pd.DataFrame,
    source_weights: dict[str, float],
    k: int,
    use_catboost: bool = True,
) -> pd.DataFrame:
    """Rank candidate set using the configured strategy.

    When ``use_catboost=True`` (default) a :class:`CatBoostRanker` is used;
    it trains a binary classifier on pseudo-incident data derived from the
    clean interaction history and scores each candidate accordingly.
    If CatBoost is unavailable or training fails the function automatically
    falls back to :class:`SimpleBlendRanker`.

    Args:
        dataset: Runtime dataset for filtering and fallback behaviour.
        candidates: Candidate rows emitted by all generators.
        source_weights: Source weights for the blend fallback.
        k: Required cutoff for returned top list.
        use_catboost: Whether to attempt the CatBoost re-ranker.

    Returns:
        Ranked DataFrame with ``user_id``, ``edition_id``, ``rank``,
        ``final_score``.
    """
    if use_catboost:
        import catboost  # noqa: F401 – probe availability

        ranker: object = CatBoostRanker()
        return ranker.rank(dataset=dataset, candidates=candidates, k=int(k))


    ranker = SimpleBlendRanker(
        source_weights={key: float(value) for key, value in source_weights.items()}
    )
    return ranker.rank(dataset=dataset, candidates=candidates, k=int(k))

