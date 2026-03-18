"""Ranking logic for participant solution outputs.

Implements a three-layer ranker stack following the improvement plan:
  Layer A: rich feature builder (source scores + user state + item popularity
           + affinity + rating + cross features), computed strictly from the
           observation window to prevent temporal leakage.
  Layer B: CatBoostRanker with YetiRank objective (NDCG-aligned) trained on
           multiple rolling pseudo-incident windows.
  Layer C: ensemble of YetiRank + pointwise classifier + RRF blend with
           popularity fallback.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset

logger = logging.getLogger(__name__)

_INCIDENT_START_TS = pd.Timestamp("2025-10-01 00:00:00")
_INCIDENT_END_TS = pd.Timestamp("2025-11-01 00:00:00")
_POST_INCIDENT_END_TS = pd.Timestamp("2025-12-01 00:00:00")
_PSEUDO_INCIDENT_DAYS = 14
_N_ROLLING_WINDOWS = 6
_INJECT_PSEUDO_POSITIVES = True


# ─────────────────────────────────────────────────────────────────────────────
# Layer A: feature builders
# ─────────────────────────────────────────────────────────────────────────────


def _source_score_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Pivot source scores to wide format; add n_sources and per-source RRF rank."""
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
    pivoted.columns = [
        f"src_{c}" if c not in ("user_id", "edition_id") else c
        for c in pivoted.columns
    ]
    src_cols = [c for c in pivoted.columns if c.startswith("src_")]
    pivoted["n_sources"] = (pivoted[src_cols] > 0).sum(axis=1).astype(float)

    # Per-source within-user rank and aggregated RRF score.
    # RRF(rank) = 1/(60+rank); rank 1 = highest score.
    rrf_total = np.zeros(len(pivoted))
    for src_col in src_cols:
        ranks = pivoted.groupby("user_id")[src_col].rank(method="first", ascending=False)
        pivoted[src_col.replace("src_", "rank_")] = ranks
        rrf_total += 1.0 / (60.0 + ranks.values)
    pivoted["rrf_score"] = rrf_total
    return pivoted


def _user_activity_features(pre_incident: pd.DataFrame, incident_start: pd.Timestamp) -> pd.DataFrame:
    """Per-user activity counts over multiple windows with log1p transforms."""
    if pre_incident.empty:
        return pd.DataFrame(columns=["user_id"])

    def unique_editions(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return (
            df.groupby("user_id", as_index=False)["edition_id"]
            .nunique()
            .rename(columns={"edition_id": col})
        )

    total = unique_editions(pre_incident, "user_n_positives")
    w7 = unique_editions(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=7)],
        "user_n_positives_w7",
    )
    w14 = unique_editions(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=14)],
        "user_n_positives_w14",
    )
    w30 = unique_editions(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=30)],
        "user_n_positives_w30",
    )
    w90 = unique_editions(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=90)],
        "user_n_positives_w90",
    )

    last_ts = (
        pre_incident.groupby("user_id", as_index=False)["event_ts"]
        .max()
        .rename(columns={"event_ts": "last_ts"})
    )
    last_ts["user_days_since_last"] = (incident_start - last_ts["last_ts"]).dt.days.astype(float)

    reads_per_user = (
        pre_incident[pre_incident["event_type"] == 2]
        .groupby("user_id", as_index=False)["edition_id"]
        .count()
        .rename(columns={"edition_id": "user_n_reads"})
    )
    wishlists_per_user = (
        pre_incident[pre_incident["event_type"] == 1]
        .groupby("user_id", as_index=False)["edition_id"]
        .count()
        .rename(columns={"edition_id": "user_n_wishlists"})
    )

    user_feats = (
        total
        .merge(w7, on="user_id", how="left")
        .merge(w14, on="user_id", how="left")
        .merge(w30, on="user_id", how="left")
        .merge(w90, on="user_id", how="left")
        .merge(last_ts[["user_id", "user_days_since_last"]], on="user_id", how="left")
        .merge(reads_per_user, on="user_id", how="left")
        .merge(wishlists_per_user, on="user_id", how="left")
    )

    fill_cols = [
        "user_n_positives_w7", "user_n_positives_w14",
        "user_n_positives_w30", "user_n_positives_w90",
        "user_n_reads", "user_n_wishlists",
    ]
    for col in fill_cols:
        user_feats[col] = user_feats[col].fillna(0.0)
    user_feats["user_days_since_last"] = user_feats["user_days_since_last"].fillna(999.0)

    user_feats["user_read_wishlist_ratio"] = (
        user_feats["user_n_reads"] / (user_feats["user_n_wishlists"] + 1.0)
    )
    user_feats["user_recent_to_long_ratio"] = (
        user_feats["user_n_positives_w30"] / (user_feats["user_n_positives"] + 1.0)
    )

    for col in [
        "user_n_positives", "user_n_positives_w7", "user_n_positives_w14",
        "user_n_positives_w30", "user_n_positives_w90",
        "user_n_reads", "user_n_wishlists",
    ]:
        user_feats[f"log1p_{col}"] = np.log1p(user_feats[col])

    return user_feats


def _item_popularity_features(observed_positives: pd.DataFrame, incident_start: pd.Timestamp, incident_end: pd.Timestamp) -> pd.DataFrame:
    """Per-item popularity over multiple windows with trend ratios and log1p."""
    pre_incident = observed_positives[observed_positives["event_ts"] < incident_start]
    if pre_incident.empty:
        return pd.DataFrame(columns=["edition_id"])

    def pop(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return (
            df.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": col})
        )

    all_pop = pop(pre_incident, "item_pop_all")
    w7_pop = pop(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=7)],
        "item_pop_w7",
    )
    w14_pop = pop(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=14)],
        "item_pop_w14",
    )
    w30_pop = pop(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=30)],
        "item_pop_w30",
    )
    w90_pop = pop(
        pre_incident[pre_incident["event_ts"] >= incident_start - pd.Timedelta(days=90)],
        "item_pop_w90",
    )
    incident_pop = pop(
        observed_positives[
            (observed_positives["event_ts"] >= incident_start)
            & (observed_positives["event_ts"] < incident_end)
        ],
        "item_pop_incident",
    )
    reads = pop(pre_incident[pre_incident["event_type"] == 2], "item_n_reads")
    wishlists = pop(pre_incident[pre_incident["event_type"] == 1], "item_n_wishlists")

    item_feats = (
        all_pop
        .merge(w7_pop, on="edition_id", how="left")
        .merge(w14_pop, on="edition_id", how="left")
        .merge(w30_pop, on="edition_id", how="left")
        .merge(w90_pop, on="edition_id", how="left")
        .merge(incident_pop, on="edition_id", how="left")
        .merge(reads, on="edition_id", how="left")
        .merge(wishlists, on="edition_id", how="left")
    )

    for col in [
        "item_pop_w7", "item_pop_w14", "item_pop_w30", "item_pop_w90",
        "item_pop_incident", "item_n_reads", "item_n_wishlists",
    ]:
        item_feats[col] = item_feats[col].fillna(0.0)

    item_feats["item_trend_short_long"] = (
        item_feats["item_pop_w14"] / (item_feats["item_pop_w90"] + 1.0)
    )
    item_feats["item_trend_very_short"] = (
        item_feats["item_pop_w7"] / (item_feats["item_pop_w30"] + 1.0)
    )
    item_feats["item_read_wishlist_ratio"] = (
        item_feats["item_n_reads"] / (item_feats["item_n_wishlists"] + 1.0)
    )

    for col in [
        "item_pop_all", "item_pop_w7", "item_pop_w14", "item_pop_w30",
        "item_pop_w90", "item_pop_incident", "item_n_reads", "item_n_wishlists",
    ]:
        item_feats[f"log1p_{col}"] = np.log1p(item_feats[col])

    return item_feats


def _item_catalogue_features(dataset: Dataset) -> pd.DataFrame:
    """Item-level catalogue attributes (year, age restriction, language)."""
    cat = dataset.catalog_df[
        ["edition_id", "publication_year", "age_restriction", "language_id"]
    ].copy()
    cat["publication_year"] = cat["publication_year"].fillna(
        cat["publication_year"].median()
    )
    cat["age_restriction"] = cat["age_restriction"].fillna(0).astype(int)
    cat["language_id"] = cat["language_id"].fillna(-1).astype(int)
    return cat


def _user_demographic_features(dataset: Dataset) -> pd.DataFrame:
    """User-level demographic features from users.csv."""
    users = dataset.users_df[["user_id", "gender", "age"]].copy()
    users["age_missing"] = users["age"].isna().astype(int)
    users["gender"] = users["gender"].fillna(0).astype(int)
    users["age"] = users["age"].fillna(users["age"].median())
    users["age"] = users["age"].fillna(0.0).astype(float)
    users["age_bucket"] = pd.cut(
        users["age"],
        bins=[-1, 18, 25, 35, 50, 200],
        labels=False,
        include_lowest=True,
    ).fillna(5).astype(int)
    return users


def _rating_features(
    positives: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean rating and count per item; mean rating per user from read events."""
    reads_with_rating = positives[
        (positives["event_type"] == 2) & positives["rating"].notna()
    ]
    if reads_with_rating.empty:
        return (
            pd.DataFrame(columns=["edition_id", "item_mean_rating", "item_rating_count"]),
            pd.DataFrame(columns=["user_id", "user_mean_rating"]),
        )

    item_ratings = (
        reads_with_rating.groupby("edition_id", as_index=False)
        .agg(item_mean_rating=("rating", "mean"), item_rating_count=("rating", "count"))
    )
    user_ratings = (
        reads_with_rating.groupby("user_id", as_index=False)
        .agg(user_mean_rating=("rating", "mean"))
    )
    return item_ratings, user_ratings


def _cross_features(
    pairs: pd.DataFrame,
    observed_positives: pd.DataFrame,
    dataset: Dataset,
    incident_start: pd.Timestamp,
    incident_end: pd.Timestamp,
    post_end: pd.Timestamp,
) -> pd.DataFrame:
    """User×item affinity for author/language/genre/publisher (all + post window)."""
    result = pairs[["user_id", "edition_id"]].copy()

    def _build_affinity_block(
        history: pd.DataFrame, suffix: str = ""
    ) -> pd.DataFrame:
        if history.empty:
            empty = result.copy()
            for col in (
                f"user_author_affinity{suffix}",
                f"user_language_affinity{suffix}",
                f"user_genre_affinity{suffix}",
                f"user_publisher_affinity{suffix}",
            ):
                empty[col] = 0.0
            return empty

        user_totals = history.groupby("user_id")["edition_id"].count().rename("user_total")

        def _affinity(
            item_map: pd.DataFrame,
            key_col: str,
            affinity_col: str,
            count_col: str,
            agg_max: bool = False,
        ) -> pd.DataFrame:
            counts = (
                history.merge(item_map, on="edition_id", how="left")
                .groupby(["user_id", key_col])["edition_id"]
                .count()
                .rename(count_col)
                .reset_index()
            )
            cf = (
                result.merge(item_map, on="edition_id", how="left")
                .merge(counts, on=["user_id", key_col], how="left")
                .merge(user_totals, on="user_id", how="left")
            )
            cf[affinity_col] = cf[count_col].fillna(0) / cf["user_total"].clip(lower=1)
            if agg_max:
                return cf.groupby(["user_id", "edition_id"], as_index=False)[affinity_col].max()
            return cf[["user_id", "edition_id", affinity_col]]

        cf_author = _affinity(
            dataset.catalog_df[["edition_id", "author_id"]],
            "author_id",
            f"user_author_affinity{suffix}",
            "author_n",
        )
        cf_lang = _affinity(
            dataset.catalog_df[["edition_id", "language_id"]],
            "language_id",
            f"user_language_affinity{suffix}",
            "lang_n",
        )
        cf_publisher = _affinity(
            dataset.catalog_df[["edition_id", "publisher_id"]],
            "publisher_id",
            f"user_publisher_affinity{suffix}",
            "publisher_n",
        )
        item_genres = (
            dataset.catalog_df[["edition_id", "book_id"]]
            .merge(dataset.book_genres_df[["book_id", "genre_id"]], on="book_id", how="inner")
            .drop(columns=["book_id"])
        )
        cf_genre = _affinity(
            item_genres,
            "genre_id",
            f"user_genre_affinity{suffix}",
            "genre_n",
            agg_max=True,
        )

        block = (
            result.merge(cf_author, on=["user_id", "edition_id"], how="left")
            .merge(cf_lang, on=["user_id", "edition_id"], how="left")
            .merge(cf_genre, on=["user_id", "edition_id"], how="left")
            .merge(cf_publisher, on=["user_id", "edition_id"], how="left")
        )
        for col in (
            f"user_author_affinity{suffix}",
            f"user_language_affinity{suffix}",
            f"user_genre_affinity{suffix}",
            f"user_publisher_affinity{suffix}",
        ):
            block[col] = block[col].fillna(0.0)
        return block

    pre_incident = observed_positives[observed_positives["event_ts"] < incident_start]
    all_block = _build_affinity_block(pre_incident, suffix="")
    
    post_incident = observed_positives[
        (observed_positives["event_ts"] >= incident_end)
        & (observed_positives["event_ts"] < post_end)
    ]
    post_block = _build_affinity_block(post_incident, suffix="_post")
    
    return all_block.merge(
        post_block,
        on=["user_id", "edition_id"],
        how="left",
    )


def _build_feature_matrix(
    candidates: pd.DataFrame,
    observed_positives: pd.DataFrame,
    dataset: Dataset,
    incident_start: pd.Timestamp,
    incident_end: pd.Timestamp,
    post_end: pd.Timestamp,
) -> pd.DataFrame:
    """Assemble the full feature matrix for (user, item) candidate pairs.

    ``observed_positives`` must contain interactions properly partitioned
    by the provided boundaries to prevent temporal leakage.
    """
    src_feats = _source_score_features(candidates)
    pairs = src_feats[["user_id", "edition_id"]].copy()

    pre_incident = observed_positives[observed_positives["event_ts"] < incident_start]

    user_feats = _user_activity_features(pre_incident, incident_start)
    user_demo_feats = _user_demographic_features(dataset)
    item_pop_feats = _item_popularity_features(observed_positives, incident_start, incident_end)
    item_cat_feats = _item_catalogue_features(dataset)
    item_rating_feats, user_rating_feats = _rating_features(pre_incident)
    cross_feats = _cross_features(pairs, observed_positives, dataset, incident_start, incident_end, post_end)

    feat = (
        src_feats
        .merge(user_feats, on="user_id", how="left")
        .merge(user_demo_feats, on="user_id", how="left")
        .merge(item_pop_feats, on="edition_id", how="left")
        .merge(item_cat_feats, on="edition_id", how="left")
        .merge(item_rating_feats, on="edition_id", how="left")
        .merge(user_rating_feats, on="user_id", how="left")
        .merge(cross_feats, on=["user_id", "edition_id"], how="left")
    )

    skip_fill = {"user_id", "edition_id", "age_restriction", "language_id"}
    for col in feat.columns:
        if col in skip_fill or feat[col].dtype == object:
            continue
        feat[col] = feat[col].fillna(0.0)

    feat["publication_year"] = feat["publication_year"].fillna(
        feat["publication_year"].median()
    )
    feat["age_restriction"] = feat["age_restriction"].fillna(0).astype(int)
    feat["language_id"] = feat["language_id"].fillna(-1).astype(int)

    # Pair-level rating relative to user average (item preference signal)
    if "item_mean_rating" in feat.columns and "user_mean_rating" in feat.columns:
        feat["rating_relative_to_user"] = feat["item_mean_rating"] - feat["user_mean_rating"]
        feat["rating_relative_to_user"] = feat["rating_relative_to_user"].fillna(0.0)

    # Multiplicative cross features
    if "item_pop_incident" in feat.columns:
        if "user_author_affinity_post" in feat.columns:
            feat["item_pop_incident_x_user_author_affinity_post"] = (
                feat["item_pop_incident"] * feat["user_author_affinity_post"]
            )
        if "user_genre_affinity_post" in feat.columns:
            feat["item_pop_incident_x_user_genre_affinity_post"] = (
                feat["item_pop_incident"] * feat["user_genre_affinity_post"]
            )
        if "user_language_affinity_post" in feat.columns:
            feat["item_pop_incident_x_user_language_affinity_post"] = (
                feat["item_pop_incident"] * feat["user_language_affinity_post"]
            )
        if "user_publisher_affinity_post" in feat.columns:
            feat["item_pop_incident_x_user_publisher_affinity_post"] = (
                feat["item_pop_incident"] * feat["user_publisher_affinity_post"]
            )

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Rolling pseudo-label builder
# ─────────────────────────────────────────────────────────────────────────────


def _inject_pseudo_positives(
    fold_candidates: pd.DataFrame,
    pseudo_pos: pd.DataFrame,
) -> pd.DataFrame:
    """Add pseudo-positive pairs that are missing from the candidate set.

    Candidate generators filter out items already in ``seen_positive_df``.
    During training, the pseudo-positive items ARE in ``seen_positive_df``
    (they are from the clean history), so they would be absent from
    ``fold_candidates``.  We inject them back with zero source scores so
    the feature builder can compute affinity/item features for them.

    Uses the first existing source name so no new source column is created,
    ensuring the inference feature schema is unchanged.
    """
    if fold_candidates.empty:
        return fold_candidates

    existing = fold_candidates[["user_id", "edition_id"]].drop_duplicates()
    new_pos = pseudo_pos[["user_id", "edition_id"]].merge(
        existing.assign(_exists=1), on=["user_id", "edition_id"], how="left"
    )
    new_pos = new_pos[new_pos["_exists"].isna()].drop(columns=["_exists"])
    if new_pos.empty:
        return fold_candidates

    # Use per-user median source score for injected items so they don't form
    # a trivially-separable group (score==0 is a pure injection artefact absent
    # at inference time; using the median prevents the first tree from splitting
    # on score==0 and stopping early).
    user_median_score = (
        fold_candidates.groupby("user_id")["score"].median().rename("med_score")
    )
    new_pos = new_pos.merge(user_median_score, on="user_id", how="left")
    global_median = float(fold_candidates["score"].median())
    new_pos["score"] = new_pos["med_score"].fillna(global_median)
    new_pos = new_pos.drop(columns=["med_score"])

    first_source = fold_candidates["source"].iloc[0]
    new_pos["source"] = first_source
    return pd.concat([fold_candidates, new_pos], ignore_index=True)


def _activity_bucket_from_counts(counts: pd.Series) -> pd.Series:
    """Map per-user unique-positive counts to coarse activity buckets."""
    return pd.cut(
        counts.fillna(0.0),
        bins=[0, 5, 20, 50, float("inf")],
        labels=["cold", "light", "medium", "heavy"],
        include_lowest=True,
    ).astype(str)


def _normalize_multiplier(mapping: dict[object, float]) -> dict[object, float]:
    """Rescale multiplier map so the average value is ~1.0."""
    if not mapping:
        return mapping
    vals = np.array(list(mapping.values()), dtype=float)
    mean_val = float(vals.mean()) if len(vals) else 1.0
    if mean_val <= 0:
        return {k: 1.0 for k in mapping}
    return {k: float(v) / mean_val for k, v in mapping.items()}


def _estimate_incident_loss_bias(
    all_positives: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, float], dict[str, float]]:
    """Estimate non-random loss pattern from observed incident statistics.

    Returns multiplicative biases for:
      - event_type (wishlist/read)
      - hour of day
      - user activity bucket
    Bias > 1 means segment appears under-represented in incident vs pre+post.
    """
    pre_df = all_positives[
        (all_positives["event_ts"] >= _INCIDENT_START_TS - pd.Timedelta(days=31))
        & (all_positives["event_ts"] < _INCIDENT_START_TS)
    ]
    incident_df = all_positives[
        (all_positives["event_ts"] >= _INCIDENT_START_TS)
        & (all_positives["event_ts"] < _INCIDENT_END_TS)
    ]
    post_df = all_positives[
        (all_positives["event_ts"] >= _INCIDENT_END_TS)
        & (all_positives["event_ts"] < _POST_INCIDENT_END_TS)
    ]

    if pre_df.empty or incident_df.empty or post_df.empty:
        return ({1: 1.0, 2: 1.0}, {h: 1.0 for h in range(24)}, {"cold": 1.0, "light": 1.0, "medium": 1.0, "heavy": 1.0})

    pre_days = max((pre_df["event_ts"].max() - pre_df["event_ts"].min()).days + 1, 1)
    incident_days = max((incident_df["event_ts"].max() - incident_df["event_ts"].min()).days + 1, 1)
    post_days = max((post_df["event_ts"].max() - post_df["event_ts"].min()).days + 1, 1)

    # 1) Event-type asymmetry
    type_mult: dict[int, float] = {}
    for et in (1, 2):
        pre_rate = float((pre_df["event_type"] == et).sum()) / pre_days
        post_rate = float((post_df["event_type"] == et).sum()) / post_days
        inc_rate = float((incident_df["event_type"] == et).sum()) / incident_days
        expected_rate = max((pre_rate + post_rate) / 2.0, 1e-9)
        loss = max(0.0, 1.0 - (inc_rate / expected_rate))
        type_mult[et] = 1.0 + loss
    type_mult = _normalize_multiplier(type_mult)

    # 2) Hour-of-day temporal clusters
    pre_h = pre_df["event_ts"].dt.hour.value_counts().to_dict()
    inc_h = incident_df["event_ts"].dt.hour.value_counts().to_dict()
    post_h = post_df["event_ts"].dt.hour.value_counts().to_dict()
    hour_mult: dict[int, float] = {}
    for h in range(24):
        pre_rate = float(pre_h.get(h, 0.0)) / pre_days
        post_rate = float(post_h.get(h, 0.0)) / post_days
        inc_rate = float(inc_h.get(h, 0.0)) / incident_days
        expected_rate = max((pre_rate + post_rate) / 2.0, 1e-9)
        loss = max(0.0, 1.0 - (inc_rate / expected_rate))
        hour_mult[h] = 1.0 + loss
    hour_mult = _normalize_multiplier(hour_mult)

    # 3) User-activity segment bias
    user_hist_counts = (
        all_positives[all_positives["event_ts"] < _INCIDENT_START_TS]
        .groupby("user_id")["edition_id"]
        .nunique()
    )
    user_bucket = _activity_bucket_from_counts(user_hist_counts)

    def _bucket_rate(df: pd.DataFrame, days: int) -> dict[str, float]:
        if df.empty:
            return {}
        tmp = df[["user_id"]].copy()
        tmp["bucket"] = tmp["user_id"].map(user_bucket).fillna("cold")
        counts = tmp["bucket"].value_counts().to_dict()
        return {k: float(v) / max(days, 1) for k, v in counts.items()}

    pre_b = _bucket_rate(pre_df, pre_days)
    inc_b = _bucket_rate(incident_df, incident_days)
    post_b = _bucket_rate(post_df, post_days)
    bucket_mult: dict[str, float] = {}
    for b in ("cold", "light", "medium", "heavy"):
        pre_rate = pre_b.get(b, 0.0)
        post_rate = post_b.get(b, 0.0)
        inc_rate = inc_b.get(b, 0.0)
        expected_rate = max((pre_rate + post_rate) / 2.0, 1e-9)
        loss = max(0.0, 1.0 - (inc_rate / expected_rate))
        bucket_mult[b] = 1.0 + loss
    bucket_mult = _normalize_multiplier(bucket_mult)
    return type_mult, hour_mult, bucket_mult


def _sample_structured_pseudo_pairs(
    window_events: pd.DataFrame,
    user_hist_counts: pd.Series,
    type_mult: dict[int, float],
    hour_mult: dict[int, float],
    bucket_mult: dict[str, float],
    rng: np.random.Generator,
    mask_fraction: float = 0.20,
) -> pd.DataFrame:
    """Sample pseudo-positive pairs with non-random incident-like bias."""
    if window_events.empty:
        return pd.DataFrame(columns=["user_id", "edition_id"])

    pair_events = (
        window_events.sort_values("event_ts")
        .drop_duplicates(subset=["user_id", "edition_id"], keep="last")
        .copy()
    )
    if pair_events.empty:
        return pd.DataFrame(columns=["user_id", "edition_id"])

    user_bucket = _activity_bucket_from_counts(user_hist_counts)
    pair_events["bucket"] = pair_events["user_id"].map(user_bucket).fillna("cold")
    pair_events["hour"] = pair_events["event_ts"].dt.hour.astype(int)

    w_type = pair_events["event_type"].map(type_mult).fillna(1.0).astype(float)
    w_hour = pair_events["hour"].map(hour_mult).fillna(1.0).astype(float)
    w_bucket = pair_events["bucket"].map(bucket_mult).fillna(1.0).astype(float)
    weights = (w_type * w_hour * w_bucket).to_numpy(dtype=float)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()

    n_select = max(1, int(len(pair_events) * mask_fraction))
    n_select = min(n_select, len(pair_events))
    selected_idx = rng.choice(len(pair_events), size=n_select, replace=False, p=weights)
    sampled = pair_events.iloc[selected_idx][["user_id", "edition_id"]].drop_duplicates()
    return sampled.reset_index(drop=True)


def _trim_fold_candidates_for_training(
    fold_candidates: pd.DataFrame,
    per_source_k: int = 120,
) -> pd.DataFrame:
    """Trim fold candidates to strongest rows per (user, source).

    This reduces training-time compute significantly while preserving
    source diversity and hard negatives from each generator.
    """
    if fold_candidates.empty:
        return fold_candidates
    trimmed = (
        fold_candidates.sort_values(
            ["user_id", "source", "score", "edition_id"],
            ascending=[True, True, False, True],
        )
        .groupby(["user_id", "source"], group_keys=False)
        .head(per_source_k)
        .reset_index(drop=True)
    )
    return trimmed


def _generate_rolling_training_data(
    candidates: pd.DataFrame,
    all_positives: pd.DataFrame,
    dataset: Dataset,
    n_windows: int = _N_ROLLING_WINDOWS,
    window_days: int = _PSEUDO_INCIDENT_DAYS,
    min_obs_interactions: int = 3,
    max_rows_per_fold: int = 300_000,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build stacked training data from rolling pseudo-incident windows.

    For each window ``i`` (starting from 1):
    - ``window_end   = INCIDENT_START - (i-1)*window_days``
    - ``window_start = window_end - window_days``
    - observation positives: strictly before ``window_start`` (no leakage)
    - pseudo-positives: pairs observed in ``[window_start, window_end)``
    - label = 1 for pseudo-positives, 0 for all other candidates

    Pseudo-positive items are injected into ``fold_candidates`` with per-user median
    source scores when absent (generators filter them as "already seen").
    This gives the model signal from affinity/item features for positive pairs
    that generators would otherwise miss.

    Window i=1 (closest to real incident) is reserved as validation fold;
    windows i=2..n are used for training.

    To keep memory within bounds, each fold is subsampled to at most
    ``max_rows_per_fold`` rows (all positives kept; negatives randomly
    down-sampled if necessary).

    Returns:
        feat_all:   feature matrix (stacked over folds, with ``_fold`` column)
        y_all:      binary label series
        groups_all: ``user_id`` series for CatBoost group_id
    """
    fold_frames: list[pd.DataFrame] = []
    fold_labels: list[pd.Series] = []
    fold_groups: list[pd.Series] = []
    user_hist_counts_full = (
        all_positives.groupby("user_id")["edition_id"].nunique().astype(float)
    )
    type_mult, hour_mult, bucket_mult = _estimate_incident_loss_bias(all_positives)

    for i in range(1, n_windows + 1):
        window_end = _INCIDENT_START_TS - pd.Timedelta(days=(i - 1) * window_days)
        window_start = window_end - pd.Timedelta(days=window_days)
        post_end = window_end + pd.Timedelta(days=30)

        obs_pos = all_positives[all_positives["event_ts"] < window_start]
        if obs_pos.empty:
            continue

        user_obs_counts = obs_pos.groupby("user_id")["edition_id"].count()
        active_users = set(
            user_obs_counts[user_obs_counts >= min_obs_interactions].index.tolist()
        )
        if not active_users:
            continue

        window_events = all_positives[
            (all_positives["event_ts"] >= window_start)
            & (all_positives["event_ts"] < window_end)
            & (all_positives["user_id"].isin(active_users))
        ][["user_id", "edition_id", "event_type", "event_ts"]].copy()
        
        pseudo_pos = _sample_structured_pseudo_pairs(
            window_events=window_events,
            user_hist_counts=user_hist_counts_full,
            type_mult=type_mult,
            hour_mult=hour_mult,
            bucket_mult=bucket_mult,
            rng=np.random.default_rng(4242 + i),
            mask_fraction=0.20,
        ).assign(label=1)
        if pseudo_pos.empty:
            continue

        # Restrict to target users (candidates only contain target users)
        target_users = set(candidates["user_id"].unique().tolist())
        active_target_users = active_users & target_users
        if not active_target_users:
            continue

        pseudo_pos = pseudo_pos[pseudo_pos["user_id"].isin(active_target_users)]
        if pseudo_pos.empty:
            continue

        fold_candidates = candidates[candidates["user_id"].isin(active_target_users)].copy()
        fold_candidates = _trim_fold_candidates_for_training(
            fold_candidates=fold_candidates,
            per_source_k=120,
        )
        # Optional: injection can make pseudo-validation too easy and hurt transfer.
        if _INJECT_PSEUDO_POSITIVES:
            fold_candidates = _inject_pseudo_positives(fold_candidates, pseudo_pos)

        # Build observed data for this fold (excluding the masked pseudo-positives)
        inc_observed = window_events.merge(
            pseudo_pos[["user_id", "edition_id"]].assign(_m=1), 
            on=["user_id", "edition_id"], 
            how="left"
        )
        inc_observed = inc_observed[inc_observed["_m"].isna()].drop(columns=["_m"])
        
        post_df = all_positives[
            (all_positives["event_ts"] >= window_end)
            & (all_positives["event_ts"] < post_end)
        ]
        post_df = post_df.merge(
            pseudo_pos[["user_id", "edition_id"]].assign(_m=1),
            on=["user_id", "edition_id"],
            how="left",
        )
        post_df = post_df[post_df["_m"].isna()].drop(columns=["_m"])
        
        fold_observed = pd.concat([obs_pos, inc_observed, post_df], ignore_index=True)

        try:
            feat = _build_feature_matrix(
                fold_candidates, 
                fold_observed, 
                dataset,
                incident_start=window_start,
                incident_end=window_end,
                post_end=post_end,
            )
        except Exception as exc:
            logger.warning("Feature build failed for window %d: %s", i, exc)
            continue

        labels = (
            feat[["user_id", "edition_id"]]
            .merge(pseudo_pos, on=["user_id", "edition_id"], how="left")["label"]
            .fillna(0)
            .astype(int)
        )

        n_pos = int((labels == 1).sum())
        if n_pos < 5:
            logger.info("Fold %d: only %d positives, skipping.", i, n_pos)
            continue

        # Subsample negatives so each fold fits in memory.
        # Keep all positives and cap negatives to memory budget.
        n_neg_budget = max(0, max_rows_per_fold - n_pos)
        pos_idx = np.where(labels.to_numpy() == 1)[0]
        neg_idx = np.where(labels.to_numpy() == 0)[0]
        if len(neg_idx) > n_neg_budget:
            rng_fold = np.random.default_rng(42 + i)
            neg_idx = rng_fold.choice(neg_idx, size=n_neg_budget, replace=False)
        keep_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
        feat = feat.iloc[keep_idx].reset_index(drop=True)
        labels = labels.iloc[keep_idx].reset_index(drop=True)

        feat = feat.copy()
        feat["_fold"] = i
        fold_frames.append(feat)
        fold_labels.append(labels)
        fold_groups.append(feat["user_id"].reset_index(drop=True))

        logger.info(
            "Fold %d: obs_cutoff=%s, window=[%s, %s], target_users=%d, pos=%d, rows=%d",
            i,
            window_start.date(),
            window_start.date(),
            window_end.date(),
            len(active_target_users),
            n_pos,
            len(feat),
        )

    if not fold_frames:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

    feat_all = pd.concat(fold_frames, ignore_index=True)
    y_all = pd.concat(fold_labels, ignore_index=True)
    groups_all = pd.concat(fold_groups, ignore_index=True)
    return feat_all, y_all, groups_all


# ─────────────────────────────────────────────────────────────────────────────
# Utility: sort pool by group (CatBoost ranking requirement)
# ─────────────────────────────────────────────────────────────────────────────


def _sort_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return X, y, groups sorted by group value (required by CatBoost Pool)."""
    order = groups.argsort()
    return (
        X.iloc[order].reset_index(drop=True),
        y.iloc[order].reset_index(drop=True),
        groups.iloc[order].reset_index(drop=True),
    )


def _to_str_cats(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
    return X


def _per_user_percentile(values: np.ndarray, user_ids: pd.Series) -> np.ndarray:
    """Convert raw model scores to per-user percentile ranks in [0, 1]."""
    frame = pd.DataFrame({"user_id": user_ids.to_numpy(), "value": values})
    frame["pct"] = (
        frame.groupby("user_id")["value"]
        .rank(method="average", pct=True)
        .astype(float)
    )
    return frame["pct"].to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback ranker (kept as robust safety net)
# ─────────────────────────────────────────────────────────────────────────────


class SimpleBlendRanker:
    """Blend sources with weighted scores and enforce top-k per user."""

    def __init__(self, source_weights: dict[str, float] | None = None) -> None:
        self.source_weights = source_weights or {}

    def _apply_weights(self, candidates: pd.DataFrame) -> pd.DataFrame:
        weighted = candidates.copy()
        weighted["weight"] = weighted["source"].map(self.source_weights).fillna(1.0)
        weighted["final_score"] = weighted["score"] * weighted["weight"]
        return weighted

    def rank(self, dataset: Dataset, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        if candidates.empty:
            return self._fallback_only(dataset, k)

        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        filtered = candidates.merge(
            seen.assign(_seen=1), on=["user_id", "edition_id"], how="left"
        )
        filtered = filtered[filtered["_seen"].isna()].drop(columns=["_seen"])
        if filtered.empty:
            return self._fallback_only(dataset, k)

        filtered = self._apply_weights(filtered)
        blended = (
            filtered.groupby(["user_id", "edition_id"], as_index=False)["final_score"]
            .max()
            .sort_values(
                ["user_id", "final_score", "edition_id"], ascending=[True, False, True]
            )
        )
        selected = blended.groupby("user_id", group_keys=False).head(k).copy()
        selected["rank"] = selected.groupby("user_id").cumcount() + 1
        selected = selected[["user_id", "edition_id", "rank", "final_score"]]

        completed = self._apply_fallback(selected, dataset, k)
        return completed.sort_values(["user_id", "rank"]).reset_index(drop=True)

    def _fallback_only(self, dataset: Dataset, k: int) -> pd.DataFrame:
        positives = dataset.interactions_df[
            dataset.interactions_df["event_type"].isin([1, 2])
        ]
        popularity = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "pop"})
            .sort_values(["pop", "edition_id"], ascending=[False, True])
        )
        ranked_editions = popularity["edition_id"].tolist()
        seen_pairs: set[tuple[int, int]] = set(
            tuple(x)
            for x in dataset.seen_positive_df[
                ["user_id", "edition_id"]
            ].drop_duplicates().to_numpy()
        )
        rows: list[dict[str, int | float]] = []
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
        self, selected: pd.DataFrame, dataset: Dataset, k: int
    ) -> pd.DataFrame:
        positives = dataset.interactions_df[
            dataset.interactions_df["event_type"].isin([1, 2])
        ]
        popularity = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "pop"})
            .sort_values(["pop", "edition_id"], ascending=[False, True])
        )
        popular_editions = popularity["edition_id"].tolist()
        seen_pairs: set[tuple[int, int]] = set(
            tuple(x)
            for x in dataset.seen_positive_df[
                ["user_id", "edition_id"]
            ].drop_duplicates().to_numpy()
        )
        chosen_pairs: set[tuple[int, int]] = set(
            tuple(x) for x in selected[["user_id", "edition_id"]].to_numpy()
        )
        missing_rows: list[dict[str, int | float]] = []
        by_user_counts = selected.groupby("user_id").size().to_dict()

        for user_id in dataset.targets_df["user_id"].tolist():
            count = int(by_user_counts.get(int(user_id), 0))
            if count >= k:
                continue
            rank = count + 1
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
            selected = pd.concat(
                [selected, pd.DataFrame(missing_rows)], ignore_index=True
            )
        return selected


# ─────────────────────────────────────────────────────────────────────────────
# Layer B/C: CatBoostRanker with YetiRank + ensemble
# ─────────────────────────────────────────────────────────────────────────────


class CatBoostRanker:
    """CatBoostRanker with YetiRank objective trained on rolling pseudo-incidents.

    Fixes vs. the old CatBoostClassifier baseline:
      1. Objective: YetiRank (NDCG-aligned) instead of Logloss.
      2. Eval metric: NDCG:top=20 to match competition cutoff exactly.
      3. Grouping: catboost.Pool with group_id=user_id (required for ranking).
      4. Early stopping: eval_set passed to model.fit() (was a no-op before).
      5. Temporal leakage fix: features built from observation window only.
      6. Multiple rolling pseudo-incident windows for richer supervision.
      7. Publisher affinity added to cross features.
      8. Rating-based features added.
      9. log1p transforms on heavy-tailed popularity counts.
      10. Ensemble: YetiRank (70%) + classifier (20%) + RRF (10%).
    """

    def __init__(
        self,
        n_rolling_windows: int = _N_ROLLING_WINDOWS,
        pseudo_incident_days: int = _PSEUDO_INCIDENT_DAYS,
        catboost_iterations: int = 400,
        catboost_depth: int = 5,
        catboost_lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_rolling_windows = n_rolling_windows
        self.pseudo_incident_days = pseudo_incident_days
        self.catboost_iterations = catboost_iterations
        self.catboost_depth = catboost_depth
        self.catboost_lr = catboost_lr
        self.seed = seed

    # ------------------------------------------------------------------
    # Layer B: model trainers
    # ------------------------------------------------------------------

    def _train_yeti_ranker(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        groups_val: pd.Series,
        cat_features: list[str],
    ) -> object | None:
        """Train CatBoostRanker with YetiRank and proper eval_set."""
        try:
            from catboost import CatBoostRanker as _CBRanker, Pool
        except ImportError:
            return None

        X_tr, y_tr, g_tr = _sort_by_group(X_train, y_train, groups_train)
        X_v, y_v, g_v = _sort_by_group(X_val, y_val, groups_val)

        train_pool = Pool(
            _to_str_cats(X_tr, cat_features),
            label=y_tr,
            group_id=g_tr.to_numpy(),
            cat_features=cat_features,
        )
        val_pool = Pool(
            _to_str_cats(X_v, cat_features),
            label=y_v,
            group_id=g_v.to_numpy(),
            cat_features=cat_features,
        )

        # Early stopping is deliberately disabled: all pseudo-incident validation
        # folds are nearly trivially predictable (recent clean-history interactions
        # have very strong affinity signals), so NDCG approaches 1.0 within a few
        # iterations regardless of fold choice.  Training the full N iterations
        # gives the model more capacity to learn fine-grained ranking signals that
        # matter on the actual incident window.
        model = _CBRanker(
            iterations=self.catboost_iterations,
            learning_rate=self.catboost_lr,
            depth=self.catboost_depth,
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=20",
            random_seed=self.seed,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            use_best_model=False,
            verbose=50,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_pool, eval_set=val_pool)

        best_iter = getattr(model, "best_iteration_", "?")
        logger.info(
            "YetiRank trained: %d train / %d val rows, best_iteration=%s",
            len(X_train),
            len(X_val),
            best_iter,
        )
        return model

    def _train_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list[str],
    ) -> object | None:
        """Train CatBoostClassifier as secondary pointwise signal."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return None

        model = CatBoostClassifier(
            iterations=300,
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
            model.fit(
                _to_str_cats(X_train, cat_features),
                y_train,
                cat_features=cat_features,
                eval_set=(
                    _to_str_cats(X_val, cat_features),
                    y_val,
                ),
            )
        return model

    # ------------------------------------------------------------------
    # Layer C: inference + ensemble + fallback
    # ------------------------------------------------------------------

    def rank(self, dataset: Dataset, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """Re-rank with YetiRank ensemble; fall back to blend on any failure."""
        if candidates.empty:
            return SimpleBlendRanker()._fallback_only(dataset, k)

        all_positives = dataset.interactions_df[
            dataset.interactions_df["event_type"].isin([1, 2])
        ]

        try:
            # Build rolling training data
            feat_all, y_all, groups_all = _generate_rolling_training_data(
                candidates=candidates,
                all_positives=all_positives,
                dataset=dataset,
                n_windows=self.n_rolling_windows,
                window_days=self.pseudo_incident_days,
            )

            id_cols = {"user_id", "edition_id", "_fold"}
            cat_features = ["age_restriction", "language_id", "gender", "age_bucket"]

            if feat_all.empty or int((y_all == 1).sum()) < 10:
                logger.warning(
                    "Insufficient training signal (%d positives); using blend fallback.",
                    int((y_all == 1).sum()) if not feat_all.empty else 0,
                )
                raise ValueError("Insufficient training data")

            # Fold 1 (most recent pseudo-window, closest to actual incident) is
            # used for the eval_set to monitor training loss.  Early stopping is
            # disabled on the ranker, so all iterations are trained.
            # Earlier folds (2..n) → training data; fold 1 → eval monitoring only.
            val_fold = int(feat_all["_fold"].min())
            
            # Source scores are computed on full observed data and can leak
            # future information into rolling pseudo-folds. Train without them
            # and add source priors only at ensemble stage.
            source_prefixes = ("src_", "rank_")
            source_exact = {"rrf_score", "n_sources"}
            feature_cols = [
                c
                for c in feat_all.columns
                if c not in id_cols
                and not any(c.startswith(p) for p in source_prefixes)
                and c not in source_exact
            ]

            train_mask = feat_all["_fold"] > val_fold
            val_mask = feat_all["_fold"] == val_fold

            X_train = feat_all.loc[train_mask, feature_cols].reset_index(drop=True)
            y_train = y_all[train_mask].reset_index(drop=True)
            g_train = groups_all[train_mask].reset_index(drop=True)

            X_val = feat_all.loc[val_mask, feature_cols].reset_index(drop=True)
            y_val = y_all[val_mask].reset_index(drop=True)
            g_val = groups_all[val_mask].reset_index(drop=True)

            n_pos_train = int((y_train == 1).sum())
            n_pos_val = int((y_val == 1).sum())
            logger.info(
                "Training set: %d rows, %d positives | Val set: %d rows, %d positives",
                len(X_train),
                n_pos_train,
                len(X_val),
                n_pos_val,
            )

            ranker_model = None
            clf_model = None

            if n_pos_train >= 10 and n_pos_val >= 5:
                ranker_model = self._train_yeti_ranker(
                    X_train, y_train, g_train,
                    X_val, y_val, g_val,
                    cat_features,
                )
                clf_model = self._train_classifier(
                    X_train, y_train, X_val, y_val, cat_features
                )

            # Build inference feature matrix with full observed history
            feat_infer = _build_feature_matrix(
                candidates, 
                all_positives, 
                dataset,
                incident_start=_INCIDENT_START_TS,
                incident_end=_INCIDENT_END_TS,
                post_end=_POST_INCIDENT_END_TS,
            )
            X_infer = feat_infer[feature_cols].copy()

            # Compute component scores
            ranker_scores: np.ndarray | None = None
            clf_scores: np.ndarray | None = None

            if ranker_model is not None:
                X_infer_cat = _to_str_cats(X_infer, cat_features)
                ranker_scores = ranker_model.predict(X_infer_cat)

            if clf_model is not None:
                X_infer_cat = _to_str_cats(X_infer, cat_features)
                clf_scores = clf_model.predict_proba(X_infer_cat)[:, 1]

            rrf_scores: np.ndarray | None = None
            if "rrf_score" in feat_infer.columns:
                rrf_scores = feat_infer["rrf_score"].to_numpy()

            # Per-user normalized ensemble is more robust than global min-max:
            # scales differ a lot across users and candidate pools.
            final_scores = np.zeros(len(feat_infer), dtype=float)
            user_ids_infer = feat_infer["user_id"].reset_index(drop=True)

            if ranker_scores is not None:
                ranker_pct = _per_user_percentile(ranker_scores, user_ids_infer)
                final_scores += 0.25 * ranker_pct

            if clf_scores is not None:
                clf_pct = _per_user_percentile(clf_scores, user_ids_infer)
                final_scores += 0.15 * clf_pct

            src_cols = [c for c in feat_infer.columns if c.startswith("src_")]
            if src_cols:
                src_sum = feat_infer[src_cols].sum(axis=1).to_numpy(dtype=float)
                src_pct = _per_user_percentile(src_sum, user_ids_infer)
                final_scores += 0.45 * src_pct

            if rrf_scores is not None:
                rrf_pct = _per_user_percentile(rrf_scores, user_ids_infer)
                final_scores += 0.15 * rrf_pct

            # If both models failed, rely on source-only ranking.
            if ranker_scores is None and clf_scores is None:
                if src_cols:
                    final_scores = feat_infer[src_cols].sum(axis=1).to_numpy(dtype=float)
                else:
                    final_scores = np.zeros(len(feat_infer), dtype=float)

            result = feat_infer[["user_id", "edition_id"]].copy()
            result["final_score"] = final_scores

        except Exception as exc:
            logger.warning("CatBoostRanker failed (%s); falling back to blend.", exc)
            result = (
                candidates.groupby(["user_id", "edition_id"], as_index=False)["score"]
                .max()
                .rename(columns={"score": "final_score"})
            )

        # Filter seen positives, select top-k, fill gaps with popularity
        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        result = result.merge(
            seen.assign(_seen=1), on=["user_id", "edition_id"], how="left"
        )
        result = result[result["_seen"].isna()].drop(columns=["_seen"])

        result = result.sort_values(
            ["user_id", "final_score"], ascending=[True, False]
        )
        selected = result.groupby("user_id", group_keys=False).head(k).copy()
        selected["rank"] = selected.groupby("user_id").cumcount() + 1
        selected = selected[["user_id", "edition_id", "rank", "final_score"]]

        completed = SimpleBlendRanker()._apply_fallback(selected, dataset, k)
        return completed.sort_values(["user_id", "rank"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def rank_predictions(
    dataset: Dataset,
    candidates: pd.DataFrame,
    source_weights: dict[str, float],
    k: int,
    use_catboost: bool = True,
) -> pd.DataFrame:
    """Rank candidates using the configured strategy.

    When ``use_catboost=True`` (default) a :class:`CatBoostRanker` with
    ``YetiRank`` objective is used.  Training is performed on-the-fly on
    multiple rolling pseudo-incident windows derived from the clean interaction
    history.  Falls back to :class:`SimpleBlendRanker` if CatBoost is
    unavailable or training fails.

    Args:
        dataset:        Runtime dataset with raw interactions and catalogue.
        candidates:     Candidate rows from all configured generators.
        source_weights: Per-source multipliers for the blend fallback.
        k:              Required top-k output per user.
        use_catboost:   Whether to attempt the CatBoost re-ranker.

    Returns:
        DataFrame with columns ``user_id``, ``edition_id``, ``rank``,
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
