"""Feature construction entrypoint for participant solution."""

from __future__ import annotations

import pandas as pd

from src.platform.core.dataset import Dataset

FEATURE_COLUMNS = [
    "feature_type",
    "user_id",
    "edition_id",
    "genre_id",
    "author_id",
    "language_id",
    "publisher_id",
    "value",
]
ID_COLUMNS = [
    "user_id",
    "edition_id",
    "genre_id",
    "author_id",
    "language_id",
    "publisher_id",
]
INCIDENT_START_TS = pd.Timestamp("2025-10-01 00:00:00")
INCIDENT_END_TS = pd.Timestamp("2025-11-01 00:00:00")
POST_INCIDENT_END_TS = pd.Timestamp("2025-12-01 00:00:00")


def _empty_features_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _finalize_block(df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    if df.empty:
        return _empty_features_frame()
    block = df.copy()
    for column in ID_COLUMNS:
        if column not in block.columns:
            block[column] = pd.NA
    block["feature_type"] = feature_type
    block["value"] = block["value"].astype(float)
    return block[FEATURE_COLUMNS]


def _edition_popularity_block(positives: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    if positives.empty:
        return _empty_features_frame()
    grouped = (
        positives.groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "value"})
    )
    return _finalize_block(grouped, feature_type)


def _normalized_profile(
    positives: pd.DataFrame,
    mapping: pd.DataFrame,
    key_column: str,
    feature_type: str,
) -> pd.DataFrame:
    if positives.empty or mapping.empty:
        return _empty_features_frame()
    joined = positives[["user_id", "edition_id"]].merge(mapping, on="edition_id", how="inner")
    if joined.empty:
        return _empty_features_frame()
    profile = (
        joined.groupby(["user_id", key_column], as_index=False)["edition_id"]
        .count()
        .rename(columns={"edition_id": "value"})
    )
    profile["value"] = profile["value"] / profile.groupby("user_id")["value"].transform("sum")
    return _finalize_block(profile, feature_type)


def _user_state_features(positives: pd.DataFrame, max_ts: pd.Timestamp) -> list[pd.DataFrame]:
    if positives.empty:
        return []
    unique_pairs = (
        positives.sort_values("event_ts")
        .drop_duplicates(subset=["user_id", "edition_id"], keep="last")
        .copy()
    )

    last_positive = (
        positives.groupby("user_id", as_index=False)["event_ts"]
        .max()
        .rename(columns={"event_ts": "last_event_ts"})
    )
    last_positive["value"] = (max_ts - last_positive["last_event_ts"]).dt.days.astype(float)
    days_since_last = _finalize_block(
        last_positive[["user_id", "value"]], "user_days_since_last_positive"
    )

    w30 = unique_pairs[unique_pairs["event_ts"] >= max_ts - pd.Timedelta(days=30)]
    w90 = unique_pairs[unique_pairs["event_ts"] >= max_ts - pd.Timedelta(days=90)]
    user_w30 = (
        w30.groupby("user_id", as_index=False)["edition_id"]
        .nunique()
        .rename(columns={"edition_id": "value"})
    )
    user_w90 = (
        w90.groupby("user_id", as_index=False)["edition_id"]
        .nunique()
        .rename(columns={"edition_id": "value"})
    )
    cnt_w30 = _finalize_block(user_w30, "user_positive_count_w30")
    cnt_w90 = _finalize_block(user_w90, "user_positive_count_w90")

    ratio = user_w90.merge(user_w30, on="user_id", how="left", suffixes=("_w90", "_w30"))
    ratio["value_w30"] = ratio["value_w30"].fillna(0.0)
    ratio["value"] = ratio["value_w30"] / (ratio["value_w90"] + 1.0)
    recent_ratio = _finalize_block(ratio[["user_id", "value"]], "user_recent_to_long_ratio")
    return [days_since_last, cnt_w30, cnt_w90, recent_ratio]


def build_features_frame(dataset: Dataset, recent_days: int) -> pd.DataFrame:
    """Build baseline feature matrix consumed by candidate generators.

    The function creates a compact long-form feature table that keeps the
    baseline extensible while staying model-agnostic. The generated feature
    blocks encode global popularity and user preference profiles over genres
    and authors.

    Args:
        dataset: Runtime dataset with interactions, catalog, and taxonomy tables.
        recent_days: Time window in days for recency popularity signal.

    Returns:
        Long-form feature DataFrame with columns:
        `feature_type`, `user_id`, `edition_id`, `genre_id`, `author_id`,
        `language_id`, `publisher_id`, `value`.
    """
    positives = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])]
    if positives.empty:
        return _empty_features_frame()

    max_ts = positives["event_ts"].max()
    feature_blocks: list[pd.DataFrame] = []

    feature_blocks.append(_edition_popularity_block(positives, "edition_popularity_all"))

    recent = positives[positives["event_ts"] >= max_ts - pd.Timedelta(days=recent_days)]
    feature_blocks.append(_edition_popularity_block(recent, "edition_popularity_recent"))

    for window in (7, 14, 30, 90):
        window_df = positives[positives["event_ts"] >= max_ts - pd.Timedelta(days=window)]
        feature_blocks.append(
            _edition_popularity_block(window_df, f"edition_popularity_w{window}")
        )

    incident_df = positives[
        (positives["event_ts"] >= INCIDENT_START_TS) & (positives["event_ts"] < INCIDENT_END_TS)
    ]
    post_incident_df = positives[
        (positives["event_ts"] >= INCIDENT_END_TS)
        & (positives["event_ts"] < POST_INCIDENT_END_TS)
    ]
    feature_blocks.append(
        _edition_popularity_block(incident_df, "edition_popularity_incident")
    )
    feature_blocks.append(
        _edition_popularity_block(post_incident_df, "edition_popularity_post_incident")
    )

    pop_w14 = (
        positives[positives["event_ts"] >= max_ts - pd.Timedelta(days=14)]
        .groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "w14"})
    )
    pop_w90 = (
        positives[positives["event_ts"] >= max_ts - pd.Timedelta(days=90)]
        .groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "w90"})
    )
    trend = pop_w90.merge(pop_w14, on="edition_id", how="left")
    trend["w14"] = trend["w14"].fillna(0.0)
    trend["value"] = trend["w14"] / (trend["w90"] + 1.0)
    feature_blocks.append(
        _finalize_block(
            trend[["edition_id", "value"]],
            "edition_popularity_trend_short_long",
        )
    )

    read_pop = positives[positives["event_type"] == 2]
    wishlist_pop = positives[positives["event_type"] == 1]
    feature_blocks.append(_edition_popularity_block(read_pop, "edition_popularity_read"))
    feature_blocks.append(
        _edition_popularity_block(wishlist_pop, "edition_popularity_wishlist")
    )
    read_counts = (
        read_pop.groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "read_value"})
    )
    wishlist_counts = (
        wishlist_pop.groupby("edition_id", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "wishlist_value"})
    )
    weighted = read_counts.merge(wishlist_counts, on="edition_id", how="outer").fillna(0.0)
    weighted["value"] = 1.25 * weighted["read_value"] + 1.0 * weighted["wishlist_value"]
    feature_blocks.append(
        _finalize_block(weighted[["edition_id", "value"]], "edition_popularity_weighted")
    )

    user_genre_map = (
        dataset.catalog_df[["edition_id", "book_id"]]
        .merge(dataset.book_genres_df[["book_id", "genre_id"]], on="book_id", how="inner")
        .drop(columns=["book_id"])
    )
    user_author_map = dataset.catalog_df[["edition_id", "author_id"]].copy()
    user_language_map = dataset.catalog_df[["edition_id", "language_id"]].copy()
    user_publisher_map = dataset.catalog_df[["edition_id", "publisher_id"]].copy()

    feature_blocks.append(
        _normalized_profile(positives, user_genre_map, "genre_id", "user_genre_profile")
    )
    feature_blocks.append(
        _normalized_profile(positives, user_author_map, "author_id", "user_author_profile")
    )
    feature_blocks.append(
        _normalized_profile(positives, user_language_map, "language_id", "user_language_profile")
    )
    feature_blocks.append(
        _normalized_profile(
            positives,
            user_publisher_map,
            "publisher_id",
            "user_publisher_profile",
        )
    )

    recent_profile_window = positives[
        positives["event_ts"] >= max_ts - pd.Timedelta(days=recent_days)
    ]
    feature_blocks.append(
        _normalized_profile(
            recent_profile_window,
            user_genre_map,
            "genre_id",
            "user_genre_profile_recent",
        )
    )
    feature_blocks.append(
        _normalized_profile(
            recent_profile_window,
            user_author_map,
            "author_id",
            "user_author_profile_recent",
        )
    )
    feature_blocks.append(
        _normalized_profile(
            recent_profile_window,
            user_language_map,
            "language_id",
            "user_language_profile_recent",
        )
    )
    feature_blocks.append(
        _normalized_profile(
            recent_profile_window,
            user_publisher_map,
            "publisher_id",
            "user_publisher_profile_recent",
        )
    )

    feature_blocks.extend(_user_state_features(positives=positives, max_ts=max_ts))

    # Incident-window user profiles: capture what users were consuming during
    # the loss period itself.  These are the most direct signal for recovery
    # since the missing interactions are drawn from the same distribution.
    feature_blocks.append(
        _normalized_profile(
            incident_df,
            user_genre_map,
            "genre_id",
            "user_genre_profile_incident",
        )
    )
    feature_blocks.append(
        _normalized_profile(
            incident_df,
            user_author_map,
            "author_id",
            "user_author_profile_incident",
        )
    )
    feature_blocks.append(
        _normalized_profile(
            incident_df,
            user_language_map,
            "language_id",
            "user_language_profile_incident",
        )
    )

    # Clean-history popularity: popularity of editions in the 150 days before
    # the incident, useful as a long-run baseline to contrast with incident pop.
    clean_history_df = positives[positives["event_ts"] < INCIDENT_START_TS]
    feature_blocks.append(
        _edition_popularity_block(clean_history_df, "edition_popularity_clean_history")
    )

    # Per-user incident-window activity count: users who were more active during
    # the incident window have higher expected number of lost interactions.
    if not incident_df.empty:
        user_incident_count = (
            incident_df.groupby("user_id", as_index=False)["edition_id"]
            .nunique()
            .rename(columns={"edition_id": "value"})
        )
        feature_blocks.append(
            _finalize_block(user_incident_count, "user_positive_count_incident")
        )

    non_empty_blocks = [block for block in feature_blocks if not block.empty]
    if not non_empty_blocks:
        return _empty_features_frame()
    return pd.concat(non_empty_blocks, ignore_index=True)

