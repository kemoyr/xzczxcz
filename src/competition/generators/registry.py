"""Registry for participant-configurable generator factories."""

from __future__ import annotations

from collections.abc import Callable

from src.competition.generators.global_popularity import GlobalPopularityGenerator
from src.competition.generators.global_temporal_popularity import (
    GlobalTemporalPopularityGenerator,
)
from src.competition.generators.item_cooccurrence import ItemCooccurrenceGenerator
from src.competition.generators.post_incident_author_profile import (
    PostIncidentAuthorProfileGenerator,
)
from src.competition.generators.post_incident_genre_profile import (
    PostIncidentGenreProfileGenerator,
)
from src.competition.generators.post_incident_language_profile import (
    PostIncidentLanguageProfileGenerator,
)
from src.competition.generators.post_incident_publisher_profile import (
    PostIncidentPublisherProfileGenerator,
)
from src.competition.generators.svd_cf import SVDCollaborativeGenerator
from src.competition.generators.user_author import UserAuthorGenerator
from src.competition.generators.user_author_recent import UserAuthorRecentGenerator
from src.competition.generators.user_genre import UserGenrePopularityGenerator
from src.competition.generators.user_genre_recent import UserGenreRecentGenerator
from src.competition.generators.user_language import UserLanguageGenerator
from src.competition.generators.user_publisher import UserPublisherGenerator

GeneratorFactory = Callable[[dict[str, float], bool], object]


def _build_global_popularity(params: dict[str, float], tqdm_enabled: bool) -> object:
    del params
    return GlobalPopularityGenerator(show_progress=tqdm_enabled)


def _build_global_temporal_popularity(params: dict[str, float], tqdm_enabled: bool) -> object:
    return GlobalTemporalPopularityGenerator(
        w_incident=float(params.get("w_incident", 2.0)),
        w_post_incident=float(params.get("w_post_incident", 1.5)),
        w_w30=float(params.get("w_w30", 1.0)),
        w_w14=float(params.get("w_w14", 0.8)),
        w_trend=float(params.get("w_trend", 0.6)),
        w_weighted=float(params.get("w_weighted", 0.5)),
        w_w90=float(params.get("w_w90", 0.4)),
        show_progress=tqdm_enabled,
    )


def _build_user_genre(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserGenrePopularityGenerator(
        genre_smoothing=float(params.get("genre_smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )


def _build_user_genre_recent(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserGenreRecentGenerator(
        genre_smoothing=float(params.get("genre_smoothing", 0.5)),
        show_progress=tqdm_enabled,
    )


def _build_user_author(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserAuthorGenerator(
        author_smoothing=float(params.get("author_smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )


def _build_user_author_recent(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserAuthorRecentGenerator(
        author_smoothing=float(params.get("author_smoothing", 0.5)),
        show_progress=tqdm_enabled,
    )


def _build_user_language(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserLanguageGenerator(
        language_smoothing=float(params.get("language_smoothing", 0.5)),
        recency_weight=float(params.get("recency_weight", 0.4)),
        show_progress=tqdm_enabled,
    )


def _build_user_publisher(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserPublisherGenerator(
        publisher_smoothing=float(params.get("publisher_smoothing", 0.5)),
        recency_weight=float(params.get("recency_weight", 0.4)),
        show_progress=tqdm_enabled,
    )


def _build_item_cooccurrence(params: dict[str, float], tqdm_enabled: bool) -> object:
    return ItemCooccurrenceGenerator(
        cooccurrence_days=int(params.get("cooccurrence_days", 90)),
        seed_days=int(params.get("seed_days", 45)),
        max_items_per_user=int(params.get("max_items_per_user", 30)),
        top_per_seed=int(params.get("top_per_seed", 200)),
        show_progress=tqdm_enabled,
    )


def _build_post_incident_genre_profile(
    params: dict[str, float], tqdm_enabled: bool
) -> object:
    return PostIncidentGenreProfileGenerator(
        genre_smoothing=float(params.get("genre_smoothing", 0.4)),
        show_progress=tqdm_enabled,
    )


def _build_post_incident_author_profile(
    params: dict[str, float], tqdm_enabled: bool
) -> object:
    return PostIncidentAuthorProfileGenerator(
        author_smoothing=float(params.get("author_smoothing", 0.4)),
        show_progress=tqdm_enabled,
    )


def _build_post_incident_language_profile(
    params: dict[str, float], tqdm_enabled: bool
) -> object:
    return PostIncidentLanguageProfileGenerator(
        language_smoothing=float(params.get("language_smoothing", 0.3)),
        show_progress=tqdm_enabled,
    )


def _build_post_incident_publisher_profile(
    params: dict[str, float], tqdm_enabled: bool
) -> object:
    return PostIncidentPublisherProfileGenerator(
        publisher_smoothing=float(params.get("publisher_smoothing", 0.3)),
        show_progress=tqdm_enabled,
    )


def _build_svd_cf(params: dict[str, float], tqdm_enabled: bool) -> object:
    return SVDCollaborativeGenerator(
        n_factors=int(params.get("n_factors", 64)),
        read_weight=float(params.get("read_weight", 2.0)),
        wishlist_weight=float(params.get("wishlist_weight", 1.0)),
        show_progress=tqdm_enabled,
    )


GENERATOR_REGISTRY: dict[str, GeneratorFactory] = {
    "global_popularity": _build_global_popularity,
    "global_temporal_popularity": _build_global_temporal_popularity,
    "user_genre": _build_user_genre,
    "user_genre_recent": _build_user_genre_recent,
    "user_author": _build_user_author,
    "user_author_recent": _build_user_author_recent,
    "user_language": _build_user_language,
    "user_publisher": _build_user_publisher,
    "post_incident_genre_profile": _build_post_incident_genre_profile,
    "post_incident_author_profile": _build_post_incident_author_profile,
    "post_incident_language_profile": _build_post_incident_language_profile,
    "post_incident_publisher_profile": _build_post_incident_publisher_profile,
    "item_cooccurrence": _build_item_cooccurrence,
    "svd_cf": _build_svd_cf,
}


def build_generator(name: str, params: dict[str, float], tqdm_enabled: bool = False) -> object:
    """Instantiate a configured generator factory by name.

    Args:
        name: Generator identifier from YAML config.
        params: Generator parameter mapping from YAML config.
        tqdm_enabled: Whether generator may display progress bars.

    Returns:
        Concrete generator instance implementing `.generate(...)`.

    Raises:
        ValueError: If no registered generator matches `name`.
    """
    try:
        factory = GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise ValueError(f"Unknown generator name: {name}. Available: {available}") from exc
    return factory(params, tqdm_enabled)

