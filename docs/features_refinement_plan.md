# Features Refinement Plan for `src/competition/features.py`

## 1) Goal and constraints

Target task: for each user in `targets.csv`, produce top-20 `edition_id` likely to be **lost positive pairs** `(user_id, edition_id)`, evaluated by `NDCG@20`.

From task and baseline context:

- positives are `event_type in {1, 2}`;
- temporal structure matters (history + incident behavior);
- feature stage feeds candidate generators and ranker;
- if a feature is not consumed by generators/ranker, it does not improve score.

## 2) What is strong in current `features.py`

Current implementation already does:

- valid positive-event filtering;
- robust baseline global popularity (`edition_popularity_all`);
- user preference profiles for genre and author (`user_genre_profile`, `user_author_profile`);
- clean long-format schema:
  - `feature_type`, `user_id`, `edition_id`, `genre_id`, `author_id`, `value`.

This is a good baseline, but currently underuses temporal and metadata signals.

## 3) Evidence from EDA and docs that drives refinement

From `notebooks/EDA.ipynb` and task docs:

- strong sparsity/long-tail behavior (global popularity alone is insufficient);
- fixed important windows are explicit:
  - incident: `2025-10-01` -> `2025-11-01`;
  - fully observed post-incident: `2025-11-01` -> `2025-12-01`;
- target coverage in this snapshot:
  - `target_users_total = 3862`;
  - `target_users_without_positive_history = 0`;
- user activity dispersion is high (targets are active but heterogeneous);
- top popular items cover small share (long-tail signal remains important);
- EDA conclusion explicitly recommends temporal counters/trends + language/publisher affinities.

Also confirmed from generator code:

- `edition_popularity_recent` is computed but **unused**;
- active generators consume only:
  - `edition_popularity_all`,
  - `user_genre_profile`,
  - `user_author_profile`.

So feature improvements must be paired with consumption logic to matter.

## 4) What to refine in `features.py` (priority order)

## P0 (highest impact, lowest risk)

### 4.1 Add explicit temporal edition popularity features

Add new `feature_type`s at `edition_id` grain:

- `edition_popularity_w7`
- `edition_popularity_w14`
- `edition_popularity_w30`
- `edition_popularity_w90`
- `edition_popularity_incident`
- `edition_popularity_post_incident`
- `edition_popularity_trend_short_long` (for example `w14 / (w90 + eps)` or `w14 - w90_norm`)

Why:

- aligns with incident nature of the problem (missing events are time-localized);
- improves ranking of currently trending items versus stale global-popularity only.

### 4.2 Add event-type-aware popularity variants

At `edition_id` grain:

- `edition_popularity_read` (`event_type=2`)
- `edition_popularity_wishlist` (`event_type=1`)
- optionally weighted combined score:
  - `edition_popularity_weighted` with tunable weights (example: read > wishlist).

Why:

- not all positive events carry same predictive strength;
- this should improve candidate quality for users with mixed behavior.

### 4.3 Add user recency/activity state features

At `user_id` grain (with nullable other IDs):

- `user_days_since_last_positive`
- `user_positive_count_w30`
- `user_positive_count_w90`
- `user_recent_to_long_ratio` (`w30 / (w90 + eps)`).

Why:

- users differ strongly by activity depth/recency;
- enables recency-adaptive scoring and fallback behavior downstream.

## P1 (high impact, moderate implementation)

### 4.4 Add additional affinity profiles from catalog metadata

At `(user_id, language_id)` and `(user_id, publisher_id)` grains:

- `user_language_profile`
- `user_publisher_profile`

Implementation principle: same as genre/author profile (normalized share), with additive smoothing for sparse users.

Why:

- directly recommended by EDA;
- metadata dimensions are available in `catalog_df` and can improve personalization when author/genre coverage is thin.

### 4.5 Add time-aware user profiles

For genre/author/language/publisher, add recency-weighted variants:

- `user_genre_profile_recent`
- `user_author_profile_recent`
- `user_language_profile_recent`
- `user_publisher_profile_recent`

Possible weighting: exponential decay by event age or fixed recent window.

Why:

- preferences drift; lost events are near incident period.

## P2 (useful but should be validated carefully)

### 4.6 Profile stabilization and calibration

- Bayesian/shrinkage normalization for user profiles:
  - blend personal share with global prior for low-count users;
- winsorize/clamp extreme values;
- optional log scaling for raw counts before export.

Why:

- prevents noisy high-confidence scores from tiny histories.

### 4.7 Add cross-features only if downstream can consume them

Examples:

- `user_author_x_recency`
- `user_genre_x_recency`

Only worthwhile if ranker/generators are updated; otherwise this adds compute with no gain.

## 5) Required compatibility notes for current pipeline

Current feature schema has no `language_id` / `publisher_id` columns, so to support new profile types safely:

- either extend schema with nullable `language_id`, `publisher_id`;
- or encode IDs into existing columns is **not recommended** (fragile and ambiguous).

Recommended update:

- evolve schema to include nullable dimension columns:
  - `language_id`, `publisher_id`;
- keep existing columns unchanged for backward compatibility.

Important:

- adding features in `features.py` alone is not enough;
- update generators/ranker to consume them (especially temporal and recent popularity signals).

## 6) Concrete implementation plan (incremental)

1. Refactor `build_features_frame()` into small helpers:
   - `_compute_popularity_features(...)`
   - `_compute_user_profiles(...)`
   - `_normalize_profile(...)`
2. Implement P0 features first (temporal + event-type + user-state).
3. Wire at least one temporal popularity feature into generators immediately:
   - `global_popularity` should use blended score from `all` + `w30`/`incident`.
4. Add P1 metadata profiles (language/publisher) and corresponding generator usage.
5. Run local validation and keep only features with positive delta.

## 7) Validation protocol to avoid false improvements

Use time-based local validation (must mimic incident mechanics):

- train features/candidates on earlier observed window;
- validate on a later window with synthetic hiding of positives;
- metric: `NDCG@20`.

Run ablation sequence:

1. baseline current;
2. baseline + P0 temporal;
3. + event-type-aware;
4. + user-state;
5. + language/publisher profiles;
6. + recency-weighted profiles.

Keep each block only if it improves or stabilizes NDCG.

## 8) Recommended first commit scope

If doing one practical improvement pass focused on ROI:

- implement temporal popularity block + incident/post windows in `features.py`;
- expose user recency/activity features;
- modify `global_popularity` and one personalized generator to consume these features;
- keep feature naming explicit and stable for future ablations.

This should be the fastest path to beat current baseline while preserving maintainability.

