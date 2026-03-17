# Feature Assessment Report (`src/competition/features.py`)

## Scope

This report summarizes:

- task requirements from `docs/task/task_description.md`,
- data schema from `docs/task/data_description.md`,
- baseline pipeline context from `docs/baseline/ONBOARDING.md`,
- key EDA conclusions from `notebooks/EDA.ipynb`,
- current implementation state and quality assessment of `src/competition/features.py`.

---

## 1) Problem and pipeline context

### Task objective

For each user in `targets.csv`, the solution must output exactly 20 ranked `edition_id` predictions representing likely **lost positive interactions**.

- Positive events: `event_type in {1, 2}` (`wishlist`, `read`)
- Prediction unit: unique `(user_id, edition_id)` pair
- Target metric: average `NDCG@20` over users

### Baseline pipeline role of features

Per baseline docs, pipeline stages are:

1. `prepare_data`
2. `build_features`
3. `generate_candidates`
4. `rank_and_select`
5. `make_submission`

`src/competition/features.py` is executed at **build_features** stage and produces `features.parquet` for candidate generators/ranker.

---

## 2) Data and EDA signals relevant to features

Based on provided documentation and notebook outputs:

- interactions are highly sparse in user-item space;
- item popularity is long-tail (top items cover limited interaction share);
- all target users in current EDA snapshot have some positive history (no strict cold-start in that split), but history length varies strongly;
- explicit time segmentation is important:
  - incident window: `2025-10-01` to `2025-11-01`
  - post-incident observed period: `2025-11-01` to `2025-12-01`
- genre/author affinities are stable and useful;
- recommended next feature directions in notebook include temporal counters/trends and additional affinity dimensions (language/publisher).

---

## 3) What currently exists in `src/competition/features.py`

The file builds a **long-form feature table** with unified schema:

- `feature_type`
- `user_id`
- `edition_id`
- `genre_id`
- `author_id`
- `value`

Non-applicable dimensions are filled with `pd.NA`.

### Implemented feature blocks

1. **`edition_popularity_all`**
   - grain: `edition_id`
   - value: number of unique users with positive interactions (`nunique(user_id)`)

2. **`edition_popularity_recent`**
   - grain: `edition_id`
   - value: same popularity metric but filtered to recent window:
     `event_ts >= max(event_ts) - recent_days`

3. **`user_genre_profile`**
   - grain: `(user_id, genre_id)`
   - built through joins:
     - positives + `catalog_df` (`edition_id -> book_id`)
     - then + `book_genres_df` (`book_id -> genre_id`)
   - value: per-user normalized share of interactions in each genre

4. **`user_author_profile`**
   - grain: `(user_id, author_id)`
   - built via positives + `catalog_df` (`edition_id -> author_id`)
   - value: per-user normalized share of interactions per author

### Technical behavior

- Positive filter is correctly aligned with task definition (`event_type` in `[1, 2]`).
- Output combines all feature blocks via `pd.concat(..., ignore_index=True)`.
- No model-specific logic inside feature builder (model-agnostic design).

---

## 4) Integration check: which features are actually used

From generator implementations:

- `global_popularity` generator uses `edition_popularity_all`
- `user_genre` generator uses:
  - `user_genre_profile`
  - `edition_popularity_all` as prior
- `user_author` generator uses:
  - `user_author_profile`
  - `edition_popularity_all` as prior

Important finding:

- `edition_popularity_recent` is currently **computed but not consumed** by existing generators/ranker.

---

## 5) Strengths of current implementation

- Correct and consistent positive-event definition.
- Good baseline decomposition: global recall + personalized affinity profiles.
- Clean and extensible long-table design (`feature_type`-driven).
- Feature naming matches downstream expectations (no contract mismatch).
- Simple deterministic computations; low risk of schema instability.

---

## 6) Gaps and risks

1. **Temporal underutilization**
   - Recent popularity is generated but not used.
   - No explicit incident/post-incident counter features in current table.

2. **No event-type intensity handling**
   - `wishlist` and `read` are treated equally in aggregation.
   - Could lose signal if types carry different confidence.

3. **No user-state features**
   - Missing activity depth/recency markers at user level.

4. **No additional affinity dimensions**
   - No user-language or user-publisher profiles despite available schema.

5. **No trend features**
   - Missing multi-window (`7/14/30/90`) counts and trend deltas/slopes.

6. **Potential evaluation leakage risk if not time-sliced for validation**
   - Current builder uses full observed interactions unless pipeline split logic constrains input externally.

---

## 7) Overall assessment

`src/competition/features.py` is a solid, readable baseline feature builder and correctly supports current baseline generators.  
It is structurally good but feature coverage is still basic for this task: the code captures static popularity and author/genre preference, while temporal behavior and richer affinity signals (highlighted by EDA) remain mostly unimplemented or unused.

---

## 8) Recommended next feature upgrades (priority order)

1. Wire `edition_popularity_recent` into at least one generator and/or ranking blend.
2. Add explicit time-window features (`7/14/30/90`) for edition/author/genre.
3. Add `user_language_profile` and `user_publisher_profile`.
4. Add user activity/recency features (e.g., days since last positive, interaction count buckets).
5. Test differentiated weights for `event_type=1` vs `event_type=2`.
6. Re-run local validation and compare `NDCG@20` against baseline.

