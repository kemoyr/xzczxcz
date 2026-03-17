# Ranker Implementation Plan

## 1. Goal

Build a stronger final ranker for recovering lost positive pairs `(user_id, edition_id)` and maximizing `NDCG@20`, while keeping compatibility with the existing pipeline:

- `prepare_data -> build_features -> generate_candidates -> rank_and_select -> make_submission`
- Output contract: exactly 20 unique `edition_id` per target user, excluding seen positives.

## 2. What exists now (baseline diagnosis)

Current ranker in `src/competition/ranking.py`:

- `SimpleBlendRanker`: weighted source blend (`max` over source scores), then top-k + popularity fallback.
- `CatBoostRanker`: binary classifier trained on one pseudo-incident window (`14` days before fixed incident start), then scored and top-k selected.

Current weaknesses:

1. **Pointwise objective mismatch**: binary `Logloss` is weaker than ranking losses for `NDCG@20`.
2. **Single pseudo-window training**: narrow supervision, weak generalization.
3. **No robust time-based CV**: high risk of unstable local gains.
4. **Limited interaction features**: mostly source scores + basic user/item stats.
5. **No calibrated ensemble**: model fallback exists, but no systematic multi-model fusion.
6. **Non-functional early stopping**: `CatBoostClassifier` is configured with `early_stopping_rounds=30` and `use_best_model=True`, but `model.fit()` is called without an `eval_set`. The model always trains for the full number of iterations, wasting capacity and risking overfit.
7. **Train–test temporal leakage in features**: `_build_feature_matrix` computes user activity, item popularity, and affinity features over **all** observed interactions (including the real incident period), while pseudo-labels are drawn from 14 days before that period. Features therefore see data from the future relative to the pseudo-label window, making local validation optimistic and training signal noisy.
8. **Publisher affinity absent from cross features**: `_cross_features()` computes author, language, and genre affinity but omits publisher, despite the `features.py` layer building `user_publisher_profile` blocks.
9. **Rating signal unused**: `interactions.csv` contains a numeric `rating` field for read events (`event_type=2`). Mean and variance of ratings per user and per item are never computed as features.

## 3. Target ranker architecture

Use a **three-layer ranker stack**.

### Layer A: Candidate-level feature builder

Build dense pair features for each `(user_id, edition_id)` candidate:

- Source features:
  - raw source scores, per-source rank, source presence indicators, `n_sources`.
- User state:
  - activity counts (`w7/w14/w30/w90`), days since last positive, incident-period activity.
- Item state:
  - popularity in `incident`, `post_incident`, `w7/w14/w30/w90`, trend ratios.
- User-item affinity:
  - author/genre/language/publisher overlap and normalized affinity.
- Cross features:
  - `source_score * user_recent_activity`, `source_score * item_trend`, etc.
- Optional advanced features:
  - text embedding similarity (user profile text centroid vs item text embedding),
  - latent embedding dot products from collaborative models.

### Layer B: Learning-to-rank model

Primary model: **`CatBoostRanker` with a listwise ranking objective** trained on pseudo-incident folds.

Recommended setup:

- Objective: `YetiRank` — directly approximates NDCG via stochastic re-sampling of permutations; strictly better aligned with the competition metric than `YetiRankPairwise` (pairwise approximation) or `PairLogit` (pairwise cross-entropy with no NDCG-awareness). Use `YetiRankPairwise` only as a fallback if `YetiRank` is too slow on the available hardware.
- Eval metric: `NDCG:top=20` (must set `top=20` explicitly; the default cutoff in CatBoost's NDCG metric does not match the competition's `@20` cutoff).
- Grouping: `catboost.Pool` with `group_id=user_id` is **required** for any ranking objective in CatBoost. Passing a plain DataFrame without `group_id` raises a runtime error or silently ignores the ranking structure.
- Training labels: pseudo-lost binary labels (0/1) from rolling time slices; for `YetiRank` these are treated as relevance grades, which is correct.
- Early stopping: must pass a held-out `eval_set` to `model.fit()` (see weakness #6 above — the current code omits this, making `early_stopping_rounds` a no-op).
- Strong regularization: `l2_leaf_reg`, `subsample`, `colsample_bylevel`.
- Categorical handling: `language_id`, `age_restriction`, source-id-derived categoricals when useful.

Backup model:

- Keep current blend as fail-safe (`SimpleBlendRanker`) for runtime robustness.

### Layer C: Ensemble and calibration

Blend multiple rankers:

1. `YetiRank` CatBoost ranker (main).
2. Pointwise CatBoost classifier (stability baseline).
3. Source-weight RRF blend (robust fallback / diversity injector).

Combine with validation-tuned weights and optional per-user-segment gating:

- heavy users -> ranker-dominant,
- sparse users -> blend/global-temporal-dominant.

## 4. Labeling and validation strategy (critical)

Move from one pseudo-window to **rolling pseudo-incidents**.

## 4.1 Rolling pseudo-label generation

Create multiple train/validation splits before the real incident:

- For each split:
  1. choose anchor time `T`,
  2. treat `(T - d, T]` as pseudo-incident (hidden positives),
  3. build observed interactions from history **strictly before `T - d`** (observation window),
  4. generate candidates on observed data,
  5. **compute all features exclusively from the observation window** (no data from `[T-d, T]` or later may enter the feature computation — this is the temporal leakage fix for weakness #7),
  6. label candidate pairs `1` if in hidden set, else `0`.
  7. Filter out users with fewer than `min_obs_interactions` (e.g., 3) in the observation window; they produce no useful training signal.

Use several anchors (e.g., 4-8 windows) to increase label diversity and robustness. Concatenate all folds into a single training pool (each fold's rows carry a distinct fold tag for diagnostic grouping).

## 4.2 Metrics to track

- Primary: `mean_ndcg@20`.
- Secondary:
  - hitrate@20,
  - candidate recall@K (for diagnostics),
  - NDCG by user activity buckets (`cold-ish`, medium, heavy),
  - stability across folds.

## 5. Feature engineering plan for ranker

Implement in ranker feature assembly code (mostly `src/competition/ranking.py`).

## 5.1 Must-have features (P0)

1. **Per-source rank features**
   - within-user rank from each generator (rank 1 = highest scored by that source),
   - Reciprocal Rank Fusion score per source: `RRF(rank) = 1 / (60 + rank)`; sum across sources gives a calibrated multi-source signal that often outperforms raw score aggregation.
2. **Temporal item features**
   - incident/post-incident/windowed popularity + trend deltas.
3. **User recency features**
   - days since last interaction, recent/long ratio.
4. **User-item affinity**
   - author, genre, language, **and publisher** overlap scores (publisher affinity is already computed in `features.py` but is missing from `_cross_features()` in `ranking.py` — see weakness #8).
5. **Frequency transforms**
   - `log1p` applied to all raw popularity counters and interaction counts before they enter the feature matrix (heavy-tailed distributions hurt tree splits; apply in `_item_popularity_features` and `_user_activity_features`).
6. **Rating-based features**
   - mean rating and rating count per `edition_id` (from `event_type=2` rows where `rating` is not null),
   - mean rating given by `user_id` (captures whether the user is a generous or harsh rater),
   - `rating - user_mean_rating` as a relative preference signal for the pair (see weakness #9).

## 5.2 High-impact features (P1)

1. **Interaction crosses**
   - source score x affinity,
   - source score x item trend,
   - user activity x popularity trend.
2. **Event-type behavior**
   - read/wishlist ratios per user and per item.
3. **Diversity-aware signals**
   - novelty score (penalize repeated author/series tendencies if over-dominant).

## 5.3 Advanced features (P2)

1. **Text semantics**
   - embeddings from `title + description`; prefer a lightweight multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2` from `sentence-transformers`) or TF-IDF + SVD as a faster fallback,
   - user text embedding as weighted average of interacted item embeddings,
   - cosine similarity between user text centroid and candidate item embedding as a ranker feature.
2. **CF embeddings from the existing SVD generator**
   - `SVDCollaborativeGenerator` already trains a truncated SVD and computes `U` (user factors) and `V` (item factors). Expose these matrices and add the dot product `U[user] · V[item]` as a ranker feature **without training a separate CF model**. This is a zero-cost way to add collaborative signal at P2.
   - For a stronger alternative: replace SVD with ALS from the [`implicit`](https://github.com/benfred/implicit) library (`implicit.als.AlternatingLeastSquares`). `implicit` uses WARP/BPR-style negative sampling optimised for top-K ranking, is significantly faster than `scipy.sparse.linalg.svds` on large matrices, and produces better recall.
3. **EASE (Embarrassingly Shallow Autoencoder)**
   - Closed-form item-item similarity model: `B = (X^T X + λI)^{-1} X^T X` with diagonal zeroed. Competitive with ALS on sparse data, trivial to implement, no iteration needed. Add EASE scores as a candidate source or ranker feature.

## 6. Model training plan

## 6.1 Phase 1 (fast uplift)

Priority bug fixes (do these before adding any new features):

1. **Fix early stopping**: pass a proper `eval_set` (e.g., last fold held out from the pseudo-label pool) to `model.fit()`. Without it `early_stopping_rounds` has no effect (weakness #6).
2. **Fix temporal leakage**: ensure all features for a given training row are computed only from interactions that occurred **before** the pseudo-incident start time of that row's fold (weakness #7). Implement a `_build_feature_matrix_for_ts(positives_up_to_T, ...)` helper that accepts a cutoff timestamp.
3. **Fix CatBoostRanker API**: switch from `CatBoostClassifier` + `Logloss` to `CatBoostRanker` + `YetiRank` objective. Wrap training data in `catboost.Pool(X, y, group_id=user_ids)`. Verify that `group_id` values are contiguous (sort by `user_id` before building the pool).
4. **Set eval metric to `NDCG:top=20`**: matches the competition cutoff exactly.

After fixing bugs:

- Add rolling pseudo-incident sampling (4–8 windows).
- Add per-source RRF rank + `log1p` temporal + affinity feature set (P0 from §5.1).
- Keep inference pipeline and fallback contract unchanged.

Expected result: significant and stable uplift over current CatBoost classifier baseline.

## 6.2 Phase 2 (ensemble)

- Train:
  - `YetiRank` CatBoost ranker (main),
  - Pointwise CatBoost classifier / `Logloss` (stability baseline),
  - deterministic RRF blend score.
- Learn blend weights on CV using **Optuna** (TPE sampler, `mean_ndcg@20` as objective, Dirichlet-simplex or softmax-constrained search space). Grid search is impractical beyond 3 models; Optuna converges in ~50–100 trials.

Expected result: more robust private leaderboard behavior.

## 6.3 Phase 3 (advanced ML)

- Add embedding features (text + collaborative).
- Optional: lightweight neural reranker for top-N candidate slice.
- Retain strict fallback path for production reliability.

## 7. Concrete repo changes

Primary files:

1. `src/competition/ranking.py`
   - refactor into modular parts:
     - dataset slicing / pseudo-label builder,
     - feature builder,
     - trainer(s),
     - inference + ensemble + fallback.
2. `configs/experiments/high_recall.yaml`
   - add ranker hyperparameters:
     - objective type,
     - rolling-window settings,
     - ensemble weights,
     - feature toggles.
3. (optional) new utility file:
   - `src/competition/ranker_features.py` for cleaner feature logic.
4. Rolling validation runner:
   - **Do not modify** `src/platform/pipeline/workflows/local_validation.py` or any other file under `src/platform/` — per `docs/baseline/ONBOARDING.md`, the platform zone is not meant to be changed.
   - Implement rolling CV as a standalone script or notebook in `notebooks/` (e.g. `notebooks/rolling_cv.ipynb`) that directly imports `src/competition/ranking.py` helpers. The platform's single-window `PseudoIncidentValidationWorkflow` can remain unchanged and still be used for quick smoke tests.

## 8. Experiment roadmap (ablation-first)

Run in strict order:

1. Baseline snapshot (current `high_recall`).
2. + bug fixes (early stopping `eval_set`, temporal leakage, `CatBoostRanker` + `YetiRank` + `NDCG:top=20`).
3. + rolling pseudo-label windows.
4. + P0 features.
5. + P1 crosses.
6. + ensemble of pairwise + pointwise + blend.
7. + advanced embeddings (if infra/runtime budget allows).

For each step, log:

- `mean_ndcg@20`,
- fold std,
- per-segment NDCG,
- inference latency and memory.

## 9. Risk management

1. **Overfitting to pseudo-label protocol**
   - mitigate with multiple rolling windows and fold averaging.
2. **Feature leakage**
   - ensure each split only uses data available before anchor time.
3. **Runtime blow-up**
   - cap candidate feature joins, use vectorized merges, cache heavy artifacts.
4. **Dependency fragility**
   - keep CatBoost-based primary path, treat advanced deep models as optional.

## 10. Definition of done

Ranker upgrade is accepted when all are true:

1. `mean_ndcg@20` improves consistently vs current baseline on rolling local validation.
2. Improvement is stable across user activity buckets (not only heavy users).
3. Inference still returns valid top-20 per user with seen-item filtering and fallback.
4. Runtime stays within practical limits for full pipeline execution.

---

This plan prioritizes high-ROI ranker improvements first (pairwise LTR + robust pseudo-labeling + stronger features), then adds ensemble and advanced representation learning for additional gains.
