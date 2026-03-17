# Candidate Generation Analysis and Improvement Plan

## 1) Task Context and Objective

The competition requires recovering lost positive `(user_id, edition_id)` interactions and ranking top-20 per target user with `NDCG@20`.

For this metric, candidate generation quality is critical:
- if a true lost item is not in candidates, ranker cannot recover it;
- higher recall in top candidate pool usually translates into better final `NDCG@20`.

Current local baseline validation is very weak (`mean_ndcg@20 ~= 0.00378`, quartiles at 0), which indicates poor recall/hit-rate before ranking.

## 2) Data/EDA Signals Relevant to Candidate Generation

From `notebooks/EDA.ipynb` and task docs:

- Dataset scale:
  - `interactions`: 443,278 events (all are positive types 1/2 in this setup)
  - unique positive pairs: 442,870
  - target users: 3,862
  - catalog editions: 460,599 (much larger than interacted items 126,002)
- Time windows matter by task design:
  - incident: `2025-10-01` to `2025-11-01`
  - post-incident (fully observed): `2025-11-01` to `2025-12-01`
- Event type mix is imbalanced (`read` ~58.3%, `wishlist` ~41.7%).
- Strong long-tail on items:
  - top 10 editions explain only ~0.95% of interactions
  - top 100: ~5.22%
  - top 1000: ~21.19%
  - global popularity alone cannot recover enough personalized lost pairs.
- Target users are not cold-start in strict sense here (`target_users_without_positive_history = 0`), but many users can still have short/noisy recent history and need robust backoff.
- Preferences by author/genre are meaningful, but not sufficient without temporal and collaborative signals.

## 3) Current Generators: What They Do

Current sources in `src/competition/generators/`:

1. `global_popularity`
   - Uses only `edition_popularity_all`.
   - Broadcasts same top-k list to all users.

2. `user_genre`
   - Uses `user_genre_profile` (full history normalized user genre weights).
   - Expands each preferred genre to top editions in that genre sorted by global popularity.
   - Score: `sum(user_genre_weight * (edition_pop + genre_smoothing))`.

3. `user_author`
   - Uses `user_author_profile` similarly.
   - Expands preferred authors to popular editions by author.
   - Score: `sum(user_author_weight * (edition_pop + author_smoothing))`.

## 4) Main Gaps in Current Generation Logic

### 4.1 Unused Feature Signal

`src/competition/features.py` already computes many useful signals that generators currently ignore:
- recency popularity: `edition_popularity_recent`, `edition_popularity_w7/w14/w30/w90`
- incident/post-incident popularity
- short/long trend: `edition_popularity_trend_short_long`
- separate read/wishlist/weighted popularity
- recent user profiles:
  - `user_genre_profile_recent`
  - `user_author_profile_recent`
  - `user_language_profile(_recent)`
  - `user_publisher_profile(_recent)`

Right now candidate generation uses mainly:
- `edition_popularity_all`
- `user_genre_profile`
- `user_author_profile`

So most temporal and context signals are not converted into recall.

### 4.2 No Collaborative Retrieval

There is no item-item or user-user co-occurrence generator.
This is a major recall gap for recovery tasks where lost interactions are often close to observed behavior of similar users/items.

### 4.3 Over-Reliance on Global Popularity Prior

Both personalized generators multiply by `edition_popularity_all`.
This pushes recommendations toward head items and can suppress long-tail yet relevant candidates.

### 4.4 Temporal Mismatch with Incident Objective

Task is explicitly incident-window based, but generators are mostly all-time.
No explicit preference toward:
- recent momentum;
- incident/post-incident consistency;
- user's recent taste drift.

### 4.5 Candidate Efficiency Issues

- Seen positives are filtered only in ranking, not at generator stage.
  - This wastes per-generator budget with known items.
- Python loops over users and `iterrows` make generation slower and harder to scale to richer sources.
- `top_per_author/top_per_genre = max(k*5, 200)` can still overfocus on frequent entities and duplicates between sources.

## 5) Recommended Generator Improvements (Priority Order)

## P0: High-Impact, Low Risk

1. Add temporal popularity generators
   - New sources:
     - `global_popularity_recent` (`w14` or `w30`)
     - `global_popularity_incident`
     - `global_popularity_post_incident`
     - `global_popularity_trend` (from short/long trend feature)
   - Why: direct alignment with incident dynamics and recency.

2. Add recent-profile variants of existing personalized generators
   - `user_genre_recent`, `user_author_recent` using `*_profile_recent`.
   - Keep current full-history variants too, then blend.
   - Why: captures short-term preference shift.

3. Add language/publisher-based generators
   - `user_language`, `user_publisher` (+ optional recent variants).
   - Why: these profiles already exist and can boost recall for sparse author/genre mappings.

4. Filter seen positives inside generators before truncating top-k
   - Prevent budget waste and increase unique unseen candidates per user.

## P1: Medium Complexity, Strong Recall Potential

5. Add item-to-item co-occurrence generator (collaborative)
   - Build co-consumption links from user histories (preferably recent window weighted).
   - For each user, seed from their recent interacted items and retrieve neighbors.
   - Score with recency-weighted co-count + item popularity prior.

6. Add user-nearest-neighbors retrieval
   - Lightweight: overlap/Jaccard over recent positives.
   - Candidate pool = items from similar users not seen by target user.

7. Source quotas per user
   - Instead of equal `per_generator_k` behavior, enforce per-source quotas (e.g. 30% personalized recent, 30% collaborative, 20% long-term profiles, 20% global fallback).
   - Reduces over-dominance of head popularity.

## P2: Optimization and Calibration

8. Score calibration per source
   - Normalize source score distributions (e.g. z-score or min-max per user/source) before ranking blend.
   - Prevent one source from dominating due to arbitrary scale.

9. Diversity constraints in candidate assembly
   - Soft caps per author/genre in candidate list to avoid near-duplicate items.

10. Vectorize generator internals
   - Replace nested Python loops with joins/groupbys to accelerate experimentation.

## P3: Advanced & Cutting-Edge Methods (High Complexity, Maximum Recall)

11. Robust Matrix Factorization for Implicit Feedback
    - Replace basic `scipy.sparse.linalg.svds` with Alternating Least Squares (ALS) or Bayesian Personalized Ranking (BPR) using the `implicit` or `LightFM` libraries.
    - These methods are state-of-the-art for implicit feedback and handle unobserved interactions naturally.

12. Approximate Nearest Neighbors (ANN) for Retrieval
    - Use `Faiss` or `nmslib` to scale up similarity searches (e.g., user-KNN, item-KNN, or embedding dot-products).
    - Avoids exhaustive scoring over all items, allowing retrieval over much larger candidate pools in milliseconds.

13. Two-Tower Neural Retrieval (DSSM)
    - Train deep learning Two-Tower models (e.g., using PyTorch) where one tower encodes the user (profile, recent history) and the other encodes the item (metadata, popularity).
    - Can leverage rich textual embeddings (like item `description` or `title`) and temporal features.

14. Graph Neural Networks (LightGCN)
    - High-order collaborative filtering using graph embeddings (e.g., `PyTorch Geometric`). Excellent for propagating signal in sparse, long-tail data scenarios.

## 6) Concrete Implementation Blueprint

Suggested new files in `src/competition/generators/`:

- `global_temporal_popularity.py`
- `user_genre_recent.py`
- `user_author_recent.py`
- `user_language.py`
- `user_publisher.py`
- `item_cooccurrence.py`
- `user_knn.py` (optional second collaborative source)

Update:
- `registry.py` to register new generators.
- experiment config (`configs/experiments/high_recall.yaml`) with expanded generator list and tuned source weights.

## 7) Suggested Scoring Formulas

Practical scoring templates:

1. Temporal popularity generator:
`score = a1*pop_w14 + a2*pop_w30 + a3*pop_post_incident + a4*trend`

2. Personalized profile generator:
`score = affinity(user, entity) * log1p(pop_w30) * freshness_boost`

3. Co-occurrence generator:
`score = sum_over_seed_items( recency_weight(seed) * co_count(seed, candidate) )`

Use simple defaults first, then tune coefficients on local validation.

## 8) Experiment Plan (Ablation-First)

Recommended sequence:

1. Baseline reference (current config).
2. + temporal popularity sources (P0-1).
3. + recent profile generators (P0-2).
4. + language/publisher generators (P0-3).
5. + seen-filter at generation stage (P0-4).
6. + item co-occurrence (P1-5).
7. Tune source weights and per-generator budget.
8. Swap naive SVD for `implicit` ALS/BPR (P3-11).
9. Integrate Two-Tower or Graph-based models for advanced recall (P3-13, P3-14).

Track per run:
- local `mean_ndcg@20`
- candidate coverage proxy on pseudo-incident positives
- unique candidates per user after seen-filter
- overlap matrix between sources (to detect redundancy).

## 9) Expected Outcome

Given current very low validation score and observed gaps, the strongest near-term gain should come from:

1. temporalizing popularity retrieval,
2. adding recent personalized generators,
3. introducing at least one collaborative source.

These changes directly target the weak points of current generation (all-time bias, low personalization depth, no collaborative recall) and should materially improve candidate recall, which is the main bottleneck for final `NDCG@20`.

