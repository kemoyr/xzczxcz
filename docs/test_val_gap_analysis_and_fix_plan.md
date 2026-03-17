# Why `validate_fast` Shows ~0.49 While Test Is ~0.029

## Executive Diagnosis

The gap is real and has a clear technical explanation:

1. `validate_fast.py` is a **leaky proxy** (it injects masked ground-truth pairs into candidates).
2. It also reuses `artifacts/candidates.parquet` built on a **different dataset scope/time context** than the pseudo-validation slice.
3. The production pipeline (`baseline validate` and leaderboard) evaluates true end-to-end retrieval, where candidate recall is the bottleneck.

So ~0.49 reflects "ranker on a near-oracle candidate pool", while 0.029 reflects real retrieval difficulty.

---

## Evidence Collected

### A) Data timeline (confirmed from EDA)

- `time_max = 2025-11-30 23:59:35` (there is a full month after incident).
- Incident: `2025-10-01 ... 2025-11-01`.
- Post-incident: `2025-11-01 ... 2025-12-01`.
- Row counts from notebook/scripts:
  - clean last-30d before incident: `61,239`
  - incident: `50,200`
  - post-incident: `64,613`

### B) Validation discrepancy in logs

- Platform local validation logs show:
  - `mean_ndcg@20 = 0.012761...` (run logs on 2026-03-17).
- Public test score is `~0.029`.
- `validate_fast.py` reports `~0.49` because of methodological leakage.

### C) Why November is useful (and how)

For target users:

- `users_with_post = 3110 / 3862`
- `target users with post but no incident activity = 186`

This means November is valuable for recovering user profile when October is sparse/missing.

Important correction from measured overlap:

- Per-user direct `edition_id` overlap between October and November is near zero:
  - `mean overlap ~ 0.001`
  - `median overlap = 0`

Therefore, November should be used primarily as a **profile signal** (author/genre/language/publisher and collaborative structure), not as a direct "same item must be lost in October" rule.

---

## Root Causes In Code

## 1) `validate_fast.py` injects pseudo-ground-truth into candidates

`validate_fast.py` explicitly appends masked pairs into candidate pool:

```python
missing_masked["score"] = 0.0
missing_masked["source"] = first_src
val_candidates = pd.concat([val_candidates, missing_masked], ignore_index=True)
```

This guarantees many positives are present in candidate set even if generators could never retrieve them.

## 2) `validate_fast.py` uses stale/full-run candidates

`validate_fast.py` reads `artifacts/candidates.parquet` generated on another dataset context, then only filters users.  
It does not regenerate candidates from the pseudo-observed dataset, so it is not equivalent to platform validation.

## 3) Training objective differs from anomaly mechanism

In `ranking.py`, pseudo-training labels are built from rolling clean windows and injected into fold candidates (`_inject_pseudo_positives`).  
This helps model training stability, but can overestimate ranking quality versus real incident losses where generator recall is the hard part.

## 4) November is underused in personalized retrieval

Current features include `edition_popularity_post_incident`, but there are no dedicated personalized generators that explicitly seed from post-incident user behavior.

---

## Fix Plan (Prioritized)

## P0 — Replace `validate_fast` as KPI

Keep `validate_fast.py` only as debugging tool.  
Primary metric must come from platform workflow (`baseline validate`) or a strict replica without leakage.

### Required validation protocol

1. Build pseudo-observed dataset by masking pairs.
2. Recompute features from masked dataset.
3. Re-run all generators on that dataset.
4. Rank.
5. Report both:
   - `candidate_recall@M` (M = merged candidate pool per user)
   - `ndcg@20`

Without candidate recall, ranker tuning is mostly noise.

## P1 — Add November-conditioned personalized generators

Given low direct item overlap, focus on **profile transfer**:

- Build user profile on post-incident month:
  - author, genre, language, publisher distributions.
- Retrieve incident-popular items matching that profile.
- Exclude seen items via `seen_positive_df`.

Concrete additions:

- `post_incident_author_profile_generator`
- `post_incident_genre_profile_generator`
- optional `post_incident_language/publisher` variants

These should be implemented similarly to `user_author_recent` / `user_genre_recent`, but with an explicit November slice and incident-prior scoring.

## P2 — Add November-aware ranker features

Extend feature/ranker stack with:

- `user_author_affinity_post`
- `user_genre_affinity_post`
- `user_language_affinity_post`
- `user_publisher_affinity_post`
- `item_pop_incident_x_user_affinity_post` interaction features

This lets CatBoost learn "November preference -> likely October loss domain" even when exact items differ.

## P3 — Increase generator breadth for sparse users

Because many users have thin incident history, improve recall before reranking:

- raise `per_generator_k` for the strongest personalized generators;
- increase diversity budget in co-occurrence/SVD candidates;
- monitor candidate recall impact per source.

## P4 — Make pseudo-training closer to incident structure

Current pseudo-labeling is temporal and useful, but anomaly may be structurally non-uniform.  
Add additional masking modes in training experiments:

- user-segment masking,
- event-type-asymmetric masking,
- time-cluster masking.

Do this only after P0/P1 so experiments are measured on realistic validation.

---

## Practical Next Iteration

1. Freeze KPI to platform validation (0.0127 baseline locally).
2. Implement 1-2 November-profile generators (author + genre first).
3. Add post-incident affinity cross-features in ranker.
4. Re-run:
   - candidate source contribution,
   - candidate recall@M,
   - ndcg@20.

If candidate recall does not move, do not tune CatBoost further.

---

## Bottom Line

`validate_fast ~0.49` is inflated by validation leakage and candidate reuse mismatch.  
`test ~0.029` is the realistic signal.

The extra month after anomaly is valuable, but mostly as **user preference transfer signal** (profile/collaborative), not direct item carry-over.  
The correct strategy is to improve candidate recall with November-conditioned personalized generators, then let the ranker exploit those signals.
