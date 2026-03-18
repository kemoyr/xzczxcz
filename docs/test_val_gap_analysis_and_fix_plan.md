# Why `validate_fast` Showed ~0.49 While Test Was ~0.029

> **Status (2026-03-18):** All P0–P3 items are implemented and the two validate_fast bugs
> listed below are fixed. The old inflated score no longer applies.

## Executive Diagnosis (Historical)

The gap was real and had a clear technical explanation:

1. `validate_fast.py` **injected masked ground-truth pairs into candidates** (direct leakage).
2. It **reused `artifacts/candidates.parquet`** built on a different dataset scope/time context.
3. The production pipeline evaluates true end-to-end retrieval, where candidate recall is the bottleneck.

So ~0.49 reflected "ranker on a near-oracle candidate pool", while 0.029 reflected real retrieval difficulty.

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

### 1) [FIXED] `validate_fast.py` injected pseudo-ground-truth into candidates

The old code appended masked pairs back into the candidate pool. **Fixed**: `validate_fast.py`
now runs generators strictly from the masked observation without any injection.

### 2) [FIXED] `validate_fast.py` used stale/full-run candidates

The old code read `artifacts/candidates.parquet` and filtered users. **Fixed**: `validate_fast.py`
now re-runs all generators from the masked dataset by default (`--reuse-existing-candidates`
flag exists for debugging only).

### 3) [FIXED] `validate_fast.py` masked pairs from entire history, not just incident window

The old code did `clean_df.merge(masked_pairs)`, removing ALL occurrences of a masked
`(user_id, edition_id)` pair including pre-incident clean history. In the actual competition,
only the incident-window events are lost; pre-incident interactions remain visible.
**Fixed**: masking now only removes events within the October incident window; pre-incident
and post-incident (November) data remain fully visible.

### 4) [FIXED] November not included in validation dataset

The old code used only the first 150 days of data as "clean history", so November was never
present in `val_dataset`. Post-incident generators (`post_incident_author_profile` etc.) always
returned empty candidates during validation.
**Fixed**: `validate_fast.py` now uses the actual October incident timestamps and includes
the full November dataset so all post-incident generators are properly exercised.

### 5) Training objective differs from anomaly mechanism

In `ranking.py`, pseudo-training labels are built from rolling clean windows and injected into
fold candidates (`_inject_pseudo_positives`). This helps model training stability, but can
overestimate ranking quality versus real incident losses where generator recall is the hard part.
(P4 addresses this — not yet implemented.)

### 6) [FIXED] November was underused in personalized retrieval

Previously, features included `edition_popularity_post_incident` but there were no personalized
generators seeded from November behavior. **Fixed**: four post-incident profile generators
(`post_incident_author_profile`, `post_incident_genre_profile`, `post_incident_language_profile`,
`post_incident_publisher_profile`) are now implemented, registered, and active in `high_recall.yaml`.

---

## Fix Plan (Prioritized)

### P0 — Replace `validate_fast` as KPI ✅ DONE

`validate_fast.py` is now a strict, non-leaky validation tool:

1. Masks 20% of October (incident-window) positive pairs as pseudo-poteryashki.
2. Keeps pre-incident clean history and November post-incident data fully visible.
3. Rebuilds features from the masked dataset (no stale candidates).
4. Re-runs all generators (default) or reuses with `--reuse-existing-candidates`.
5. Reports:
   - `pair_candidate_recall@M`
   - `mean_user_candidate_recall@M`
   - `ndcg@20`
   - per-activity-bucket NDCG breakdown

### P1 — Add November-conditioned personalized generators ✅ DONE

Four generators are implemented and active (`high_recall.yaml`):

| Generator | Feature consumed | Source name |
|---|---|---|
| `PostIncidentAuthorProfileGenerator` | `user_author_profile_post_incident` | `post_incident_author_profile` |
| `PostIncidentGenreProfileGenerator` | `user_genre_profile_post_incident` | `post_incident_genre_profile` |
| `PostIncidentLanguageProfileGenerator` | `user_language_profile_post_incident` | `post_incident_language_profile` |
| `PostIncidentPublisherProfileGenerator` | `user_publisher_profile_post_incident` | `post_incident_publisher_profile` |

Each generator retrieves incident-popular items matching the user's November preference profile
and correctly excludes seen pairs via `seen_positive_df`.

The corresponding feature blocks (`user_*_profile_post_incident`) are built in `features.py`
from the November slice (`INCIDENT_END_TS … POST_INCIDENT_END_TS`).

> **Known limitation**: During rolling pseudo-training inside `CatBoostRanker`, the
> post-incident feature blocks are always 0 (training folds predate November). The model
> cannot learn to use post-affinity features from training signal; they are informative only
> at inference time. This is an inherent structural limitation of the timeline.

### P2 — Add November-aware ranker features ✅ DONE

`ranking.py → _cross_features` computes affinity blocks with suffix `_post` from
`positives[event_ts >= INCIDENT_END_TS]`:

- `user_author_affinity_post`
- `user_language_affinity_post`
- `user_genre_affinity_post`
- `user_publisher_affinity_post`
- `item_pop_incident_x_user_author_affinity_post`
- `item_pop_incident_x_user_genre_affinity_post`
- `item_pop_incident_x_user_language_affinity_post`
- `item_pop_incident_x_user_publisher_affinity_post`

Cross features (`src_max_x_author_affinity_post`, `src_max_x_genre_affinity_post`) are also
included in the feature matrix.

### P3 — Increase generator breadth for sparse users ✅ DONE

`configs/experiments/high_recall.yaml`: `per_generator_k: 300`.
14 generator sources active, covering global/personalized/collaborative signals.

### P4 — Make pseudo-training closer to incident structure ⬜ TODO

Current pseudo-labeling uses rolling clean-history windows (uniform random masking).
Structural improvements for later:

- user-segment masking
- event-type-asymmetric masking
- time-cluster masking

**Do not start P4 until candidate_recall@M is confirmed to have improved with P1/P3.**

---

## Practical Next Iteration

1. ✅ Freeze KPI: `validate_fast.py` is now the correct non-leaky metric.
2. ✅ November-profile generators (all 4: author, genre, language, publisher).
3. ✅ Post-incident affinity cross-features in ranker.
4. **Run `uv run python validate_fast.py`** and confirm:
   - `pair_candidate_recall@M` improves vs. baseline (target: > 0.25).
   - `ndcg@20` improves (current platform baseline: ~0.0127).
5. If candidate recall did not move → add more breadth (bigger SVD, longer co-occurrence window).
6. If candidate recall improved but NDCG did not → investigate ranker feature quality.

Note: validate_fast.py NDCG will now be lower than the old leaky ~0.49 and closer to the
realistic platform score (~0.012–0.03 range), because November is the harder scenario
(direct item overlap between October and November is near zero).

---

## Bottom Line

The old `validate_fast ~0.49` was inflated by validation leakage and candidate reuse mismatch.
`test ~0.029` was (and remains) the realistic signal.

The extra month after anomaly is valuable, but mostly as **user preference transfer signal**
(profile/collaborative), not direct item carry-over. This is now properly captured by:

- Post-incident profile generators (P1) that retrieve October-popular items matching
  the user's November author/genre/language/publisher preferences.
- Post-incident affinity cross-features (P2) that let CatBoost exploit the November
  preference signal at inference time.
- A correct, non-leaky `validate_fast.py` (P0) that now uses the actual October incident
  window and includes November data, so all generators are properly tested.

**The next action is to run `validate_fast.py` and measure whether candidate recall improved.**
