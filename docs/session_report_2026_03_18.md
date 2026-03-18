# Session Report — 2026-03-18

## Что было сделано

### 1. Аудит кодовой базы и плана улучшений

Полностью прочитаны:
- Постановка задачи и описание данных (`docs/task/`)
- Онбординг и архитектура пайплайна (`docs/baseline/ONBOARDING.md`)
- План устранения разрыва val/test (`docs/test_val_gap_analysis_and_fix_plan.md`)
- `src/competition/features.py`, все генераторы, `src/competition/ranking.py`
- `validate_fast.py`, `run_rank_only.py`, `configs/experiments/high_recall.yaml`

Установлено, что планы P0–P3 уже были реализованы в коде, но в `validate_fast.py`
оставались два структурных бага, из-за которых нельзя было корректно измерить эффект улучшений.

---

### 2. Исправления в `validate_fast.py`

#### Баг 1 — Неверный скоп маскирования

**Было:** `clean_df.merge(masked_pairs)` — удалял ВСЕ строки для пары
`(user_id, edition_id)` из всего датасета, включая чистую историю до инцидента.

**Проблема:** в реальном соревновании теряются только события из окна инцидента
(октябрь). Пре-инцидентная история (июнь–сентябрь) остаётся видимой. Старый код
делал валидацию искусственно сложнее и некорректной.

**Исправлено:** маскировка теперь применяется только к событиям внутри окна инцидента.
Пре- и пост-инцидентные данные остаются нетронутыми.

#### Баг 2 — Псевдо-инцидент не включал ноябрьские данные

**Было:** псевдо-инцидент строился из первых 150 дней данных (примерно июнь–октябрь),
поэтому `val_dataset` никогда не содержал ноябрьских взаимодействий. Это означало:
- `post_incident_*` генераторы всегда возвращали пустые кандидаты при валидации
- Post-incident affinity фичи в ранкере всегда равнялись 0.0
- Улучшения P1 и P2 нельзя было измерить

**Исправлено:** `validate_fast.py` теперь использует реальные временны́е метки инцидента
(`2025-10-01 → 2025-11-01`) и включает ноябрьские данные в `observed`,
что позволяет всем генераторам корректно работать при валидации.

#### Дополнительно

Исправлен `UnicodeEncodeError` в логгере: символ `≤` в названиях бакетов активности
заменён на `<=` (cp1251 не поддерживает `≤`).

---

### 3. Новый скрипт `run_full_v2.py`

Написан standalone-скрипт для генерации свежего сабмита без перезаписи
существующих артефактов:

| Файл | Назначение |
|---|---|
| `artifacts/candidates_v2.parquet` | Новый пул кандидатов (все 14 генераторов) |
| `artifacts/predictions_v2.parquet` | Ранжированные предсказания |
| `artifacts/submission_v2.csv` | Готовый сабмит для загрузки |

---

### 4. Обновление документации

`docs/test_val_gap_analysis_and_fix_plan.md` обновлён: каждый пункт плана
помечен статусом (✅ DONE / ⬜ TODO), описания исправленных багов актуализированы.

---

## Результаты валидации

### Запуск 1 — `--reuse-existing-candidates` (старые кандидаты)

| Метрика | Значение |
|---|---|
| mean NDCG@20 | 0.001895 |
| pair_candidate_recall@M | **0.000000** |
| Пользователи | 7 569 |

**Причина нулевого recall:** старый `candidates.parquet` был сгенерирован с октябрьскими
взаимодействиями в `seen_positive_df`, поэтому октябрьские издания исключались из
кандидатов как «уже просмотренные». Кандидаты несовместимы с новым setup'ом валидации.

### Запуск 2 — полная регенерация кандидатов (правильная валидация)

| Метрика | Значение |
|---|---|
| **mean NDCG@20** | **0.009218** |
| pair_candidate_recall@M | **0.4192** |
| mean_user_candidate_recall@M | **0.4530** |
| Пользователи | 7 569 |
| Кандидатов | 24 952 738 |
| Источников | 14 (включая все `post_incident_*`) |

**По бакетам активности:**

| Бакет | NDCG@20 |
|---|---|
| cold (<=5) | 0.001613 |
| light (6-20) | 0.009912 |
| medium (21-50) | 0.011803 |
| heavy (>50) | 0.011035 |

### Интерпретация

- **Recall 41.9%** — основной результат. Генераторы находят в пуле ~4 200 из 10 036
  замаскированных пар. Все 14 источников работают, включая `post_incident_*`.
- **NDCG 0.0092 vs платформенный baseline 0.0127** — разница объяснима: наша
  валидация усредняет по 7 569 пользователям (все октябрьские), из которых многие
  «холодные» с нулевым NDCG. Платформа оценивает только ~3 862 конкретных target users.
- Паттерн по бакетам правильный: активные пользователи ≈ лучше (кроме very heavy,
  что типично — у них уже видна большая история, реально потерянное сложнее).
- `q25=q50=q75=0` нормально при ~1.33 замаскированной паре на пользователя в среднем
  и 20 слотах из 25M кандидатов.

---

## Текущее состояние реализации плана

| Пункт | Статус | Что сделано |
|---|---|---|
| P0 — Честная валидация | ✅ | `validate_fast.py` полностью переписан: маска только на инцидент-окно, ноябрь включён, recall репортится |
| P1 — Post-incident генераторы | ✅ | 4 генератора: author/genre/language/publisher profile |
| P2 — Post-incident фичи ранкера | ✅ | `*_affinity_post` + `item_pop_incident_x_*_post` кросс-фичи |
| P3 — Ширина кандидатов | ✅ | `per_generator_k=300`, 14 генераторов |
| P4 — Структура псевдо-обучения | ⬜ | Не реализовано (после P0/P1 по плану) |

---

## Как формируется финальный сабмит

### Схема пайплайна

```
data/ (CSV)
  │
  ▼
build_features_frame()          ← features.py
  │  34 типа фич: popularity, user profiles (full/recent/incident/post_incident)
  │
  ▼
run_generators()                ← generators/ (14 генераторов)
  │  candidates_v2.parquet: 13.98M строк, 3 862 пользователя
  │
  ├── global_popularity          (холодный старт)
  ├── global_temporal_popularity (взвешенная по окнам)
  ├── user_genre                 (полная история)
  ├── user_author                (полная история)
  ├── user_language              (полная история)
  ├── user_publisher             (полная история)
  ├── user_genre_recent          (последние 30 дней)
  ├── user_author_recent         (последние 30 дней)
  ├── post_incident_genre_profile   ← ноябрьский профиль
  ├── post_incident_author_profile  ← ноябрьский профиль
  ├── post_incident_language_profile ← ноябрьский профиль
  ├── post_incident_publisher_profile ← ноябрьский профиль
  ├── item_cooccurrence          (коллаборативный)
  └── svd_cf                     (SVD матричная факторизация)
  │
  ▼
rank_predictions()              ← ranking.py
  │
  ├── _generate_rolling_training_data()
  │     6 скользящих окон по 14 дней (псевдо-инциденты из чистой истории)
  │     Fold 1 → eval set, Folds 2–6 → train set
  │
  ├── _build_feature_matrix()
  │     src_* (source scores + RRF rank)
  │     user activity (w7/w14/w30/w90, reads/wishlists, log1p)
  │     item popularity (w7/w14/w30/w90/incident, trend, log1p)
  │     item catalogue (year, age_restriction, language_id)
  │     rating features (item_mean_rating, user_mean_rating, relative)
  │     cross features: user_*_affinity + user_*_affinity_post (ноябрь)
  │     interaction: item_pop_incident × user_*_affinity_post
  │
  ├── CatBoostRanker (YetiRankPairwise, 400 iter)
  ├── CatBoostClassifier (Logloss, 300 iter, early stopping)
  └── Ensemble: 70% YetiRank + 20% Classifier + 10% RRF
  │
  ▼
submission_v2.csv               ← 3 862 пользователя × 20 позиций
  user_id, edition_id, rank
```

### Запуск

```bash
# Полный пересчёт (кандидаты + ранк + сабмит) в новые файлы:
uv run python run_full_v2.py

# Только ранк на уже готовых candidates_v2.parquet:
uv run python run_rank_only.py
# (нужно вручную заменить candidates_v2.parquet → candidates.parquet,
#  или отредактировать путь в run_rank_only.py)

# Валидация (честная, без leakage):
uv run python validate_fast.py
```

### Ключевые конфиги

| Параметр | Значение | Файл |
|---|---|---|
| `per_generator_k` | 300 | `configs/experiments/high_recall.yaml` |
| `recent_days` | 30 | `configs/base.yaml` → `high_recall.yaml` |
| `n_rolling_windows` | 6 | `ranking.py` (`_N_ROLLING_WINDOWS`) |
| `catboost_iterations` | 400 | `ranking.py` (`CatBoostRanker`) |
| `ensemble` | 70/20/10 | `ranking.py` (YetiRank/Clf/RRF) |
| Инцидент | 2025-10-01 → 2025-11-01 | `features.py`, `ranking.py` |

---

## Следующие шаги (если нужно улучшать дальше)

1. **Поднять recall** (сейчас 41.9% на валидации):
   - Увеличить `per_generator_k` до 500 для сильных персонализированных генераторов
   - Поднять `n_factors` SVD с 64 до 128–256
   - Расширить `cooccurrence_days` с 90 до 120–150

2. **P4 — Улучшить структуру псевдо-обучения:**
   - Асимметричное маскирование по типам событий (wishlist vs read)
   - Кластеризация по времени внутри инцидента
   - Маскирование по сегментам пользователей

3. **Добавить user-level признаки** из `users.csv` (age, gender) в feature matrix ранкера
