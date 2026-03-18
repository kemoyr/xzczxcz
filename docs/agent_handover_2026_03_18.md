# Agent Handover Report (2026-03-18)

## 1) Контекст и цель

Цель итерации: поднять leaderboard score (с ~0.03-0.04) и устранить расхождение между offline сигналами и реальным test поведением, учитывая, что потери в окне инцидента, вероятно, **не случайные**.

Ключевая гипотеза: текущий train proxy в ранкере дает тривиальный валидационный сигнал (`bestTest=1`, `bestIteration=1`), из-за чего модель не переносится на реальный hidden test.

---

## 2) Какие файлы менялись

Измененные файлы на момент отчета:

- `src/competition/ranking.py`
- `validate_fast.py`
- `run_rank_only.py`
- `configs/experiments/high_recall.yaml`

---

## 3) Что сделано по коду

### 3.1 `validate_fast.py`

Сделано:

- Добавлен флаг `--eval-scope`:
  - `targets` (default) — оценка по `targets.csv`;
  - `incident_users` — диагностический режим.
- Валидационный scope выровнен под `targets.csv` (чтобы proxy был ближе к leaderboard-user set).
- Защита от пустого incident scope (выход с ошибкой).

Зачем:

- Ранее local NDCG считался по ~7569 октябрьским пользователям, что было несопоставимо с продовым scope (~3862).

---

### 3.2 `run_rank_only.py`

Сделано:

- Добавлен `argparse` (`--config`, `--candidates-path`, `--predictions-path`, `--submission-path`).
- Подключена загрузка конфига через `load_config(...)` вместо жестко зашитых параметров.
- По умолчанию скрипт берет `artifacts/candidates_v2.parquet`, если файл существует; иначе `artifacts/candidates.parquet`.

Зачем:

- Чтобы rank-only гарантированно работал на более сильном candidate pool (14 источников, включая post-incident генераторы) и был управляем через CLI.

---

### 3.3 `configs/experiments/high_recall.yaml`

Изменения:

- `per_generator_k` оставлен `300` (после отката с `400`, т.к. 400 сильно перегружал генерацию).
- Текущие параметры:
  - `item_cooccurrence`: `cooccurrence_days=120`, `seed_days=60`, `top_per_seed=250`
  - `svd_cf`: `n_factors=96`

Зачем:

- Держать баланс между recall и временем/памятью.

---

### 3.4 `src/competition/ranking.py`

Было сделано много итераций; ниже итог по важным изменениям.

#### A. Таймлайн-фикс и дополнительные фичи

- Явно задано `POST_INCIDENT_END_TS = 2025-12-01`.
- Пост-инцидентные affinity строятся в пределах `[INCIDENT_END, POST_INCIDENT_END)`.
- Добавлены user demographic фичи (`gender`, `age`, `age_bucket`, `age_missing`).

#### B. Псевдо-обучение под неслучайные потери

- Добавлены функции оценки bias:
  - по `event_type` (wishlist/read),
  - по часу суток,
  - по activity bucket пользователя.
- Псевдо-пары в rolling windows семплируются структурно (`_sample_structured_pseudo_pairs`) с учетом этих bias.

#### C. Оптимизация fold-size и обучения

- Добавлен тренировочный trimming кандидатов по `(user, source)`:
  - `_trim_fold_candidates_for_training(per_source_k=120)`.
- Негативы ограничиваются memory budget (`max_rows_per_fold - n_pos`).
- Fold-таблицы стабильно ограничены до ~300k строк.

#### D. Ансамбль на инференсе

- Используется per-user percentile нормализация компонент.
- Текущая (последняя) версия весов с более retrieval-heavy приоритетом:
  - ranker: `0.25`
  - classifier: `0.15`
  - src_sum: `0.45`
  - rrf: `0.15`

#### E. Анти-утечные правки

- В CatBoostRanker включено `use_best_model=False` (чтобы не shrink-иться к 1-2 итерациям).
- Для fold пост-окна из `post_df` удаляются пары из `pseudo_pos`, чтобы уменьшить прямую подсказку masked-пар.

#### F. Инжекция псевдо-позитивов

- Был эксперимент с `_INJECT_PSEUDO_POSITIVES=False` (радикально),
  но это привело к `Fold 1: only 0 positives`.
- После этого флаг возвращен в `True` для сохранения обучающего сигнала.

---

## 4) Что наблюдалось в логах (важно)

### Стабильная проблема

В нескольких прогонах `run_rank_only` на больших train folds наблюдалось:

- `bestTest = 1`
- `bestIteration = 1`
- ранее было `Shrink model to first 2 iterations` (когда `use_best_model` был включен).

Это означает, что вал-фолд для ranker остается слишком легким/утечным относительно реального test.

### Негативный эксперимент

После жесткого отключения инжекции псевдо-позитивов:

- `Fold 1: only 0 positives, skipping`
- запуск аварийно завершался.

То есть полностью убирать инжекцию в текущей схеме нельзя.

---

## 5) Текущее состояние (критично для следующего агента)

1. Последняя версия `ranking.py` уже содержит анти-утечные правки + структурный sampling + retrieval-heavy blend.
2. НЕТ подтвержденного успешного финального прогона после последних правок с валидным `submission.csv` и анализом качества.
3. Ветка находится в dirty-state (см. `git status`), коммит не создавался.

---

## 6) Главные оставшиеся проблемы

### Problem 1: Proxy fold все еще слишком тривиален

Даже после части правок ranker показывает `bestIteration` очень близко к началу.

Что это ломает:

- Модель не учится реально ранжировать инцидентные “иголки”, а переобучается на proxy-паттерн.

### Problem 2: Непрозрачная дельта “код -> сабмит”

Много перезапусков и остановок; нет чистой таблицы:

- hash/версия `ranking.py`
- какой файл кандидатов использовался
- время запуска
- итоговый private/public score.

### Problem 3: Перегрузка процессами

Были параллельные процессы (`run_rank_only`, `run_full_v2`, `validate_fast`) из разных терминалов, что искажало время и могло мешать стабильности.

---

## 7) Что делать следующему агенту (пошагово)

### Шаг 0 — Санитация окружения

1. Убедиться, что нет активных `run_rank_only.py`, `run_full_v2.py`, `validate_fast.py`.
2. Запускать только ОДИН тяжелый процесс за раз.

### Шаг 1 — Зафиксировать baseline для текущего кода

1. `uv run python run_rank_only.py`
2. Убедиться по логу, что есть:
   - все 6 folds,
   - нет `only 0 positives`,
   - нет runtime fallback из-за exception.
3. Проверить контракт:
   - `3862 * 20 = 77240` строк в `submission.csv`,
   - no duplicate `(user_id, edition_id)`.

### Шаг 2 — Если снова `bestIteration=1`, изменить train proxy, не инференс blend

Приоритетные варианты:

1. Разделить train/infer feature-space:
   - train без `post_*` affinity целиком,
   - infer с `post_*` оставить.
2. Уменьшить overlap между псевдо-окном и сигналами, которые его же объясняют:
   - сдвиг/дропаут части affinity блоков на train.
3. Ослабить валидационный fold:
   - использовать fold 2 как val, fold 3-6 train (проверить чувствительность).

### Шаг 3 — Отладка через быстрый A/B на rank-only

Сравнивать только ranker-варианты на одном и том же `candidates_v2.parquet`.

---

## 8) Быстрые команды для нового агента

Проверка процессов:

```powershell
Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -eq 'python.exe' -or $_.Name -eq 'uv.exe') -and
  ($_.CommandLine -match 'run_rank_only.py|run_full_v2.py|validate_fast.py')
} | Select-Object ProcessId,ParentProcessId,Name,CommandLine
```

Rank-only:

```bash
uv run python run_rank_only.py
```

Validate-fast (targets scope):

```bash
uv run python validate_fast.py --eval-scope targets
```

---

## 9) Честная оценка текущего прогресса

- Улучшена инженерная часть (scope валидатора, управляемость run_rank_only, контроль таймлайна, процессный хаос).
- Сделаны сильные гипотезы про неслучайные потери.
- Но ключевая задача (устойчивый резкий рост leaderboard score) пока НЕ решена.

То есть проект переведен в более управляемое состояние, но требуется еще 1-2 целевых итерации именно по train proxy/ranker transfer, чтобы выйти из зоны `bestIteration≈1` и получить измеримый прирост на скрытом тесте.

