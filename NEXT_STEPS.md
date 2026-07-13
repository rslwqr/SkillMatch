# Следующие шаги для запуска SkillMatch

В репозитории уже находятся запросы, экспертная разметка и скрипты поиска.
Большие артефакты не хранятся в GitHub: их нужно скачать отдельно и положить
в проект без изменения имён.

## 1. Подготовить окружение

Из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Добавить готовые артефакты

```text
data/processed/corpus_500k.parquet
data/embeddings/e5_small_v2_embeddings.npy
```

Корпус и эмбеддинги должны быть одной версии. Нельзя отдельно сортировать или
пересобирать корпус: строка с индексом `i` в матрице эмбеддингов соответствует
строке `i` в Parquet-файле.

Перед долгими запусками рекомендуется проверить:

- в корпусе 500 000 строк;
- в массиве эмбеддингов 500 000 векторов;
- тип эмбеддингов — `float32`;
- обязательные столбцы корпуса и запросов доступны скриптам.

## 3. Запустить BM25

```bash
python3 src/bm25_search.py
```

Будут созданы локальный SQLite-индекс и файл:

```text
results/bm25_top20.csv
```

## 4. Запустить точный Dense Search

```bash
python3 src/dense_search.py
```

Если файл `e5_small_v2_embeddings.npy` лежит по указанному пути, скрипт
использует его и не генерирует эмбеддинги повторно. Не передавайте параметр
`--rebuild-embeddings`.

Будут созданы:

```text
data/embeddings/e5_small_v2_flat.index
results/dense_flat_top20.csv
results/dense_flat_metrics.json
```

При первом запуске Sentence Transformers может скачать модель
`intfloat/e5-small-v2`.

## 5. Запустить HNSW

Dense Search необходимо выполнить первым, потому что HNSW использует точный
Flat-индекс как эталон для расчёта ANN Recall@10.

```bash
python3 src/hnsw_search.py
```

По умолчанию проверяются `efSearch = 32, 64, 128`. Основные результаты:

```text
results/hnsw/
results/hnsw_metrics.csv
results/hnsw_metrics.json
```

## 6. Посчитать экспертные метрики

```bash
python3 src/evaluation.py
```

По умолчанию сравниваются BM25, Dense Flat и HNSW с `efSearch=64` на глубине
`k=10`. Оценки 2 и 3 считаются релевантными, а nDCG использует полную шкалу
0–3.

Рассчитываются:

- nDCG@10;
- MRR@10;
- Precision@10;
- Recall@10;
- средняя и p95 latency (если она есть в файлах выдачи);
- покрытие результатов экспертной разметкой.

Результаты сохраняются в:

```text
results/evaluation/summary_metrics.csv
results/evaluation/per_query_metrics.csv
results/evaluation/category_metrics.csv
```

Если требуется сравнить другой HNSW-запуск:

```bash
python3 src/evaluation.py \
  --run bm25=results/bm25_top20.csv \
  --run dense_flat=results/dense_flat_top20.csv \
  --run hnsw_efs128=results/hnsw/hnsw_m32_efc200_efs128_top20.csv
```

## 7. Что проверить перед отчётом

- `mean_judgement_coverage_at_k` желательно должен быть равен `1.0`;
- ANN Recall@10 относится только к близости HNSW к точному Dense Search;
- обычный Recall@10 рассчитывается по экспертной разметке;
- итоговое сравнение должно включать качество, latency, RAM и размер индекса;
- для таблиц и презентации следует зафиксировать версии корпуса, модели и
  параметры HNSW.
