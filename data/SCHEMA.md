# Data layer schema reference

Handoff doc for teammates (modeling, eval, and data consumers). Describes the **Bronze → Silver → Gold** medallion layout, column contracts from code (`data/ingest/ingest_reviews.py`), and dtypes observed on disk.

**Owner:** Charlie + Ha (Data & Eval)  
**Snapshot:** `run_date=2026-06-06`, default **3-year** Silver scope per source

---

## Current dataset snapshot (on disk)

| Layer | Format | Partitions | Rows | Date range |
|-------|--------|------------|------|------------|
| Silver | Parquet | 1,120 | 226,433 | 2019-01-19 … 2022-04-10 |
| Gold `feature_store` | Parquet | 1,120 | 226,433 | 2019-01-19 … 2022-04-10 |
| Gold `label_store` | Parquet | 1,120 | 226,433 | 2019-01-19 … 2022-04-10 |

Bronze retains the full archive (~713k Yelp + ~17k TripAdvisor logical rows); Silver/Gold above are after the 3-year per-source filter.

---

## On-disk layout

| Layer | Format | Path pattern |
|-------|--------|--------------|
| **Bronze** | CSV | `data/bronze/<source>/dt=YYYY-MM-DD/reviews.csv` (+ `business.csv` for Yelp) |
| **Silver** | Parquet | `data/silver/reviews/review_date=YYYY-MM-DD/part.parquet` |
| **Gold** | Parquet | `data/gold/feature_store/review_date=YYYY-MM-DD/part.parquet` |
| | | `data/gold/label_store/review_date=YYYY-MM-DD/part.parquet` |

Partition keys:

- **Bronze:** `dt=<ingestion_date>` — when the batch was landed.
- **Silver / Gold:** `review_date=<event_date>` — calendar date of the review (Hive-style directory name).

Inspect without re-running the pipeline:

```bash
python scripts/peek_layers.py --silver-only
python scripts/peek_layers.py --gold-only
```

---

## Layer overview

```
Bronze (source-native CSV + provenance)
    ↓  data/refine/build_silver.py
Silver (harmonised parquet, rating only — no label)
    ↓  data/refine/build_gold.py
Gold feature_store + label_store (training-ready parquet)
```

Bronze has **no single unified schema** — each adapter keeps raw export columns plus `_source` and `_ingested_at`. Silver harmonises both sources; Gold derives `label` from `rating`.

---

## Bronze

### Yelp — `reviews.csv` (11 columns)

Path: `data/bronze/yelp/dt=<ingestion_date>/reviews.csv`

| Column | Dtype | Description |
|--------|-------|-------------|
| `review_id` | `str` | Yelp review ID |
| `user_id` | `str` | Yelp user ID |
| `business_id` | `str` | FK to `business.csv` |
| `stars` | `float64` | Review rating (1–5) |
| `useful` | `int64` | Yelp vote count |
| `funny` | `int64` | Yelp vote count |
| `cool` | `int64` | Yelp vote count |
| `text` | `str` | Review body |
| `date` | `str` | Raw Yelp timestamp, e.g. `2018-07-07 22:09:11` |
| `_source` | `str` | Always `"yelp"` |
| `_ingested_at` | `str` | UTC provenance, e.g. `2026-06-06T01:41:46Z` |

### Yelp — `business.csv` (11 columns)

Path: `data/bronze/yelp/dt=<ingestion_date>/business.csv`

| Column | Dtype | Description |
|--------|-------|-------------|
| `business_id` | `str` | Primary key |
| `name` | `str` | Business name |
| `address` | `str` | Street address |
| `city` | `str` | City |
| `state` | `str` | State / region |
| `postal_code` | `str` | Postal code |
| `stars` | `float64` | Business average rating |
| `review_count` | `int64` | Total reviews on business |
| `categories` | `str` | Yelp category string |
| `_source` | `str` | Always `"yelp"` |
| `_ingested_at` | `str` | UTC provenance |

Joined to reviews in Silver (`business_id` → `name`, `city`).

### TripAdvisor — `reviews.csv` (9 columns)

Path: `data/bronze/tripadvisor/dt=<ingestion_date>/reviews.csv`

| Column | Dtype | Description |
|--------|-------|-------------|
| `Author` | `str` | Reviewer name |
| `Title` | `str` | Review title |
| `Review` | `str` | Review body |
| `Rating` | `int64` | Star rating (1–5) |
| `Dates` | `str` | Raw text, e.g. `Reviewed 6 February 2022` |
| `Restaurant` | `str` | Venue name |
| `Location` | `str` | City / area |
| `_source` | `str` | Always `"tripadvisor"` |
| `_ingested_at` | `str` | UTC provenance |

---

## Silver

**File:** `data/silver/reviews/review_date=YYYY-MM-DD/part.parquet`  
**Columns:** 8  
**Contract:** `SILVER_COLUMNS` + `date` + `_ingested_at` in `data/ingest/ingest_reviews.py`

| Column | Dtype | Source mapping |
|--------|-------|----------------|
| `text` | `str` | Yelp `text` / TripAdvisor `Review` |
| `rating` | `float64` | Yelp `stars` / TripAdvisor `Rating` |
| `source` | `str` | `"yelp"` or `"tripadvisor"` |
| `source_id` | `str` | Yelp: `review_id`; TripAdvisor: SHA-256 of `(restaurant, text, parsed date)` |
| `restaurant` | `str` | Yelp business `name` / TripAdvisor `Restaurant` |
| `location` | `str` | Yelp `city` / TripAdvisor `Location` |
| `date` | `str` | Yelp: raw timestamp string; TripAdvisor: ISO `YYYY-MM-DD` |
| `_ingested_at` | `str` | Dedup key across ingestion batches (not used in GE contract) |

**Notes:**

- No `label` column — labels are derived in Gold only.
- Rows with invalid/missing dates may land in `review_date=__null__` if present.
- Default scope: last **3 calendar years per source** (each source anchors on its own max review date). Bronze keeps full history.

**Example row (TripAdvisor):**

| Field | Value |
|-------|-------|
| `text` | `The buffet Ramadhan has many varieties of food...` |
| `rating` | `4.0` |
| `source` | `tripadvisor` |
| `source_id` | `f0bc2e96d509ddd1...` (64-char hex) |
| `restaurant` | `The Resort Cafe` |
| `location` | `Shah Alam` |
| `date` | `2022-04-10` |
| `_ingested_at` | `2026-06-06T01:42:26Z` |

---

## Gold

Gold splits **features** and **labels** into two parquet stores per `review_date`. Join on `review_id` (= Silver `source_id`).

### `feature_store/part.parquet` (8 columns)

Path: `data/gold/feature_store/review_date=YYYY-MM-DD/part.parquet`

| Column | Dtype | Description |
|--------|-------|-------------|
| `review_id` | `str` | Canonical per-review key (= Silver `source_id`) |
| `review_date` | `str` | ISO `YYYY-MM-DD` (normalised event date) |
| `text` | `str` | Review body |
| `rating` | `float64` | Star rating (1–5) |
| `source` | `str` | `"yelp"` or `"tripadvisor"` |
| `restaurant` | `str` | Venue name |
| `location` | `str` | City / area |
| `text_len` | `int64` | Character length of `text` |

### `label_store/part.parquet` (4 columns)

Path: `data/gold/label_store/review_date=YYYY-MM-DD/part.parquet`

| Column | Dtype | Description |
|--------|-------|-------------|
| `review_id` | `str` | Same as `feature_store.review_id` |
| `review_date` | `str` | ISO `YYYY-MM-DD` |
| `label` | `str` | `positive`, `negative`, or `neutral` |
| `label_source` | `str` | `derived_from_rating` |

### Label derivation rule

| Rating | Label |
|--------|-------|
| ≥ 4 | `positive` |
| ≤ 2 | `negative` |
| 3 | `neutral` |

Implemented in `data/refine/build_gold.py` (`label_from_rating`).

**Example (Yelp):**

| Store | Field | Value |
|-------|-------|-------|
| feature_store | `review_id` | `7jB65y5k5Gg5-MPZnyqXNQ` |
| feature_store | `review_date` | `2019-01-19` |
| feature_store | `rating` | `3.0` |
| feature_store | `text_len` | `469` |
| label_store | `label` | `neutral` |
| label_store | `label_source` | `derived_from_rating` |

---

## Training handoff (for modelers)

`models/train.py` and `models/splits.py` expect a flat table with at least:

| Column | Gold source |
|--------|-------------|
| `text` | `feature_store.text` |
| `label` | `label_store.label` |
| `rating` | `feature_store.rating` |
| `source` | `feature_store.source` |
| `restaurant` | `feature_store.restaurant` |
| `location` | `feature_store.location` |
| `date` | `feature_store.review_date` (for out-of-time splits) |

Merge Gold stores on `review_id` (and `review_date` if needed), or export a single CSV for `python models/train.py --data <path>`.

Build Gold from existing Silver (if not already built):

```bash
python -m data.refine.build_gold --silver-root data/silver/reviews --gold-root data/gold
```

---

## Column count summary

| Artifact | Columns |
|----------|---------|
| Bronze Yelp `reviews.csv` | 11 |
| Bronze Yelp `business.csv` | 11 |
| Bronze TripAdvisor `reviews.csv` | 9 |
| Silver `part.parquet` | 8 |
| Gold `feature_store/part.parquet` | 8 |
| Gold `label_store/part.parquet` | 4 |

---

## Related docs

- Pipeline & daily run: [`data/README.md`](README.md)
- Architecture: [`ARCHITECTURE.md`](../ARCHITECTURE.md) §3 (medallion data flow)
- Code contracts: [`data/ingest/ingest_reviews.py`](ingest/ingest_reviews.py)
