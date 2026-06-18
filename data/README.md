# Data — Medallion Layers



**Owner:** Charlie + Ha



This folder holds the **code** that builds the medallion layers. The data itself lives in MinIO (bronze, gold artifacts) and Postgres (silver, gold tables). See [ARCHITECTURE.md §3](../ARCHITECTURE.md#3-data-flow--medallion-architecture) for the full picture.



## Layout



```

data/

├── ingest/         # SOURCES → BRONZE. Raw, source-native loaders (no transformation).

│   ├── yelp_loader.py            # Yelp tar → raw reviews.csv + business.csv (+ provenance)

│   ├── malaysia_review_loader.py # Malaysia TripAdvisor → raw reviews.csv (+ provenance)

│   └── replay.py     # Replay simulator: demo_data (stable/spike) -> data/replay/<scenario>/

├── refine/         # BRONZE → SILVER → GOLD. Join, clean, dedup, label derivation.

│   ├── build_silver.py # Bronze → Silver partitions (no labels) + ISO date + source_id

│   ├── build_gold.py   # Silver → Gold feature_store + label_store partitions

│   ├── dedupe.py

│   └── pii_mask.py

├── run_daily.py    # Incremental driver: bronze → silver → GE gate → gold for one date

├── expectations/   # Great Expectations suites (gate between layers)

├── schemas/        # SQL DDL + Pydantic types shared across services

└── sample/         # In-repo seed used by the smoke test (committed)

                    # data/sample/reviews_sample.csv

```



> The big raw dataset (`Malaysia Restaurant Review Datasets/`, ~227 MB) is **gitignored**. Distribute it via release asset / S3 / DVC, not git.



## Daily incremental layout (reviews are events, not snapshots)



Reviews are **immutable events** keyed by `review_date` (when the review was written), not monthly state snapshots. Each daily run lands new pulls under an **ingestion date** and upserts the affected **event-date** partitions.



| Layer | Partition key | Local path pattern |

|---|---|---|

| **Bronze** | `ingestion_date` (`dt=YYYY-MM-DD`) | `data/bronze/<source>/dt=YYYY-MM-DD/reviews.csv` (+ `business.csv` for Yelp) |

| **Silver** | `review_date` | `data/silver/reviews/review_date=YYYY-MM-DD/part.parquet` |

| **Gold feature store** | `review_date` | `data/gold/feature_store/review_date=YYYY-MM-DD/part.parquet` |

| **Gold label store** | `review_date` | `data/gold/label_store/review_date=YYYY-MM-DD/part.parquet` |



Run the full daily pipeline:



```bash

python -m data.run_daily --run-date 2026-06-06 --sources yelp tripadvisor
# Bronze = full archive (~730k reviews); Silver/Gold default to last 3 years per source (~226k).
# Full history in Silver/Gold: add --all-years

```



With the multi-root workspace open, `run_daily` auto-discovers sibling datasets (see `data/paths.py`):

| Source | Default path (from repo root) |
|---|---|
| Yelp tar | `../../Yelp_JSON/yelp_dataset/yelp_dataset` |
| TripAdvisor CSV | `../../TripAdvisor_data_cleaned.csv/TripAdvisor_data_cleaned.csv` |

Override via `YELP_TAR_PATH`, `YELP_REVIEWS_PATH`+`YELP_BUSINESS_PATH`, or `TRIPADVISOR_CSV_PATH` (copy `.env.example` → `.env`). If no source file is found, pre-seed bronze under `dt=<run-date>/` and pass `--skip-bronze`.



**Idempotency:** re-running the same `--run-date` overwrites that day's bronze partition and re-merges silver/gold for affected `review_date` keys. **Dedup:** Silver keeps the latest row per `(source, source_id)` by `_ingested_at`. **Late arrivals:** a review ingested today but written in 2016 updates `review_date=2016-03-09`, not today's partition.



Rows with unparseable dates land in `review_date=__null__` and join the in-time training pool (see `models/splits.py`).



## Medallion layers — where each one lives



| Layer | Code that builds it | Storage |

|---|---|---|

| **Sources** | external; replay simulator (`ingest/replay.py`) | URLs / `data/sample/` (seed); `demo_data/` (replay stable/spike windows) |

| **Bronze** — raw + provenance | `ingest/*.py`, `run_daily.py` | MinIO `s3://datasets/bronze/{source}/dt={YYYY-MM-DD}/` |

| **Silver** — validated, deduped | `refine/build_silver.py` + GE gate | Postgres `reviews_silver` + `data/silver/reviews/` |

| **Gold** — per-review features + labels | `refine/build_gold.py` | `data/gold/feature_store/` + `data/gold/label_store/` |



## Sources (Phase 2 onwards)



| Source | Adapter (Bronze) | Notes |

|---|---|---|

| **Yelp Open Reviews** | `ingest/yelp_loader.py` | English. Streams the ~9 GB tar (`--from-tar`) → **raw** `reviews.csv` + `business.csv` under `dt=`. `source_id` = `review_id`. |

| **Malaysia Restaurant Reviews** | `ingest/malaysia_review_loader.py` | TripAdvisor Malaysia; mixed English / Malay. `source_id` = SHA-256 of `(restaurant, text, parsed date)`. |

| **Replay simulator** | `ingest/replay.py` | Replays a fixed timeline into Bronze at configurable speed — used for drift demos and CI smoke tests |



> **Beverage scope + layering.** BrewLeaf is a bubble-tea brand, so both adapters default to the drinks/beverage slice (row selection only; values stay raw). Silver joins/cleans/normalises `date` to ISO (no labels). Gold derives `label` from `rating`. All layer outputs under `data/bronze/`, `data/silver/`, `data/gold/` are gitignored — regenerate from raw sources.



## Canonical tables — see ARCHITECTURE.md §7



The full DDL lives in [`ARCHITECTURE.md §7`](../ARCHITECTURE.md#7-postgres-schema-canonical). When schemas change, update them via a migration PR in `data/schemas/`, not in-place.



Key tables: `reviews_silver`, `reviews_gold`, `predictions`, `monitoring_reports`, `human_corrections`.



## Great Expectations checkpoints



**Phase 1 — minimum viable (Bronze → Silver gate, run by `run_daily.py`):**

- `text` not null, length 1–5000

- `source` in {`yelp`, `malaysia`, `replay`, `google`, `tripadvisor`}

- `source_id` not null

- `rating` in [1, 5]

- `restaurant` / `location` not null

- Silver must **not** contain a `label` column



**Phase 2 — full suite:**

- Language detection matches `language` column

- Duplicate rate (by `(source, source_id)`) < 1%

- Label distribution drift vs. reference



When GE fails, the daily run fails — Gold is not written for that batch.



## Layer schemas



### Silver (`refine/build_silver.py` → `data/silver/reviews/`)

**Recency scope (default):** after harmonise + dedup, Silver keeps only the **last 3 calendar years per source**, anchored on each source's newest review (`DEFAULT_SILVER_RECENT_YEARS` in `ingest_reviews.py`). Yelp and TripAdvisor can end on different dates, so cutoffs are computed independently. Bronze is unchanged. Disable with `--all-years` on `run_daily` / `build_silver`.




Harmonised reviews — **no derived labels**. Columns (`SILVER_COLUMNS` + `date`):



| Column | Type | Notes |

|---|---|---|

| `text` | str | Review text. Non-null, > 0 chars |

| `rating` | float | 1.0–5.0 star rating from the source |

| `source` | str | e.g. `yelp`, `tripadvisor`, `replay` |

| `source_id` | str | Stable per-review key within the source |

| `restaurant` | str | Restaurant name (joined in Silver for Yelp) |

| `location` | str | City / region |

| `date` | str (nullable) | Review timestamp normalised to ISO in Silver. Drives the OOT split (`models/splits.py`). |



`_ingested_at` is stored in the parquet for dedup but is not part of the GE contract.



### Gold (`refine/build_gold.py` → `data/gold/`)



Per-review stores keyed by `review_id` (= Silver `source_id`):



**feature_store:** `review_id`, `review_date`, `text`



**label_store:** `review_id`, `review_date`, `label`



Label rule: `rating <= 2 → negative`, `rating == 3 → neutral`, `rating >= 4 → positive` (derived in `build_gold.py` from Silver `rating`).



Legacy batch mode still supports a single combined Gold CSV via `--silver` + `--out`.



### Sample CSV (Gold-shaped seed for smoke tests)



`data/sample/reviews_sample.csv` is a **Gold-shaped** seed (~1k labelled rows) for CI. It matches `EXPECTED_COLUMNS` (the 6-column training contract).



### Bronze ≠ this contract (strict raw)



Bronze is **not** the contract — it is the raw source, verbatim, per adapter, plus `_source` / `_ingested_at`:



| Bronze table | Columns (source-native) |

|---|---|

| `bronze/yelp/dt=…/reviews.csv` | `review_id, user_id, business_id, stars, useful, funny, cool, text, date, _source, _ingested_at` |

| `bronze/yelp/dt=…/business.csv` | `business_id, name, address, city, state, postal_code, stars, review_count, categories, _source, _ingested_at` |

| `bronze/tripadvisor/dt=…/reviews.csv` | `Author, Title, Review, Rating, Dates, Restaurant, Location, _source, _ingested_at` |



Change a contract column? Open a PR that updates the CSV, the Pydantic schemas in `api/app/schemas.py`, the Postgres DDL in `data/schemas/`, *and* `models/baseline_sklearn.py` in one commit. No silent drift.



## Phase 1 status



`data/sample/reviews_sample.csv` is committed (~1k labelled rows) so the smoke test never needs network. The Phase 2 `replay` simulator instead replays the purpose-built demo windows in `demo_data/` (`stable` / `spike`) — see [ARCHITECTURE.md §Replay simulator](../ARCHITECTURE.md).


