# Data — Medallion Layers

**Owner:** Charlie + Ha

This folder holds the **code** that builds the medallion layers. The data itself lives in MinIO (bronze, gold artifacts) and Postgres (silver, gold tables). See [ARCHITECTURE.md §3](../ARCHITECTURE.md#3-data-flow--medallion-architecture) for the full picture.

## Layout

```
data/
├── ingest/         # SOURCES → BRONZE. Loaders + replay simulator.
│   ├── yelp_loader.py
│   ├── malaysia_loader.py
│   └── replay.py
├── refine/         # BRONZE → SILVER. Validation, dedupe, PII masking.
│   ├── dedupe.py
│   └── pii_mask.py
├── expectations/   # Great Expectations suites (gate between layers)
├── schemas/        # SQL DDL + Pydantic types shared across services
└── sample/         # In-repo seed used by the smoke test (committed)
                    # data/sample/reviews_sample.csv
```

> The big raw dataset (`Malaysia Restaurant Review Datasets/`, ~227 MB) is **gitignored**. Distribute it via release asset / S3 / DVC, not git.

## Medallion layers — where each one lives

| Layer | Code that builds it | Storage |
|---|---|---|
| **Sources** | external; replay simulator (`ingest/replay.py`) | URLs / `data/sample/` |
| **Bronze** — raw + provenance | `ingest/*.py`, called by `airflow/dags/ingest_bronze.py` | MinIO `s3://datasets/bronze/{source}/{YYYY-MM-DD}/` |
| **Silver** — validated, deduped, PII-masked | `refine/*.py` + GE checks, called by `airflow/dags/refine_silver.py` | Postgres `reviews_silver` + MinIO `silver/` |
| **Gold** — embeddings, aggregates, labels | `models/embeddings.py` + aggregation SQL, called by `airflow/dags/build_gold.py` | Postgres `reviews_gold` + MinIO `gold/embeddings/` |

## Sources (Phase 2 onwards)

| Source | Adapter | Notes |
|---|---|---|
| **Yelp Open Reviews** | `ingest/yelp_loader.py` | Public dataset; English |
| **Malaysia Restaurant Reviews** | `ingest/malaysia_loader.py` | Mixed English / Malay — decide multilingual policy in week 1 |
| **Replay simulator** | `ingest/replay.py` | Replays a fixed timeline into Bronze at configurable speed — used for drift demos and CI smoke tests |

## Canonical tables — see ARCHITECTURE.md §7

The full DDL lives in [`ARCHITECTURE.md §7`](../ARCHITECTURE.md#7-postgres-schema-canonical). When schemas change, update them via a migration PR in `data/schemas/`, not in-place.

Key tables: `reviews_silver`, `reviews_gold`, `predictions`, `monitoring_reports`, `human_corrections`.

## Great Expectations checkpoints

**Phase 1 — minimum viable (Bronze → Silver gate):**
- `text` not null, length 1–5000
- `source` in {`yelp`, `malaysia`, `replay`}
- `source_id` not null
- `stars` in [1, 5] or NULL

**Phase 2 — full suite:**
- Language detection matches `language` column
- Duplicate rate (by `(source, source_id)`) < 1%
- Label distribution drift vs. reference

When GE fails, the DAG fails — silver is never written from bad bronze.

## Sample CSV schema (the contract)

`data/sample/reviews_sample.csv` is **the contract** every layer downstream agrees on. Columns:

| Column | Type | Notes |
|---|---|---|
| `text` | str | Review text. Non-null, > 0 chars |
| `label` | str | One of `negative`, `neutral`, `positive` (matches `models.baseline_sklearn.LABELS`) |
| `rating` | float | 1.0–5.0; used by Gold builder when `label` is missing |
| `source` | str | e.g. `google`, `yelp`, `replay` |
| `restaurant` | str | Restaurant name |
| `location` | str | City / region |

Change a column? Open a PR that updates the CSV, the Pydantic schemas in `api/app/schemas.py`, the Postgres DDL in `data/schemas/`, *and* `models/baseline_sklearn.py` in one commit. No silent drift.

## Phase 1 status

`data/sample/reviews_sample.csv` is committed (~1k labelled rows) so the smoke test never needs network. The Phase 2 `replay` simulator reads from this same shape.
