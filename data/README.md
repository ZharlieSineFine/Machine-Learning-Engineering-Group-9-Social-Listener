# Data — Ingestion, Schemas, Validation

**Owner:** Charlie + Ha

Everything to do with getting reviews into the system and proving they're well-formed.

## Layout

```
data/
├── ingest/         # Source adapters (CSV loader, S3 puller, API client)
├── expectations/   # Great Expectations suite JSONs
├── schemas/        # SQL DDL for Postgres + Pydantic shared types
└── samples/        # Tiny seed datasets for tests (committed to git)
```

## Canonical tables (Postgres)

```sql
reviews (
  id              BIGSERIAL PRIMARY KEY,
  source          TEXT NOT NULL,            -- amazon | yelp | appstore
  text            TEXT NOT NULL,
  label           SMALLINT,                  -- 0=neg, 1=neu, 2=pos (nullable until labelled)
  stars           SMALLINT,
  language        TEXT DEFAULT 'en',
  ingested_at     TIMESTAMPTZ DEFAULT now(),
  source_id       TEXT                       -- upstream id for dedup
);

predictions (
  id              BIGSERIAL PRIMARY KEY,
  review_id       BIGINT REFERENCES reviews(id),
  model_name      TEXT NOT NULL,
  model_version   INT NOT NULL,
  label           SMALLINT NOT NULL,
  score           REAL NOT NULL,
  predicted_at    TIMESTAMPTZ DEFAULT now()
);

monitoring_reports (
  id              BIGSERIAL PRIMARY KEY,
  ran_at          TIMESTAMPTZ DEFAULT now(),
  report_path     TEXT NOT NULL,            -- s3://minio/monitoring/...
  drift_score     REAL,
  f1_macro        REAL,
  passed_gate     BOOLEAN
);
```

These schemas are a contract. Change them via a migration PR, not in-place.

## Great Expectations checkpoints

Phase 1 — minimum viable:
- `text` not null, length between 1 and 5000
- `source` in allowed set
- `label` in {0, 1, 2} or NULL

Phase 2 — full suite: language match, duplicate rate, label cardinality drift.

## Phase 1 stub
Ingest a committed sample CSV in `data/samples/` so the rest of the pipeline has something to chew on from day one.
