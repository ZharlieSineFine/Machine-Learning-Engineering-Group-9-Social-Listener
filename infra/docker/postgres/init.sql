-- Postgres bootstrap. Runs once on first start of the postgres container.
--
-- Layout:
--   sentiment   — application data (reviews, predictions, monitoring reports)
--   airflow     — Airflow metadata DB
--   mlflow      — MLflow backend store
--
-- POSTGRES_DB=sentiment is created automatically by the image, so we only
-- need to CREATE the other two here.

CREATE DATABASE airflow;
CREATE DATABASE mlflow;

\connect sentiment

-- =========================================================================
-- reviews — ingested review rows. The ingestion DAG (step 2) appends here.
-- Contract owned by Charlie/Ha; consumed by Van (modelling) and Amelia (UI).
-- =========================================================================
CREATE TABLE IF NOT EXISTS reviews (
    id           BIGSERIAL PRIMARY KEY,
    text         TEXT NOT NULL,
    label        TEXT NOT NULL CHECK (label IN ('negative', 'neutral', 'positive')),
    rating       REAL,
    source       TEXT NOT NULL,
    restaurant   TEXT,
    location     TEXT,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS reviews_ingested_at_idx ON reviews (ingested_at);
CREATE INDEX IF NOT EXISTS reviews_label_idx        ON reviews (label);

-- =========================================================================
-- predictions — every /predict call gets logged here (Phase 2 wiring).
-- Lets the dashboard show "what is the model saying right now" and gives
-- the monitoring DAG a target-drift signal.
-- =========================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    review_id       BIGINT REFERENCES reviews(id) ON DELETE SET NULL,
    text            TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS predictions_predicted_at_idx ON predictions (predicted_at);

-- =========================================================================
-- monitoring_reports — pointer rows for Evidently HTML reports stored in
-- MinIO. The dashboard reads the latest row per type to embed the report.
-- =========================================================================
CREATE TABLE IF NOT EXISTS monitoring_reports (
    id                 BIGSERIAL PRIMARY KEY,
    run_date           DATE NOT NULL,
    report_type        TEXT NOT NULL,  -- 'data_drift' | 'target_drift' | 'performance'
    report_url         TEXT NOT NULL,  -- s3://monitoring/...
    drift_score        REAL,
    blocked_promotion  BOOLEAN NOT NULL DEFAULT FALSE,
    triggered_retrain  BOOLEAN NOT NULL DEFAULT FALSE,  -- set by monitoring/retrain_trigger.py when a retrain is kicked
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS monitoring_reports_run_date_idx ON monitoring_reports (run_date DESC);

-- =========================================================================
-- reviews_silver — harmonised, deduped reviews published from the medallion
-- (data/publish.py). NO labels (labels live in Gold). Keyed by the loader's
-- natural key (source, source_id). Mirrors data/storage/warehouse.py DDL.
-- =========================================================================
CREATE TABLE IF NOT EXISTS reviews_silver (
    source       TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    text         TEXT NOT NULL,
    text_len     INTEGER,
    rating       REAL,
    restaurant   TEXT,
    location     TEXT,
    review_date  TEXT,
    ingested_at  TEXT,
    PRIMARY KEY (source, source_id)
);
CREATE INDEX IF NOT EXISTS reviews_silver_review_date_idx ON reviews_silver (review_date);

-- =========================================================================
-- reviews_gold — per-review training set (feature_store + label_store joined).
-- review_id == Silver source_id. Label derived from rating in Gold.
-- =========================================================================
CREATE TABLE IF NOT EXISTS reviews_gold (
    review_id    TEXT PRIMARY KEY,
    review_date  TEXT,
    text         TEXT NOT NULL,
    label        TEXT NOT NULL CHECK (label IN ('negative', 'neutral', 'positive')),
    label_source TEXT NOT NULL DEFAULT 'derived_from_rating',
    text_len     INTEGER,
    built_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS reviews_gold_review_date_idx ON reviews_gold (review_date);
CREATE INDEX IF NOT EXISTS reviews_gold_label_idx       ON reviews_gold (label);

-- =========================================================================
-- human_corrections — reviewer fixes that feed the next training run.
-- =========================================================================
CREATE TABLE IF NOT EXISTS human_corrections (
    id              BIGSERIAL PRIMARY KEY,
    review_id       TEXT NOT NULL,
    corrected_label TEXT NOT NULL CHECK (corrected_label IN ('negative', 'neutral', 'positive')),
    corrected_by    TEXT,
    corrected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
