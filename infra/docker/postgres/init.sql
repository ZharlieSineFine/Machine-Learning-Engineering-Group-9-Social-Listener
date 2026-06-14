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
    stage           TEXT CHECK (stage IS NULL OR stage IN ('Production', 'Staging')),
    score           REAL,
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
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS monitoring_reports_run_date_idx ON monitoring_reports (run_date DESC);
