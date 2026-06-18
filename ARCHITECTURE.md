# Architecture — Brand Sentiment Analysis Platform

> **One-line:** A medallion-architecture MLOps pipeline that ingests restaurant reviews, refines them through Bronze → Silver → Gold layers, classifies sentiment with a fine-tuned transformer, and serves predictions through a shadow-deployed REST API and a Streamlit dashboard — all reproducible with `docker compose up`.

---

## 1. Goal

Build a production-style sentiment analysis system for online restaurant brand reviews. The system must be:

- **End-to-end** — sources → bronze → silver → gold → train → register → shadow serve → monitor → feedback
- **Locally deployable** — every component runs in Docker on a developer laptop
- **Cloud-portable (stretch)** — swap Postgres → RDS, MinIO → S3 and the stack lifts to AWS/GCP/Azure
- **Reproducible** — pinned dependencies; CI runs the same images as local dev
- **Thin-slice first** — a working skeleton end-to-end before depth in any one layer
- **Closed-loop** — drift detected by Evidently triggers retraining; human-corrected labels feed the next training run

---

## 2. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **Airflow** | Schedules every step on a **6-hour batch cycle** |
| Storage (metadata + features) | **Postgres** | Source of truth for reviews, predictions, run metadata |
| Object storage | **MinIO** | S3-compatible local store for raw JSON, model artifacts, drift reports |
| Modeling | **HuggingFace Transformers + scikit-learn** | DistilBERT fine-tune; sklearn baseline + utilities |
| Topic modelling (stretch) | **BERTopic** | Theme extraction from negative reviews |
| Experiment tracking + registry | **MLflow** | Logs runs, metrics, params; promotes models `None → Staging → Production` |
| Data validation | **Great Expectations** | Schema + value-range checks between bronze → silver |
| Drift + performance monitoring | **Evidently** | Data quality, model drift, prediction confidence — fires retraining when thresholds break |
| Serving | **FastAPI** | REST endpoint; **shadow deploy** (candidate model runs alongside production for comparison before promotion) |
| Dashboard | **Streamlit** | KPI tiles, sentiment timelines, alerts, digest |
| Tests | **pytest** | Unit + integration; smoke container in compose |
| CI/CD | **GitHub Actions** | Lint, test, build images, push to registry |
| Containerisation | **Docker / Docker Compose** | One-command spin-up |

---

## 3. Data Flow — Medallion Architecture

Airflow schedules every step on a **6-hour batch inference cycle**.

```
   SOURCES                BRONZE              SILVER              GOLD
┌───────────────┐    ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│ Yelp Open     │    │   Raw         │   │   Cleaned     │   │ Feature + Label   │
│ Reviews       │───►│ Original JSON │──►│ • Validated   │──►│ Embeddings,       │
│ Malaysia      │    │ + provenance  │   │ • Deduplicated│   │ aggregates,       │
│ Restaurant    │    │               │   │ • PII-masked  │   │ sentiment labels  │
│ Reviews       │    └───────────────┘   └───────────────┘   └────────┬──────────┘
│ Replay        │                                                     │
│ Simulator     │                                                     │
└───────────────┘                                                     │
                                                                      ▼
                                         ┌────────────────────────────────────────┐
                                         │  Train + Register   DistilBERT · MLflow│
                                         │  Inference (shadow) FastAPI            │
                                         │  Dashboard          Streamlit · alerts │
                                         │  Monitoring         Evidently          │
                                         └────────────────┬───────────────────────┘
                                                          │
                              ┌───────────────────────────┘
                              ▼
        MLOps Monitoring & Feedback Loop:
        Evidently watches data quality, model drift, and prediction confidence.
        Drift > threshold → retraining is triggered automatically.
        Human-corrected labels feed the next training run.
```

### Layer contracts (what each layer guarantees the next one)

Reviews are **immutable events**, not monthly state snapshots. The daily driver (`data/run_daily.py`) lands each pull under an **ingestion date** (`dt=YYYY-MM-DD`) and upserts **event-date** partitions (`review_date=YYYY-MM-DD`). Re-running the same ingestion date is idempotent; late-arriving reviews update the partition for their `review_date`, not today's.

| Layer | Storage | Contract | Owner |
|---|---|---|---|
| **Sources** | external + `data/sample/reviews_sample.csv` (in repo, ~1k labelled rows for smoke + baseline) | JSON / CSV from upstream, no guarantees. Sample columns: `text, label, rating, source, restaurant, location` | Charlie + Ha |
| **Bronze** | MinIO `s3://datasets/bronze/{source}/dt={YYYY-MM-DD}/` (locally `data/bronze/{source}/dt=…/`) | **Raw, source-native** rows — verbatim source columns + `_source`, `_ingested_at`. No join/label/cleaning. Partition = load date. | Charlie + Ha |
| **Silver** | Postgres `reviews_silver` + `data/silver/reviews/review_date=…/` | GE-gated, deduped by `(source, source_id)` keeping latest `_ingested_at`. **Scoped to the last 3 years per source** (each source anchors on its own max review date); Bronze keeps full history. Carries `rating` + ISO `date` + `source_id`. **No labels.** Partition = review event date. Pass `--all-years` to `run_daily` for the full archive. | Charlie + Ha |
| **Gold** | `data/gold/feature_store/` + `data/gold/label_store/` (+ Postgres `reviews_gold` in prod) | Per-review `feature_store` (`review_id`, `review_date`, `text`) and `label_store` (`review_id`, `review_date`, `label`) keyed by `review_id` (= `source_id`). Labels derived from `rating` in Gold only. | Charlie + Ha → Van |

### Replay simulator

Lives in `data/ingest/replay.py`. Replays a pre-built demo window (`demo_data/`) as a
time-ordered BrewLeaf "operational" stream at configurable speed, written to
`data/replay/<scenario>/review_date=YYYY-MM-DD/part.parquet` — a standalone stream the
monitoring step consumes as the "current" window (it does not round-trip through Bronze).

Two scenarios drive the drift demo (both June 2026, the same 2,056 reviews and overall label
mix — only the *timing* of negatives differs):
- `stable` — negatives steady at ~19-20%/day → the drift gate should pass
- `spike` — ~17-18%/day, then a sudden 60% on 2026-06-21 → the drift gate should fire

The spike is baked into the demo data (no synthetic drift injection), and the demo windows
carry no shop-name column, so the text is replayed verbatim (no brand replacement). Run:
`python -m data.ingest.replay --scenario {stable,spike}`.

---

## 4. Model Lifecycle — with Shadow Deploy

```
   Gold ──► train_model ──► MLflow run ──► register model
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Stage: Staging  │
                                    └────────┬────────┘
                                             │ shadow deploy:
                                             │ candidate predicts on
                                             │ live traffic ALONGSIDE
                                             │ Production. Predictions
                                             │ are logged but NOT served.
                                             ▼
                              ┌──────────────────────────┐
                              │  Compare candidate vs.   │
                              │  Production over N hours │
                              │  (F1, agreement rate,    │
                              │   latency, error rate)   │
                              └────────────┬─────────────┘
                                           │
                          gate passes?     │     gate fails?
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
              Promote Staging → Production            Reject; keep Production
                  (manual approval)                   (alert in dashboard)
```

The shadow window is **two 6-hour batch cycles by default** (12h) before promotion is considered.

### Train / validation / test / OOT split

`train_model` doesn't split the Gold set at random. It uses the **Silver** `date` column (normalised to ISO from each source's raw Bronze stamp) to build an **out-of-time (OOT)** hold-out (`models/splits.py`): the most recent slice of reviews (by timestamp) is set aside as a stand-in for "reviews that arrive after we ship", and everything older is split — stratified on label — into **train** (fit), **validation** (tune / model selection), and **test** (in-time estimate).

- `test` measures *in-distribution* generalisation (same period as train).
- `oot` measures *temporal* generalisation (genuinely later reviews).
- The **test → OOT F1 drop** is the offline preview of the drift Evidently watches for in production; it's logged to MLflow (`f1_macro`, `f1_macro_oot`, `oot_cutoff_date`) so the promotion gate can refuse models that hold up in-time but fall apart out-of-time.

Rows with a null `date` join the in-time pool; if no row is dated (e.g. the seed CSV) the split degrades to a plain stratified train/val/test.

---

## 5. Component Map

```
mle_project/
├── airflow/                      # DAGs + Airflow config         (Charlie, Ha; Anh on infra)
│   ├── dags/
│   │   ├── ingest_bronze.py      # Sources → Bronze         (6h)
│   │   ├── refine_silver.py      # Bronze → Silver          (6h, GE-gated)
│   │   ├── build_gold.py         # Silver → Gold            (6h)
│   │   ├── train_model.py        # Gold → MLflow run        (daily)
│   │   ├── shadow_score.py       # Candidate predicts on live (6h)
│   │   └── evaluate_and_monitor.py  # Evidently drift + promotion gate (6h)
│   └── plugins/
├── api/                          # FastAPI service               (Amelia)
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── model_loader.py       # Loads Production + (optional) Staging
│   │   └── shadow.py             # Logs Staging predictions alongside Prod
│   └── Dockerfile (in infra/docker/api/)
├── dashboard/                    # Streamlit                     (Amelia)
│   ├── app.py
│   └── pages/                    # KPIs, drift, alerts, digest, model comparison
├── models/                       # Training code                 (Van + Amelia)
│   ├── train.py
│   ├── evaluate.py
│   ├── baseline_sklearn.py       # Phase 1
│   ├── distilbert_finetune.py    # Phase 2
│   └── embeddings.py             # Used by Gold layer
├── data/                         # Medallion layers              (Charlie + Ha)
│   ├── ingest/                   # SOURCES → BRONZE (raw, source-native + provenance)
│   │   ├── yelp_loader.py        # Yelp tar → dt= partitions (reviews + business)
│   │   ├── malaysia_review_loader.py  # Malaysia TripAdvisor → dt= partitions
│   │   └── replay.py             # Replay simulator (demo_data stable/spike)
│   ├── refine/                   # BRONZE → SILVER (join, clean, dedup); Silver → Gold (labels)
│   │   ├── build_silver.py       # Bronze → review_date= Silver parquet partitions
│   │   ├── build_gold.py         # Silver → feature_store + label_store partitions
│   │   ├── dedupe.py
│   │   └── pii_mask.py
│   ├── run_daily.py              # Incremental driver (bronze → silver → GE → gold)
│   ├── expectations/             # Great Expectations suites
│   ├── schemas/                  # SQL DDL for *_silver, *_gold + Pydantic types
│   └── sample/                   # Tiny in-repo seed for CI smoke
├── monitoring/                   # Evidently + retraining trigger (Charlie + Ha)
│   ├── drift_checks.py
│   └── retrain_trigger.py        # Emits Airflow trigger when drift > τ
├── infra/                        # Docker, compose, CI            (Anh)
│   ├── docker/
│   └── github-actions/
├── tests/                        # Integration                    (Amelia + all)
├── notebooks/                    # Exploration (not in CI)
├── scripts/                      # bootstrap, demo, reset
├── docker-compose.yml
├── ARCHITECTURE.md
├── WORKFLOW.md
└── README.md
```

---

## 6. Services in `docker-compose`

| Service | Port | Profile | Depends on |
|---|---|---|---|
| `postgres` | 5432 | default | — |
| `minio` + `minio-init` | 9000 / 9001 | default | — |
| `mlflow` | 5000 | default | postgres, minio-init |
| `airflow-init` / `webserver` / `scheduler` | 8080 | default | postgres |
| `api` | 8000 | default | mlflow |
| `dashboard` | 8501 | default | api, postgres |
| `smoke` | — | `smoke` (opt-in) | — |

`docker compose up` brings the default stack online.
`docker compose run --rm smoke` runs the isolated smoke test container.

---

## 7. Postgres Schema (canonical)

The sample CSV at `data/sample/reviews_sample.csv` is the source of truth for column shape. Labels are **strings** (`negative | neutral | positive`) to match `models.baseline_sklearn.LABELS`. Ratings come in as 1.0–5.0 floats.

```sql
-- silver: cleaned, deduped, PII-masked
reviews_silver (
  id              BIGSERIAL PRIMARY KEY,
  source          TEXT NOT NULL,        -- google | yelp | malaysia | replay
  restaurant      TEXT,
  location        TEXT,
  text            TEXT NOT NULL,
  rating          REAL,                  -- 1.0 .. 5.0
  label           TEXT,                  -- 'negative' | 'neutral' | 'positive' (nullable until labelled)
  ingested_at     TIMESTAMPTZ NOT NULL,
  cleaned_at      TIMESTAMPTZ DEFAULT now()
);
-- Dedup key (whatever the loader produces); for sources without stable IDs,
-- a hash of (source, restaurant, text, ingested_date) works.
CREATE UNIQUE INDEX reviews_silver_dedup
  ON reviews_silver (source, md5(restaurant || '|' || text));

-- gold: features + labels (the training set)
reviews_gold (
  review_id       BIGINT PRIMARY KEY REFERENCES reviews_silver(id),
  embedding       BYTEA,                 -- (or VECTOR(384) if pgvector enabled)
  text_len        INT,
  label           TEXT NOT NULL,         -- 'negative' | 'neutral' | 'positive'
  -- label_source (future/prod): provenance tracking for human corrections / model labels
  built_at        TIMESTAMPTZ DEFAULT now()
);

predictions (
  id              BIGSERIAL PRIMARY KEY,
  review_id       BIGINT REFERENCES reviews_silver(id),
  model_name      TEXT NOT NULL,
  model_version   INT NOT NULL,
  stage           TEXT NOT NULL,         -- 'Production' | 'Staging' (shadow)
  label           TEXT NOT NULL,         -- 'negative' | 'neutral' | 'positive'
  score           REAL,                  -- nullable (LinearSVC has no predict_proba)
  predicted_at    TIMESTAMPTZ DEFAULT now()
);

monitoring_reports (
  id              BIGSERIAL PRIMARY KEY,
  ran_at          TIMESTAMPTZ DEFAULT now(),
  report_path     TEXT NOT NULL,         -- s3://minio/monitoring/...
  drift_score     REAL,
  f1_macro        REAL,
  passed_gate     BOOLEAN,
  triggered_retrain BOOLEAN DEFAULT FALSE
);

human_corrections (
  id              BIGSERIAL PRIMARY KEY,
  review_id       BIGINT REFERENCES reviews_silver(id),
  corrected_label TEXT NOT NULL,         -- 'negative' | 'neutral' | 'positive'
  corrected_by    TEXT,
  corrected_at    TIMESTAMPTZ DEFAULT now()
);
```

The training DAG joins `reviews_gold` with `human_corrections` and prefers human labels when present — that's how the feedback loop closes. The Gold builder derives `label` from `rating` (`<=2 → negative`, `3 → neutral`, `>=4 → positive`) via `label_from_rating()` in `data/refine/build_gold.py`.

> **Implemented schema (prototype).** The medallion keys on the loaders' natural **string**
> ids, so the live `reviews_silver` / `reviews_gold` tables (DDL in
> `infra/docker/postgres/init.sql`, mirrored in `data/storage/warehouse.py`) use
> `PRIMARY KEY (source, source_id)` and `review_id` (= Silver `source_id`) rather than the
> `BIGSERIAL` ids sketched above, and **Silver carries no `label`** (labels live only in
> Gold). `data/publish.py` populates these tables and mirrors the bronze/silver/gold objects
> to MinIO `s3://datasets/…` from the on-disk medallion (see [data/README.md](../data/README.md#publishing-to-minio--postgres)).

---

## 8. Environments

| Env | Where | Used for |
|---|---|---|
| Local dev | Docker Compose on laptop | Day-to-day development |
| CI | GitHub Actions (same images) | Lint, unit, smoke container, integration |
| Production (stretch) | Any cloud — swap Postgres → managed RDS, MinIO → S3 | Live demo |

Every dependency is pinned. CI uses the same images as local. "Works on my machine" = "works in CI."

---

## 9. Design Principles

- **Thin-slice first.** A sklearn baseline served via FastAPI and shown on the dashboard is more valuable on day 7 than a perfectly tuned DistilBERT nobody can call. Depth comes after the skeleton is green.
- **Medallion discipline.** Don't reach back into Bronze from Gold-side code. Each layer's owner publishes the next one. If you need new columns in Silver, propose it via PR; don't bypass.
- **Contracts between layers are explicit.** Pydantic schemas in `api/app/schemas.py`, SQL DDL in `data/schemas/`. Don't pass implicit dicts across team boundaries.
- **Everything observable.** MLflow logs every run; every DAG run leaves a trace; the dashboard surfaces what's live in prod right now.
- **No secret hand-offs.** Shared values (model name, table name, bucket) belong in `infra/.env.example`, not in chat.
- **The feedback loop is part of the system, not a stretch goal.** `human_corrections` exists in Phase 1 even if no one writes to it yet, so the training DAG already knows how to read it.

---

## 10. Open Questions (resolve in week 1)

- **Label scheme** — derive from stars (1–2 → neg, 3 → neu, 4–5 → pos) or use an external labelled set first?
- **Embedding model** — `sentence-transformers/all-MiniLM-L6-v2` (384d) is a reasonable default; lock it in or evaluate alternatives?
- **Shadow window length** — default 12h (2 cycles); is that long enough to detect a regression?
- **BERTopic on negative reviews** — confirm in scope for Phase 2 vs. stretch
- **Multilingual** — Malaysia Restaurant Reviews likely contain Malay; English-only filter for v1, or attempt multilingual sentiment from day one?
