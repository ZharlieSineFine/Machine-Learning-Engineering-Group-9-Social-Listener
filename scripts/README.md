# Scripts — Developer ergonomics

One-shot helpers that aren't part of the runtime pipeline.

| Script | Purpose |
|---|---|
| `bootstrap.sh` | First-run setup: copy `.env.example` → `.env`, init Postgres, init MinIO bucket |
| `demo.sh` | Phase 3 end-to-end demo: seed data, trigger DAGs, open browser tabs |
| `seed_sample_reviews.py` | Load `data/samples/*.csv` into Postgres |
| `reset.sh` | Wipe volumes, re-run bootstrap (useful when schemas change) |
