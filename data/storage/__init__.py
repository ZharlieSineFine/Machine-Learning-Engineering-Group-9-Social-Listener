"""Storage layer for the medallion: MinIO (S3 object store) + Postgres warehouse.

Modules:
    config      — connection settings from env (Postgres DSN, S3 endpoint/keys, buckets)
    warehouse   — Postgres DDL + idempotent upserts for reviews_silver / reviews_gold
    objectstore — MinIO/S3 client + mirror local medallion trees to s3://datasets/

`data/publish.py` ties them together: local medallion -> MinIO objects + Postgres tables.

Owner: Charlie + Ha (Data & Eval).
"""
