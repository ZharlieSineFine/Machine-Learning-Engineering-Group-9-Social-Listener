"""Connection config for the medallion's Postgres warehouse and MinIO (S3) object store.

Reads the same env vars docker-compose injects (see infra/.env.example):

    Postgres : POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
    MinIO/S3 : AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_ENDPOINT_URL

Inside the docker network the hosts are the service aliases (`postgres`, `minio`). For a
host-side run (pytest / CLI on your laptop) the containers are reachable on mapped ports,
so override `POSTGRES_HOST=localhost` and `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`.

`from_env` returns None when the required vars are absent, so callers can degrade gracefully
(e.g. skip publishing / skip integration tests) instead of crashing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

# bronze/silver/gold objects all live under this bucket (created by minio-init).
DATASETS_BUCKET = "datasets"
DEFAULT_REGION = "us-east-1"


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: str
    user: str
    password: str
    db: str

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["PostgresConfig"]:
        env = env if env is not None else os.environ
        user, password, db = env.get("POSTGRES_USER"), env.get("POSTGRES_PASSWORD"), env.get("POSTGRES_DB")
        if not (user and password and db):
            return None
        return cls(
            host=env.get("POSTGRES_HOST", "localhost"),
            port=str(env.get("POSTGRES_PORT", "5432")),
            user=user,
            password=password,
            db=db,
        )


@dataclass(frozen=True)
class S3Config:
    endpoint_url: str
    access_key: str
    secret_key: str
    region: str = DEFAULT_REGION

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["S3Config"]:
        env = env if env is not None else os.environ
        endpoint = env.get("MLFLOW_S3_ENDPOINT_URL") or env.get("S3_ENDPOINT_URL")
        access, secret = env.get("AWS_ACCESS_KEY_ID"), env.get("AWS_SECRET_ACCESS_KEY")
        if not (endpoint and access and secret):
            return None
        return cls(
            endpoint_url=endpoint,
            access_key=access,
            secret_key=secret,
            region=env.get("AWS_REGION", DEFAULT_REGION),
        )
