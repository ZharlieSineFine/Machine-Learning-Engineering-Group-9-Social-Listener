import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text

# The monitoring package is mounted at /opt/project/monitoring in the Airflow
# image (see docker-compose volumes), which is not on sys.path by default.
_PROJECT_ROOT = '/opt/project'
if os.path.isdir(_PROJECT_ROOT) and _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def greet():
    print('Howdy fellas')


def _engine():
    """Build a SQLAlchemy engine for the app Postgres DB from env vars.

    Avoids needing the apache-airflow-providers-postgres package or a
    configured Airflow connection — SQLAlchemy + psycopg2 are already in
    the image, and the POSTGRES_* vars come from .env via env_file.
    """
    user = os.environ['POSTGRES_USER']
    password = os.environ['POSTGRES_PASSWORD']
    host = os.environ.get('POSTGRES_HOST', 'postgres')
    port = os.environ.get('POSTGRES_PORT', '5432')
    db = os.environ['POSTGRES_DB']
    return create_engine(
        f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'
    )


def testing():
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reviews (
                id SERIAL PRIMARY KEY,
                review_text TEXT,
                created_at TIMESTAMP DEFAULT now()
            );
        """))
        conn.execute(
            text('INSERT INTO reviews (review_text) VALUES (:review_text);'),
            {'review_text': 'dummy review from phase 1'},
        )
    print('Inserted dummy row!')


def drift_check():
    """Phase 1 Evidently wiring smoke task.

    Runs the DataDriftPreset stub from ``monitoring/drift_checks.py`` on the
    sample CSV vs. itself (no train data exists yet), so the monitoring slice
    of the pipeline is exercised end-to-end inside Airflow. Always passes by
    design — Phase 2 swaps in a real reference vs. last-7d current frame.
    """
    from monitoring.drift_checks import run_drift_check

    result = run_drift_check()
    print(f'Drift check ran: {result}')
    # Phase 2: append this row to the `monitoring_reports` table.


with DAG(
    dag_id='test',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['phase1', 'test'],
) as dag:

    hello = PythonOperator(task_id='greet', python_callable=greet)
    dummy = PythonOperator(task_id='testing', python_callable=testing)
    drift = PythonOperator(task_id='drift_check', python_callable=drift_check)

    hello >> dummy >> drift
