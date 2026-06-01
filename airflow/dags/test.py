import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text


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


with DAG(
    dag_id='test',
    schedule='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['phase1', 'test'],
) as dag:

    hello = PythonOperator(task_id='greet', python_callable=greet)
    dummy = PythonOperator(task_id='testing', python_callable=testing)

    hello >> dummy
