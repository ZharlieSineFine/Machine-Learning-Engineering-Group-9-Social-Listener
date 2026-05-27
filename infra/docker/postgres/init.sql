-- Bootstrap databases on first start of the postgres container.
-- (POSTGRES_DB="sentiment" is already created by the image.)

CREATE DATABASE airflow;
CREATE DATABASE mlflow;
