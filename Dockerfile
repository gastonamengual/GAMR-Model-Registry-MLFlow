FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip \
    psycopg2-binary \
    mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
    "--backend-store-uri", "postgresql://mlflow_database_user:GexY8t4Wq656Rj8OUD7spq4DbLvu1eLP@dpg-cv84vnaj1k6c73bk2f40-a.frankfurt-postgres.render.com/mlflow_database", \
    "--host", "0.0.0.0", \
    "--port", "5000"]

# docker build -t mlflow-server .
# docker run -p 5000:5000 mlflow-server
