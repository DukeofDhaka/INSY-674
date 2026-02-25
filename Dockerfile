FROM python:3.11-slim

WORKDIR /app

COPY requirements /app/requirements
RUN pip install --no-cache-dir -r /app/requirements/deployment.txt

COPY . /app
RUN pip install --no-cache-dir -e .
RUN python -m src.train_pipeline

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --app-dir app-fastapi --host 0.0.0.0 --port ${PORT:-8000}"]
