PYTHON ?= python3

.PHONY: install train test run-api build-package docker-build

install:
	$(PYTHON) -m pip install -r requirements/research.txt

train:
	$(PYTHON) -m src.train_pipeline

test:
	pytest app-fastapi/app/tests -q

run-api:
	uvicorn app.main:app --app-dir app-fastapi --host 0.0.0.0 --port 8000 --reload

build-package:
	$(PYTHON) -m build

docker-build:
	docker build -t insy674-ml-api .
