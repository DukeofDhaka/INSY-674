VENV ?= .venv
PYTHON ?= $(shell if [ -x $(VENV)/bin/python ]; then echo $(VENV)/bin/python; elif command -v python3.11 >/dev/null 2>&1; then echo python3.11; else echo python3; fi)

.PHONY: bootstrap check-python install train test run-api build-package docker-build

bootstrap:
	python3.11 -m venv --clear $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements/research.txt

check-python:
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ required (recommended 3.11). Run make bootstrap to recreate .venv.")'

install: check-python
	$(PYTHON) -m pip install -r requirements/research.txt

train: check-python
	$(PYTHON) -m src.train_pipeline

test: check-python
	$(PYTHON) -m pytest app-fastapi/app/tests src/monitoring/test_drift.py -q

run-api: check-python
	$(PYTHON) -m uvicorn app.main:app --app-dir app-fastapi --host 0.0.0.0 --port 8000 --reload

build-package: check-python
	$(PYTHON) -m build

docker-build:
	docker build -t insy674-ml-api .
