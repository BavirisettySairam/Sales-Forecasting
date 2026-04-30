.PHONY: setup train train-states train-one serve dashboard test lint format clean ci-lint ci-test ci

PYTHON := gcc_env/python.exe

setup:
	poetry install
	poetry run pre-commit install

# Train all states sequentially (use WORKERS=N for parallel, e.g. make train-states WORKERS=4)
train-states:
	$(PYTHON) -m src.pipeline.train --data data.csv --all-states --config config/training_config.yaml --cv-splits 3

train: train-states

# Train a single state: make train-one STATE="California"
train-one:
	$(PYTHON) -m src.pipeline.train --data data.csv --state "$(STATE)" --config config/training_config.yaml --cv-splits 3

serve:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run src/dashboard/app.py --server.port 8501

test:
	poetry run pytest tests/ -v --tb=short

lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/

format:
	poetry run ruff check --fix src/ tests/
	poetry run black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name "*.lock" -path "*/models/*" -delete

ci-lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/

ci-test:
	poetry run pytest tests/ -v --tb=short --junitxml=test-results.xml

ci: ci-lint ci-test
