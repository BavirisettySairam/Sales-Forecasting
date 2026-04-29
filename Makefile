.PHONY: setup train serve test lint clean docker-up docker-down

setup:
	poetry install
	poetry run pre-commit install
	poetry run alembic upgrade head

train:
	poetry run python -m src.pipeline.train --config config/training_config.yaml

serve:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	poetry run streamlit run src/dashboard/app.py --server.port 8501

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
	rm -rf models/*.pkl models/*.pt

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v

docker-logs:
	docker-compose logs -f

ci-lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/

ci-test:
	poetry run pytest tests/ -v --tb=short --junitxml=test-results.xml

ci: ci-lint ci-test
