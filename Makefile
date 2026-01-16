.PHONY: setup download preprocess train test lint run-api clean

setup:
	pip install -e .[dev]

download:
	# Expects the WESAD.zip to be in data/ or user has to put it there manually as we can't automate login-walled downloads easily without credentials
	python -m src.data.validate_raw

preprocess:
	python -m src.data.make_dataset

features:
	python -m src.features.build_features

train-baseline:
	python -m src.models.train --model logistic

train-deep:
	python -m src.models.train --model deep

evaluate:
	python -m src.models.evaluate

test:
	pytest tests/

lint:
	ruff check .
	ruff format .

run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
