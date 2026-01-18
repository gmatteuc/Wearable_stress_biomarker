.PHONY: setup setup-gpu download preprocess train test lint run-api clean

setup:
	pip install -e .[dev]

setup-gpu:
	@echo "Installing Production Dependencies..."
	pip install -e .[dev]
	@echo "Overwriting PyTorch with CUDA 12.4 enabled version..."
	pip uninstall -y torch torchvision torchaudio
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


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

docker-build:
	docker build -t outcomes/stress-detection .

docker-run:
	docker run -p 8000:8000 outcomes/stress-detection

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
