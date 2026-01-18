FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# 1. Copy core source and configs first to cache dependencies
COPY src/ /app/src/
COPY configs/ /app/configs/

# 2. Install dependencies (including the project itself in editable mode or standard)
RUN pip install --no-cache-dir .

# 3. Copy the rest of the project (Crucially: 'reports/' containing trained models)
# Note: Data and Notebooks are excluded via .dockerignore
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
