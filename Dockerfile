FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Create dummy file to setup dependencies
COPY src/ /app/src/
COPY configs/ /app/configs/
# Install dependencies
RUN pip install .

# Copy source code and artifacts
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
