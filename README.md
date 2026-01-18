# Wearable Stress Detection with Quality-Aware MLOps

## Project Status
**Active Development** - Phase: MLOps & Deployment (CHEST Modality)

## Key Features
- **Deep Learning**: ResNet-1D architecture optimized for physiological time-series (CHEST modality).
- **Explainability**: Instance Normalization visualization to audit model inputs.
- **REST API**: FastAPI implementation for real-time stress prediction (`src/api/app.py`).
- **MLOps**: Automated Data Drift detection (Covariate Shift) to monitor population shifts (`src/monitoring/drift_report.py`).
- **Reproducibility**: End-to-end consistency between Training and Inference logic.

## Overview
This project demonstrates a production-grade Machine Learning Engineering pipeline for detecting stress using multi-sensor wearable data (WESAD dataset). 

**Key Goals:**
1. **Realistic Domain Shift:** Evaluation using **Leave-One-Subject-Out (LOSO)** validation to ensure the model generalizes to new people.
2. **Reliability:** Implementation of **Signal Quality Indices (SQI)** to filter noisy data and **Calibration** to provide trustworthy probability estimates.
3. **MLOps Engineering:** Reproducible pipeline, experiment tracking, drift monitoring, and a deployment-ready FastAPI endpoint packaged with Docker.

## Dataset
We use the **WESAD (Wearable Stress and Affect Detection)** dataset. 
- **Modalities Used:** Chest-worn ACC, ECG, EDA, EMP, RESP, TEMP.
- **Classes:** Baseline (Neutral), Stress, Amusement.
- **Preprocessing:** 
  - Validated and unzipped automatically.
  - Resampled to 35 Hz.
  - Segmented him into 60s windows with 50% overlap.

## Project Structure
```
.
├── configs/            # Configuration (YAML)
├── data/               # Raw and processed data
├── src/                # Source code
│   ├── api/            # FastAPI deployment
│   ├── data/           # ETL pipeline
│   ├── features/       # SQI and Feature Engineering
│   ├── models/         # Training and Deep Learning modules
│   └── monitoring/     # Drift detection
├── tests/              # Unit tests
├── Makefile            # Command shortcuts
└── README.md           # Documentation
```

## Setup

1. **Environment:**
   ```bash
   make setup
   ```

2. **Data:**
   Place `WESAD.zip` in `data/` (if not already present).
   ```bash
   make download    # Validates and unzips
   make preprocess  # Parsing, Resampling, Windowing (to parquet)
   ```

3. **Feature Engineering:**
   ```bash
   make features    # Extracts statistical features for classical models
   ```

## Usage

### Training
Train the classical baseline (Logistic Regression) with LOSO split:
```bash
make train-baseline
```
Train the Deep 1D-CNN model:
```bash
make train-deep
```
Artifacts (models, metrics, plots) are saved in `reports/`.

### Deployment
Run the API locally:
```bash
make run-api
```
Test the endpoint (Swagger UI at http://localhost:8000/docs):
```json
POST /predict_window
{
  "EDA": [0.5, 0.51, ...],
  "ACC_x": [0.01, 0.02, ...], 
  ...
}
```

### Docker
```bash
docker build -t outcomes/stress-detection .
docker run -p 8000:8000 outcomes/stress-detection
```

### Monitoring
Check for data drift between training reference and new batch:
```bash
# Example python script usage (if applicable) or refer to Notebook 05
python -m src.monitoring.drift_report --ref data/processed/train.parquet --curr data/processed/batch_01.parquet
```
See `notebooks/05_inference_demo.ipynb` for the interactive Drift Dashboard.
```bash
python -m src.monitoring.drift_report --reference data/processed/features.parquet --current data/new_batch.parquet
```

## Methodological Details

### Signal Quality (SQI)
We implement a rule-based SQI layer (`src.features.sqi`) that checks for:
- Signal completeness
- Flatline artifacts (sensor disconnect)
- High-intensity motion (via Accelerometer)

If SQI is below threshold, the API returns an `abstain` flag to prevent unreliable predictions.

### Calibration
Models are wrapped in `CalibratedClassifierCV` (Platt Scaling) to ensuring that a predicted probability of 0.7 truly corresponds to a 70% chance of stress. We evaluate this using ECE (Expected Calibration Error).

### Evaluation
We strictly enforce subject-disjoint splits. Standard random splits overestimate performance in wearable biometrics due to individual physiological uniqueness.

## Next Steps
- Implement HR/HRV feature extraction from ECG/BVP.
- Add MC Dropout quantification of epistemic uncertainty for the Deep model.
- Integrate MLflow for experiment tracking server.
