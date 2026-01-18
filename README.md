# Wearable Stress Biomarker & MLOps Pipeline

<img src="misc/dataset-cover.png" width="800">

## Project Overview
This project implements an end-to-end "Engineering Grade" Machine Learning pipeline to detect physiological stress from multi-modal wearable sensor data. Typically, physiological signal processing is confined to research notebooks; here, I demonstrate how to bridge the gap between biomedical research and production-ready MLOps.

The system processes raw biosignals (ECG, EDA, Respiration, Temperature, Accelerometry) to classify the user's state as **Baseline** or **Stress** in real-time.

**Use Case Scenario:**
*Consider an occupational safety system for high-stakes professions (e.g., pilots, first responders). The goal is to monitor physiological stress benchmarks in real-time, flagging cognitive overload before performance degrades or safety `is compromised. This requires not just high accuracy, but robust generalization to new users and explainable reliability metrics.*

**Key Value Points:**
1.  **Rigorous Validation**: Implements **Leave-One-Subject-Out (LOSO)** cross-validation to ensure the model generalizes to unseen individuals (preventing the "identity leakage" common in amateur biomedical AI).
2.  **Deep Learning Architecture**: A custom **ResNet-1D** with Squeeze-and-Excitation blocks to learn morphological features directly from raw time-series, outperforming traditional feature engineering.
3.  **Full MLOps Lifecycle**: Includes data versioning, Signal Quality Indices (SQI), Model Drift detection, and a FastAPI deployment endpoint.

Created by [Your Name] in 2026.

> **Development Note**: The current pipeline is fully validated and tested on the **CHEST** sensor modality (High-fidelity ECG, Chest EDA). The infrastructure currently supports the **WRIST** modality (PPG/BVP), but specific validation benchmarks and hyperparameter tuning for wrist-based signals are currently in development.

## Dataset
The project utilizes the **WESAD (Wearable Stress and Affect Detection)** dataset.
- **Subjects**: 15 participants in a controlled lab study.
- **Signals**: 
  - **Chest**: ECG (700Hz), EDA, EMG, Respiration, Temperature, Accelerometer.
  - **Wrist**: BVP (64Hz), EDA, Skin Temperature.
- **Context**: Three affective states: Neutral (Baseline), Stress (TSST protocol), Amusement.

## Methodology

### 1. Preprocessing & Signal Quality
Raw sensor streams are messy. The pipeline implements:
- **Resampling**: Harmonizing all signals to 35Hz for deep learning ingestion.
- **Windowing**: Sliding vectors of 60 seconds with 50% overlap.
- **Signal Visualization**: Automated generation of signal snapshots to verify integrity.

<img src="misc/Subject_S2_-_Raw_Chest_Signals_60s_snapshot_example.png" width="600">
*Example of a 60-second window of raw physiological signals from the Chest sensor.*

### 2. Machine Learning Modeling
Two modeling approaches were compared to establish a robust benchmark:
- **Baseline (Classical ML)**: Logistic Regression on statistically engineered features (Mean, Std, Peaks, Dynamic Range).
- **Deep Learning (ResNet-1D)**: A 1D Convolutional Neural Network that learns representations directly from the raw `(Channels x Time)` tensor.

### 3. Evaluation Strategy
We strictly adhere to a **Leave-One-Subject-Out (LOSO)** protocol. If we randomly split windows, the model would learn the subject's unique heart rate, not "stress". LOSO ensures we test on a person the model has never seen before.

### 4. MLOps & Deployment
- **FastAPI**: A robust REST API for real-time inference.
- **Drift Monitoring**: Statistical tests (Kolmogorov-Smirnov) to detect when incoming data deviates from the training distribution (Covariate Shift).

## Key Findings
- **Performance**: The Deep Learning approach significantly outperforms the classical baseline (Acc **~96%** vs **~86%**).
- **Why?**: The CNN captures complex local morphologies (e.g., the specific slope of an EDA reaction or R-peak interval variability) that global statistical aggregations miss.
- **Reliability**: We analyze prediction confidence histograms to identify "ambiguous" zones where the model should abstain from predicting.

<img src="misc/deep_Confidence_Abstention_example.png" width="800">
*Reliability Audit: Analyzing model confidence and calibration to prevent silent failures.*

## 💻 Project Structure
```
├── configs/            # YAML configuration for experiments
├── data/               # Data management (Raw vs Processed)
├── notebooks/          # Verification & Demo Notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 04_deep_learning_verification.ipynb
│   └── 05_inference_demo.ipynb
├── reports/            # Training artifacts (Models, Logs, Plots)
├── src/
│   ├── api/            # FastAPI application
│   ├── data/           # ETL & Validation logic
│   ├── features/       # SQI & Feature Extraction
│   ├── models/         # PyTorch (ResNet) & Scikit-Learn logic
│   └── monitoring/     # Drift detection modules
├── tests/              # Unit tests
├── Dockerfile          # Container definition
├── Makefile            # Automation
└── README.md           # Documentation
```

## ⚙️ Installation & Usage

1. **Environment Setup**:
   The project uses `conda` and `pip` via a Makefile for reproducible setup.
   ```bash
   make setup
   ```

2. **Data Pipeline**:
   Download and process the raw WESAD data (assumes WESAD.zip is in `data/raw`).
   ```bash
   make preprocess 
   ```

3. **Train Models**:
   Execute the full training pipeline (Baseline + Deep):
   ```bash
   make train-baseline
   make train-deep
   ```

4. **Run API**:
   Launch the inference server locally.
   ```bash
   make run-api
   ```
   Access the Swagger UI at `http://localhost:8000/docs`.

## Dependencies
- **Core**: `pandas`, `numpy`, `scipy`
- **Deep Learning**: `torch`, `torchvision` (1D ResNet adaptation)
- **ML**: `scikit-learn`, `joblib`
- **Deployment**: `fastapi`, `uvicorn`, `docker`
- **Visualization**: `matplotlib`, `seaborn`
