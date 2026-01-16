# Development Log & Project Context

**Last Updated:** January 16, 2026
**Project:** Wearable Stress Biomarker (WESAD)

## 1. Project Overview
This project targets stress detection using the WESAD dataset (Chest data: EDA, ECG, Temp, ACC).
We are building an "Engineering Grade" pipeline with strict quality controls (SQI) before machine learning.

## 2. Current Progress Status
- [x] **Data Ingestion**: `WESADPreprocessor` created. Handles loading and windowing (60s windows, 30s overlap).
- [x] **Pipeline Verification**: `notebooks/01_pipeline_verification.ipynb` validates the data loading and windowing.
- [x] **Signal Quality Index (SQI)**: `SignalQualityIndex` module implemented.
    - **Dead Sensor Check**: Range thresholds (e.g., Temp < 0.01 change).
    - **Noise Check**: Motion variance (ACC > 0.5g).
- [x] **Config Centralization**: Params moved to `configs/default.yaml`.
- [x] **Feature Extraction**: `src/features/feature_extraction.py` implemented.
    - **EDA**: Phasic (SCR peaks) vs Tonic (Low-pass) decomposition.
    - **ECG**: Bandpass (1-15Hz) + R-Peak detection -> HRV metrics.
    - **Temp**: Slope calculation for stress trends.
- [x] **Feature Audit**: `notebooks/02_feature_extraction.ipynb` created to visually debug features.

## 3. Key Technical Decisions
1.  **Sampling Rate**: Downsampling all signals to **35Hz** for ML consistency.
2.  **SQI Thresholds**:
    - Temperature "Flatline" set to 0.01 (uC) instead of 0.0 to account for sensor precision.
    - ACC Noise threshold at 0.5g std dev.
3.  **Preprocessing**:
    - **Detrending**: REJECTED for EDA/Temp to preserve baseline stress shifts (Tonic levels are important features).
    - **Filtering**: EDA Tonic = Low-pass (<0.05Hz). ECG = Bandpass (1-15Hz).

## 4. Next Steps
1.  Open `notebooks/02_feature_extraction.ipynb`.
2.  Run the notebook to verify that the extracted features (EDA Peaks, ECG R-peaks) look correct visually.
3.  Analyze the "Feature Separability" plots at the end of the notebook to confirm features distinguish Stress vs Baseline.
4.  If verified, proceed to build the full dataset `src/data/build_features.py`.

## 5. How to Resume
If you rename the project folder, the VS Code Chat history will be lost.
**Refer to this file** to restore context for the AI. You can simply ask:
*"Read DEV_LOG.md and help me with the next step."*
