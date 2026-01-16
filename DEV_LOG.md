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

- [x] **Feature Verification**: `notebooks/02_feature_extraction_verification.ipynb` executed successfully.
    - Verified EDA Phasic (SCR) detection.
    - Verified ECG R-peak detection at 35Hz.
    - Confirmed feature separability (EDA Mean, SCR Count, Temp Slope) for Baseline vs Stress.
- [x] **Feature Engineering**: Updated `src/features/feature_extraction.py` with `process_windows` method for batch processing.
- [x] **Visualization Refactoring**: Moved plotting logic to `src/visualization/plots.py` with consistent Deep Purple/Orange palette, removing ad-hoc green plots and using Violin+Strip plots for distribution analysis.


## 4. Evaluation Harmonization & MLOps Pipeline (Jan 2026)

- **Evaluation Harmonization Plan:**
    - Refactor all metric computation and evaluation logic into `src/models/evaluate.py` (`evaluate_model`).
    - `evaluate_model` will compute and return (or save) all metrics and intermediate results needed for advanced notebook plots:
        - Per-sample predictions, probabilities, and true labels (for abstention/confidence analysis)
        - Per-subject results (for subject-level plots)
        - All confusion matrices, ROC curves, calibration data, etc.
    - The notebook will call `evaluate_model` and use only its outputs for all diagnostics and plots, never recomputing metrics inline.
    - Plotting functions in `src/visualization/plots.py` will only visualize, not compute metrics.
    - This ensures the notebook is a true test of the deployed pipeline, with no code drift or logic duplication.

- **Recent Improvements:**
    - Unified all heavy plotting code in `plots.py` (confidence, abstention, calibration, subject-level accuracy, etc.).
    - Added error bars and improved color consistency in diagnostic plots.
    - Subject accuracy/abstention plots now use consistent subject colors across all subplots.
    - All advanced diagnostics (abstention, calibration, subject-level) are preserved and harmonized with the pipeline outputs.


## 4. Recent Debugging & Outstanding Issues (Jan 2026)


    - **EDA SCR Detection Improvements (Jan 2026):**
        - Switched EDA SCR peak detection from a fixed height threshold to a prominence-based approach (`scipy.signal.find_peaks` with `prominence`).
        - Tuned prominence threshold to 0.015 to balance sensitivity and noise rejection.
        - Result: Fewer spurious SCRs on noisy/ripple sections, but more true SCRs detected after large events and in low-baseline regions.
        - Visualization: SCR dots are now semi-transparent with a white outline, making it easier to see the underlying phasic curve and event alignment.
        - Rationale: Prominence-based detection is more robust to local baseline shifts and avoids missing SCRs that occur after large peaks or in negative-going phasic regions.
        - Confirmed visually in `02_feature_extraction_verification.ipynb` that SCR events now align better with physiological responses and are less affected by noise.

    - **Harmonic Correction in Respiration (Jan 2026):**
        - Issue: Respiration logic was detecting 20bpm (2nd harmonic) instead of true 10-12bpm.
        - Fix: Implemented multi-peak validation checking 1/2 and 1/3 sub-harmonics in `src/features/feature_extraction.py`.
        - Verified: Notebook 02 confirmed correct 12bpm detection vs manual inspection.

    - **Model Verification (Jan 2026):**
        - Created `notebooks/03_model_verification.ipynb`.
        - Implemented **Leave-One-Subject-Out (LOSO)** validation.
        - **Result**: Logistic Regression Baseline achieves **F1-Stress = 0.82**.
        - Calibration: ECE = 0.14 (slightly overconfident).
        - Status: Pipeline validated end-to-end on full dataset (S2-S17).

    - **Respiration & Temperature Audit (Jan 2026):**
        - **Problem:** Respiration rate extraction using simple Welch peak detection was picking up 2nd harmonics (e.g., 20bpm instead of actual 10-12bpm).
        - **Fix:** Implemented sophisticated peak logic: searches for peaks >= 20% max power, then checks for presence of fundamental frequency at ~1/2 or ~1/3 of the dominant peak.
        - **Verification:** Confirmed on Subject S2 that rate dropped from 20bpm (incorrect harmonic) to 12bpm (physiologically correct baseline).
        - **Temperature:** Verified the linear slope Feature correctly captures the global trend (acute vasoconstriction) despite local sensor noise. Updated auditing visualization to use the *actual* calculated feature slope rather than re-computing it, ensuring strict verification.

    - **Pipeline Refactoring (Jan 2026):**
        - Refactored `src/models/evaluate.py` to return granular results (DataFrame) enabling advanced diagnostics in notebooks without code duplication.
        - Updated `src/models/train.py` (Trainer) to pass subject IDs during evaluation and apply MVP binary filtering (Baseline vs Stress) consistent with verification goals.
        - **Critical Fixes**:
            - **Data Leakage**: Identified and fixed leakage where `start_idx` was inadvertently included as a feature. Explicitly excluded metadata columns in `Trainer`.
            - **Hyperparameters**: Restored `class_weight='balanced'` and `solver='liblinear'` in the production pipeline to match the validated notebook baseline.
            - **Validation Logic**: Implemented full `LeaveOneGroupOut` cross-validation in `train.py` to replicate the "Train on N-1, Test on 1" strategy exactly.
        - Refactored `notebooks/03_model_verification.ipynb` to use the production `Trainer` class directly, ensuring the notebook validates the *actual* deployed code path.

## 5. Next Steps
1.  [x] **Finalize Feature Extraction**:
    - Run `python -m src.features.build_features` to generate `data/processed/features.parquet`.
2.  [x] **Model Baseline & MLOps Verification**:
    - Create `notebooks/03_model_verification.ipynb`.
    - Implemented strict **Leave-One-Subject-Out (LOSO)** validation.
    - Verified **Calibration** (Reliability Diagrams) and **Abstention Policy**.
    - **Status**: Logistic Regression Baseline confirmed (F1-Stress ~0.90 after leakage fix).
3.  [ ] **Deep Learning Model (CNN)**:
    - The `train_deep` method exists in `train.py` but uses a basic `Simple1DCNN`.
    - **Action**: Create `notebooks/04_deep_learning.ipynb` (or extend 03) to verify the Deep Learning pipeline on Raw Windows.
    - **Comparison**: Benchmark Deep Learning (Raw Signals) vs. Classical (Features).
4.  [ ] **Documentation & Deployment**:
    - Finalize `README.md` with reproduction steps.
    - Export `requirements.txt`.
        - **Artifact Versioning:** Saving `scaler.joblib`, `model.pkl`, and `metrics.json` clearly in `models/` folder.
        - **Traceability:** Logging specific commit hash or experiment ID.
3.  [ ] **Training Pipeline**:
    - Create `src/models/train.py` implementing the logic verified in the notebook.

## 6. Next Steps: Deep Learning (Jan 2026)

- **Planned Implementation:**
    - Build a Deep MLP (Multi-Layer Perceptron) in PyTorch for tabular feature classification.
    - Integrate into `src/models/deep.py` and add support in `Trainer` for model_type='deep'.
    - Structure: Input (feature vector) → Dense layers (ReLU, Dropout) → Output (binary stress prediction).
    - Train and evaluate using the same LOSO cross-validation pipeline for direct comparison with Logistic Regression and Random Forest.

- **Rationale:**
    - Our features are window-based and tabular (not raw signals), so MLP is the most appropriate deep architecture.
    - CNNs are reserved for future work if we move to raw signal input.
    - Goal: Assess if deep learning can outperform the current 86% accuracy baseline and provide more flexible modeling.

## 7. How to Resume
If you rename the project folder, the VS Code Chat history will be lost.
**Refer to this file** to restore context for the AI. You can simply ask:
*"Read DEV_LOG.md and help me with the next step."*

## 5. How to Resume
If you rename the project folder, the VS Code Chat history will be lost.
**Refer to this file** to restore context for the AI. You can simply ask:
*"Read DEV_LOG.md and help me with the next step."*

- [x] **Visualization System Upgrade (Jan 2026):**
    - **Auto-Save Implementation**: Updated all 14 plotting functions in `src/visualization/plots.py` to accept `save_folder`. Plots are now automatically saved to `notebooks/outputs/{LOCATION}/` (e.g., CHEST/WRIST) with sanitized filenames.
    - **Configuration Integration**: Removed hardcoded sampling rates (`fs=35`) and thresholds (`scr_threshold=0.015`). These are now centrally managed in `configs/default.yaml` and injected dynamically into notebooks and backend code.
    - **Orphan Cleanup**: Removed redundant plotting functions (e.g., `plot_resampling_comparison`) to maintain a clean codebase.
    - **Notebook Standardization**: Updated Notebooks 01, 02, and 03 to use the new auto-save logic and dynamic configuration. Verified end-to-end execution of `03_model_fitting_verification.ipynb` (LOSO Validation).
