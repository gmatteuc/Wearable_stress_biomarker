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

## 2026-01-18: Pipeline Harmonization, Label Remapping, and Safety Audit

**Objectives:**
- Ensure the deep learning pipeline matches the notebook logic for binary stress classification (Baseline vs Stress).
- Fix persistent confusion matrix annotation issues by harmonizing label remapping ([1,2] → [0,1]) in both pipeline and notebook.
- Guarantee all diagnostic plots and metrics are reproducible and correct.

**Actions Taken:**
1. Verified that the pipeline and notebook both filter for binary classification and remap labels consistently.
2. Updated `src/models/train.py` to explicitly remap labels from [1,2] to [0,1] for binary classification, matching notebook logic.
3. Validated that all diagnostic plots (especially confusion matrices) now show correct annotations in all cells.
4. Performed a full codebase scan to ensure no downstream code expects the original labels ([1,2]) and that all modules (data, features, API, evaluation, visualization) are compatible with the remapped indices.
5. Confirmed that multiclass scenarios are handled generically and that the pipeline is robust to future class additions.

**Results:**
- Confusion matrix annotation bug resolved; all cells are now correctly labeled and annotated.
- No safety issues found: all downstream code uses remapped indices or class names, so the change is safe and robust.
- Pipeline and notebook are now fully harmonized for binary stress classification.

**Next Steps:**
- Continue to monitor for edge cases if new classes are added.
- Maintain strict notebook/pipeline alignment for reproducibility and portfolio quality.


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

    - **Deep Learning Upgrade (Jan 2026):**
        - Initial CNN performance (Simple 3-layer) was poor (~72% acc) compared to Feature Engineering (~86%).
        - **Architecture Overhaul**: Implemented `ResNet1D` with **Squeeze-and-Excitation (SE)** blocks in `src.models.deep`.
        - **Rationale**: ResNet connections allow deeper feature extraction, while SE blocks provide "Channel Attention," allowing the model to dynamically weight informative sensors (e.g., EDA vs ACC) per window.
        - **Result**: Accuracy improved to **94.6%** with F1-Stress **0.92**. Validated in `notebooks/04_deep_learning_verification.ipynb`.
        - **Learning Dynamics**: Observed higher variance in validation loss due to LOSO on small N (15 subjects), but final generalization is superior.


## 5. Current ML Progress (as of Jan 17, 2026)
- [x] **Feature Extraction**: Complete and reproducible for CHEST modality.
- [x] **Classical Baseline**: LOSO-validated logistic regression, with calibration and abstention analysis.
- [x] **Deep Learning**: ResNet1D + SE, with instance normalization, LOSO validation, and full diagnostic plots. All code refactored to use `ResNet1D` naming.
- [x] **Visualization**: All key plots (timeline, heatmap, learning curves, ROC, calibration, abstention) are auto-saved and standardized.
- [x] **Notebook Refactoring**: All ML logic now lives in scripts/modules; notebooks are for verification and reporting only.

## 6. MLOps Next Steps (CHEST, then WRIST)

**A. Pipeline Automation & Reproducibility**
- [ ] Refactor all notebook logic (data loading, feature extraction, training, evaluation) into scripts/functions in `src/`.
- [ ] Implement a CLI or single script (e.g., `python run_pipeline.py --modality chest`) to run the full pipeline and produce all artifacts/plots.
- [ ] Parameterize pipeline for CHEST/WRIST modality selection.

**B. Automated Reporting & Diagnostics**
- [ ] Ensure all key diagnostic plots and metrics are auto-generated and saved to `reports/` or `notebooks/outputs/`.
- [ ] (Optional) Generate a minimal HTML/Markdown report summarizing results for each run.

**C. Deployment & Inference**
- [ ] Implement FastAPI endpoint for `/predict_window` (CHEST, then WRIST), including SQI/abstention logic.
- [ ] Add `/health` endpoint and minimal OpenAPI docs.

**D. Testing & CI**
- [ ] Expand unit tests to cover data, features, and model code.
- [ ] Add a CI workflow (e.g., GitHub Actions) for linting and tests.

**E. Documentation & Environment**
- [ ] Update README with clear instructions for running the pipeline, training, and serving the model.
- [ ] Export and pin `requirements.txt` or `environment.yml`.
- [ ] Add a “MLOps Checklist” section to README.

**F. (Optional) Experiment Tracking**
- [ ] Integrate a lightweight experiment tracker (e.g., MLflow, wandb, or CSV logs).

**G. (Optional) Wrist Modality**
- [ ] Rerun the entire pipeline and reporting for WRIST data, using the same structure and diagnostics as CHEST.

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

## 2026-01-17: MLOps Refactoring of Deep Learning Verification

**User Request:** Refactor 
otebooks/04_deep_learning_verification.ipynb to match the 'MLOps standard' of 
otebooks/03_model_fitting_verification.ipynb.
- Move inline plotting code to src/visualization/plots.py.
- Use the pipeline (src.models.train.Trainer) instead of reimplementing training loops.
- Use standard evaluation plots (plot_confidence_abstention_panel) for results.
- Simplify markdown.

**Plan:**

1.  **Library Enhancement (src/visualization/plots.py)**:
    -   Add plot_timeline_segmentation: specific to the gap analysis of WESAD.
    -   Add plot_multiscale_heatmap: specific to the macro/micro feature view.

2.  **Notebook Refactoring (
otebooks/04_deep_learning_verification.ipynb)**:
    -   **Section 1: Data Integrity Verification**: Load raw data, generate the Timeline and Heatmap plots using the new library functions.
    -   **Section 2: Pipeline Execution**: Instantiate Trainer and run the full 	rain_deep pipeline (LOSO or Single Split).
    -   **Section 3: Performance Audit**: Load the artifacts (predictions.csv) generated by the pipeline and visualize them using plot_confidence_abstention_panel.

## 2026-01-17: GPU Acceleration Support

**Status:** IN PROGRESS (Blocked by Bandwidth)

**Issue:** Training 1D-CNN is significantly slow.
**Diagnosis:** 
- `torch.cuda.is_available()` returns `False`.
- Installed PyTorch version is `2.0.1+cpu`.
- Hardware check (`nvidia-smi`) confirms **NVIDIA RTX 2000 Ada Generation** is available and idle.

**Action Taken:**
1.  Updated `src.models.train.Trainer` to support device-agnostic training (`.to(device)`).
2.  Updated Notebook 04 to include explicit GPU verification steps.
3.  Attempted to reinstall PyTorch with CUDA 11.8 support:
    `pip install torch --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu118`

**Blocker:**
- The required wheel (~2.8GB) is too large for the current internet connection speed.
- **Decision:** Paused installation. Will resume when better bandwidth is available.

**Next Steps:**
- [x] Run the pip install command again on a faster network (pip install torch --upgrade     --force-reinstall --index-url https://download.pytorch.org/whl/cu118).
- [x] Confirm `torch.cuda.is_available() == True` in `04_deep_learning_verification.ipynb`.
- [x] Benchmark the speed improvement.

## 5. Deep Learning Pipeline Refactoring (Jan 17 2026 - Afternoon)

### Objectives
- Establish an industry-standard Deep Learning (DL) experimentation loop.
- Compare DL performance (Raw Signals) against Baseline (Feature Engineering).
- Address early overfitting and subject-specific generalization failures.

### Actions Taken
1.  **Refactoring**: 
    - Moved model architecture to `src/models/deep.py`.
    - Integrated PyTorch training loop into `src/models/train.py` (Trainer Class).
2.  **Infrastructure Refactoring**:
    - Fixed GPU utilization issues by forcing a clean install of PyTorch (CUDA 12.4) via `Makefile` (`setup-gpu` target).
3.  **Model Engineering (The "S2 Fix")**:
    - **Issue**: Initial model (72% acc) failed completely on Subject 2 (Dry skin, low amplitude EDA). Global Normalization crushed S2's signal to constant negative values.
    - **Solution**: Implemented **Instance Normalization** (Per-Window Mean/Std subtraction). This allows the model to learn morphology independent of absolute DC offset.
    - **Regularization**: Added Dropout (0.2), Weight Decay (1e-3), and Label Smoothing (0.1) to combat the small dataset size (~1k samples).
    - **Architecture**: Increased Receptive Field (Kernel 5->15) to capture physiological rise times (e.g., SCR onset) rather than just high-frequency texture.

### Next Steps
- [ ] **Hyperparameter Tuning**: Run a systematic sweep (Optuna?) on the stabilized 1D-CNN.
- [ ] **Architecture Exploration**: Test DeepConvLSTM or Attention-based heads to better capture temporal dependencies over the 60s window.

## 2026-01-17: Final Deep Learning Optimization and Analysis

- **Longer Training & Early Stopping Tuning:**
    - Increased training epochs to 50 and set early stopping patience to 15 in `src/models/train.py`.
    - Rationale: Allow the model to converge more fully, especially on difficult LOSO folds, without overfitting.
    - Result: Validation loss and accuracy curves show continued improvement and stabilization, with no overfitting observed.
    - Individual subject curves (thin lines) and mean curves (thick lines) are now clearly visualized in the learning curves, showing both subject variability and robust average performance.

- **Final Results:**
    - **Global accuracy**: ~95% (LOSO, subject-normalized)
    - **Calibration**: Model remains well-calibrated (reliability diagram matches ideal).
    - **Abstention**: Model abstains more on the most difficult subjects, as expected.
    - **No further overfitting**: Training and validation losses both decrease and plateau, and validation accuracy remains high.

- **Conclusion:**
    - The model is now fully optimized for the WESAD dataset (CHEST), with both high accuracy and reliability.
    - Further performance gains are likely to be marginal and may risk overfitting or reduce generalizability.
    - The pipeline is ready for deployment, monitoring, and documentation.

## 5. Visualization Consistency Fixes (Jan 18, 2026)
- [x] **Confusion Matrix Annotation Bug**:
    - **Issue**: Pipeline-generated confusion matrices appeared to be missing annotations for the "Stress" class (bottom row).
    - **Root Cause**: Two-fold issue:
        1. `confusion_matrix` omitted rows/cols for missing classes (Fixed by `labels=[0, 1]`).
        2. **Contrast Issue**: Text color was defaulting to Black on Dark Purple backgrounds in the non-interactive backend, making annotations invisible.
    - **Fix**: Replaced seaborn's internal annotation with a custom `_add_heatmap_annotations` function in `src/visualization/plots.py`. It explicitly forces White text for any cell with intensity > 30%, ensuring visibility in all environments.
- [x] **Column Naming Standardization**:

    - Confirmed that `src/models/train.py` handles column renaming (`y_true` -> `true`, `prob_stress` -> `prob`) before passing data to plotting functions, ensuring compatibility with `src/visualization/plots.py`.




