"""
Wearable Stress Biomarker - Visualization Module
================================================

This module provides a unified styling and plotting interface for the project.
It adheres to a strict "Deep Purple & Orange" color theme for consistent,
publication-ready figures.

Functions:
    set_plot_style: Configures global matplotlib/seaborn defaults.
    plot_raw_signals: Visualizes synchronized signal modalities.
    plot_sqi_analysis: Visualizes signal quality metrics.
"""

import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import itertools
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
from pathlib import Path
from matplotlib.gridspec import GridSpec
import scipy.signal as signal
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)

def _save_plot(fig, title: str, save_folder: str = None):
    """
    Internal helper to save figures to notebooks/outputs/{save_folder}.
    """
    if not save_folder:
        return
        
    try:
        # Determine base directory
        cwd = Path.cwd()
        if cwd.name == 'notebooks':
            base_dir = cwd / 'outputs'
        elif (cwd / 'notebooks').exists():
            base_dir = cwd / 'notebooks' / 'outputs'
        else:
             base_dir = cwd / 'outputs'

        target_dir = base_dir / save_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize title
        safe_title = "".join([c if c.isalnum() or c in (' ', '_', '-') else '' for c in title]).strip()
        safe_title = safe_title.replace(' ', '_')
        if not safe_title:
             safe_title = "untitled_plot"
             
        filepath = target_dir / f"{safe_title}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {str(filepath)}")
        
    except Exception as e:
        logger.warning(f"Failed to save plot '{title}': {str(e)}")

def set_plot_style():
    """
    Sets the project-wide matplotlib and seaborn style.
    
    Theme:
        Primary: Deep Purple (#4B0082)
        Contrast: Dark Orange (#FF8C00)
    """
    plt.style.use('ggplot')
    
    # Custom Palette defined in design specs
    selected_colors = [
        "#4B0082",  # Indigo/Deep Purple (Primary)
        "#FF8C00",  # Dark Orange (Contrast)
        "#9370DB",  # Medium Purple
        "#FFA500",  # Orange
        "#BA55D3",  # Medium Orchid
        "#DAA520",  # Goldenrod (Darker Yellow)
    ]
    
    # Configure Cycle
    color_cycle = cycler('color', selected_colors)
    mpl.rcParams['axes.prop_cycle'] = color_cycle
    
    # Configure Seaborn
    sns.set_palette(sns.color_palette(selected_colors))
    
    # Typography & Lines
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['grid.alpha'] = 0.5
    
    return selected_colors

def _improve_heatmap_annotations(ax, cmap_bg='purple', threshold=0.3):
    """
    Deprecated: Use _add_heatmap_annotations instead for robust manual placement.
    """
    pass

def _add_heatmap_annotations(ax, data, fmt='d', threshold=0.3):
    """
    Manually adds text to heatmap cells. 
    Bypasses seaborn's annot=True which can be flaky (skipping rows).
    
    Args:
        ax: Matplotlib axes.
        data: The numerical matrix used for coloring.
        fmt: Format string ('d', '.2f') OR an numpy array/list of strings for custom labels.
        threshold: Intensity threshold (0-1) to switch to white text.
    """
    rows, cols = data.shape
    max_val = np.max(data) if np.max(data) > 0 else 1
    
    # Check if fmt is actually an annotation array (custom strings)
    annot_array = None
    if isinstance(fmt, (np.ndarray, list)):
        annot_array = fmt
        fmt = '' # disable formatting
        
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            
            # Determine Color
            # If data is essentially max-normalized (like recall), intensity ~ val
            # Otherwise (counts), intensity ~ val/max
            if max_val <= 1.05 and np.issubdtype(data.dtype, np.floating):
                intensity = val
            else:
                intensity = val / max_val
            
            # Force White on dark background (> threshold)
            color = 'white' if intensity > threshold else 'black'
            
            # Determine Text
            if annot_array is not None:
                text = str(annot_array[i][j])
            else:
                if fmt == 'd':
                    text = f"{int(val)}"
                elif fmt:
                    try:
                        text = format(val, fmt)
                    except:
                        text = str(val)
                else:
                    text = str(val)
            
            # Manually place text at center of cell
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color=color, fontweight='bold', fontsize=14)

def plot_confidence_abstention_panel(results_df, confidence_threshold=0.7, title_prefix="", save_folder: str = None):
    """
    Plots a 2x2 panel:
    (1) High-confidence confusion matrix
    (2) Probability distribution with abstention zone
    (3) Calibration plot (reliability diagram)
    (4) Abstention distribution across subjects
    """
    set_plot_style()
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix
    from sklearn.calibration import calibration_curve
    from collections import Counter

    DEEP_PURPLE = "#4B0082"
    DARK_ORANGE = "#FF8C00"
    # Revert to LinearSegmentedColormap for better contrast control
    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", DEEP_PURPLE])

    # Calculate confidence
    results_df = results_df.copy()
    results_df['confidence'] = results_df['prob'].apply(lambda x: max(x, 1-x))
    mask_confident = results_df['confidence'] >= confidence_threshold
    df_clean = results_df[mask_confident]
    df_abstained = results_df[~mask_confident]
    n_total = len(results_df)
    n_kept = len(df_clean)
    n_abs = n_total - n_kept

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) High-confidence Confusion Matrix
    # Always show both classes (0, 1) for binary stress classification
    cm_clean = confusion_matrix(df_clean['true'], df_clean['pred'], labels=[0, 1])
    acc_clean = (df_clean['true'] == df_clean['pred']).mean() if len(df_clean) > 0 else 0
    
    # Disable internal annot, use manual robust function
    sns.heatmap(cm_clean, annot=False, cmap=cmap_purple, ax=axes[0,0], cbar=True,
                xticklabels=['Baseline', 'Stress'],
                yticklabels=['Baseline', 'Stress'])
    _add_heatmap_annotations(axes[0,0], cm_clean, fmt='d', threshold=0.3)
    
    axes[0,0].set_title(f'{title_prefix}High Confidence CM - Acc: {acc_clean:.1%}\n(Removed {n_abs} uncertain samples)')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')

    # (2) Probability Distribution with Abstention Zone

    sns.histplot(results_df['prob'], bins=20, kde=True, ax=axes[0,1], color=DEEP_PURPLE, alpha=0.6)
    # Make abstention zone more orange and less transparent
    axes[0,1].axvspan(1-confidence_threshold, confidence_threshold, color=DARK_ORANGE, alpha=0.35, label='Abstention Zone')
    axes[0,1].axvline(0.5, color='gray', linestyle='--')
    axes[0,1].set_title(f'{title_prefix}Prediction Probability Distribution')
    axes[0,1].set_xlabel('prob')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()

    # (3) Calibration Plot (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(results_df['true'], results_df['prob'], n_bins=10)
    # Calculate SE for each bin (binomial SE)
    bin_counts = np.histogram(results_df['prob'], bins=np.linspace(0,1,11))[0]
    se = np.sqrt(prob_true * (1 - prob_true) / np.maximum(bin_counts, 1))
    axes[1,0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    axes[1,0].errorbar(prob_pred, prob_true, xerr=None, yerr=se, fmt='o', color=DEEP_PURPLE, label='Model Compliance', capsize=5, elinewidth=2)
    axes[1,0].set_xlabel("Mean Predicted Confidence")
    axes[1,0].set_ylabel("Fraction of Positives")
    axes[1,0].set_title(f'{title_prefix}Calibration Plot (Reliability Diagram)')
    axes[1,0].legend()

    # (4) Abstention Distribution Across Subjects
    abstention_counts = df_abstained['subject_id'].value_counts().sort_index()
    total_counts = results_df['subject_id'].value_counts().sort_index()
    abstention_frac = (abstention_counts / total_counts).fillna(0)
    subjects = abstention_frac.index.tolist()
    axes[1,1].bar(subjects, abstention_frac.values, color=DARK_ORANGE, alpha=0.8)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_ylabel('Fraction Abstained')
    axes[1,1].set_xlabel('Subject')
    axes[1,1].set_title(f'{title_prefix}Abstention Fraction by Subject')
    for i, v in enumerate(abstention_frac.values):
        axes[1,1].text(i, v + 0.02, f"{v:.0%}", ha='center', fontsize=10)

    plt.tight_layout()
    _save_plot(fig, title_prefix + "Confidence_Abstention", save_folder)
    return fig


def plot_raw_signals(
    time_axis: np.ndarray, 
    signals: Dict[str, np.ndarray], 
    title: str = "Raw Sensor Signals",
    save_folder: str = None
):
    """
    Generates a multi-row subplot for synchronized physiological signals.
    
    Args:
        time_axis: Common time axis in seconds.
        signals: Dictionary mapping modality name (str) to signal array (np.ndarray).
        title: Overall figure title.
        save_folder: Output folder to auto-save figure (e.g. 'CHEST').
    """
    set_plot_style()
    n_sig = len(signals)
    fig, axes = plt.subplots(n_sig, 1, figsize=(12, 3 * n_sig), sharex=True)
    if n_sig == 1: axes = [axes]
    
    # Use itertools.cycle to reuse colors if we have more signals than colors in the palette
    colors = itertools.cycle(sns.color_palette())
    
    for ax, (name, sig), color in zip(axes, signals.items(), colors):
        ax.plot(time_axis, sig, color=color, label=name)
        ax.set_title(f"{name} Signal")
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # Add basic stats annotation
        mu, sigma = np.mean(sig), np.std(sig)
        ax.text(0.02, 0.9, f"$\mu={mu:.2f}, \sigma={sigma:.2f}$", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def plot_rel_diagram(y_true, y_prob, title="Reliability Diagram", save_folder: str = None):
    """
    Plots a calibration curve with confidence histograms.
    """
    from sklearn.calibration import calibration_curve
    set_plot_style()
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Model curve
    ax.plot(prob_pred, prob_true, marker='o', label='Model Compliance', color='#4B0082')
    
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    
    _save_plot(fig, title, save_folder)
    return fig

def plot_resampling_verification_grid(
    chest_data: Dict[str, np.ndarray],
    sample_window: Dict[str, np.ndarray],
    start_time: float,
    raw_fs: int = 700,
    target_fs: int = 35,
    plot_duration: int = 30,
    save_folder: str = None
):
    """
    Generates a grid of overlay plots to verify resampling fidelity for ALL modalities.
    
    Args:
        chest_data: Dictionary of raw signals (entire session).
        sample_window: Dictionary of processed window data (resampled).
        start_time: Time in seconds where the window starts.
        raw_fs: Original sampling rate.
        target_fs: Target/processed sampling rate.
        plot_duration: Duration to visualize (seconds).
    
    Returns:
        plt.Figure: The generated Multi-row plot.
    """
    
    # Configuration: (RawKey, WindowKey, Unit, Label)
    verification_targets = [
        ('EDA',  'EDA',  'uS',  'Electrodermal Activity'),
        ('Resp', 'RESP', 'Arb', 'Respiration (Chest)'),
        ('Temp', 'TEMP', '°C',  'Skin Temperature'),
        ('ECG',  'ECG',  'mV',  'Electrocardiogram') # Adding ECG
    ]
    
    # Also Check Accelerometer if possible (usually split in X, Y, Z in window)
    # Adding logic to check for ACC_x, ACC_y, ACC_z if available
    if 'ACC' in chest_data and 'ACC_x' in sample_window:
        verification_targets.append(('ACC', 'ACC_x', 'g', 'ACC X-axis'))
        verification_targets.append(('ACC', 'ACC_y', 'g', 'ACC Y-axis'))
        verification_targets.append(('ACC', 'ACC_z', 'g', 'ACC Z-axis'))
        
    set_plot_style()
    n_plots = len(verification_targets)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    
    start_idx_raw = int(start_time * raw_fs)
    raw_n = int(raw_fs * plot_duration)
    res_n = int(target_fs * plot_duration)
    
    for ax, (raw_key, win_key, unit, label) in zip(axes, verification_targets):
        
        # 1. Prepare Raw
        if raw_key == 'ACC':
            # Handle 3D ACC array: map ACC_x -> col 0, ACC_y -> col 1, etc.
            axis_map = {'ACC_x': 0, 'ACC_y': 1, 'ACC_z': 2}
            axis_idx = axis_map[win_key]
            raw_sig = chest_data[raw_key][start_idx_raw : start_idx_raw + raw_n, axis_idx].flatten()
        else:
            raw_sig = chest_data[raw_key][start_idx_raw : start_idx_raw + raw_n].flatten()
            
        raw_t = np.arange(len(raw_sig)) / raw_fs + start_time
        
        # 2. Prepare Resampled
        res_sig = np.array(sample_window[win_key])[:res_n]
        res_t = np.arange(len(res_sig)) / target_fs + start_time
        
        # 3. Plot Overlay
        # Raw = Primary Purple, Resampled = Contrast Orange/Goldenrod
        ax.plot(raw_t, raw_sig, label=f'Raw ({raw_fs}Hz)', alpha=0.5, linewidth=2, color='#4B0082')
        ax.plot(res_t, res_sig, label=f'Resampled ({target_fs}Hz)', linewidth=1.5, linestyle='--', color='#FF8C00', marker='o', markersize=3)
        
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc='upper right')
        ax.set_title(label + " Fidelity Alignment")
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time (s)")
    title = f"Pipeline Verification: Signal Fidelity (Aligned at t={start_time:.1f}s)"
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def plot_sqi_comparison(scenarios: List[tuple], signal_key: str = 'EDA', thresholds: Dict[str, float] = None, save_folder: str = None) -> plt.Figure:
    """
    Plots a comparison of signal quality scenarios (Good, Flatline, Motion).
    
    Args:
        scenarios: List of tuples (window_data, title, scores_dict).
        signal_key: Which physiological signal to inspect (EDA, TEMP, ECG). Default 'EDA'.
        thresholds: Optional dictionary of flatline thresholds to override defaults.
    
    Returns:
        plt.Figure: The generated figure.
    """
    set_plot_style()
    n_scenarios = len(scenarios)
    
    # Create a grid: Rows = Scenarios, Cols = 2 (Signal, ACC Magnitude)
    fig, axes = plt.subplots(n_scenarios, 2, figsize=(14, 4 * n_scenarios), constrained_layout=True)
    
    if n_scenarios == 1:
        axes = [axes]
        
    # Mapping for Labels and Units
    meta = {
        'EDA':  {'unit': 'uS', 'label': 'Skin Conductance', 'req': 0.005},
        'TEMP': {'unit': '°C', 'label': 'Skin Temperature', 'req': 0.01},
        'ECG':  {'unit': 'mV', 'label': 'ECG', 'req': 0.05},
        'RESP': {'unit': '%',  'label': 'Respiration', 'req': 0.5}
    }
    
    # Override defaults if thresholds provided
    if thresholds:
        for k, v in thresholds.items():
            if k in meta:
                meta[k]['req'] = v
    
    info = meta.get(signal_key, {'unit': 'a.u.', 'label': signal_key, 'req': 0.0})
    
    for i, (window, title, scores) in enumerate(scenarios): 
        # Row handling
        ax_sig = axes[i][0] if n_scenarios > 1 else axes[0]
        ax_acc = axes[i][1] if n_scenarios > 1 else axes[1]
        
        # 1. Plot Selected Signal (EDA/TEMP/ECG)
        if signal_key in window:
            sig = np.array(window[signal_key])
            ax_sig.plot(sig, color='#4B0082', label=f'{info["label"]} Signal')
            
            # Visualize SQI Logic: Range Check
            s_min, s_max = np.min(sig), np.max(sig)
            s_range = s_max - s_min
            
            # Draw "Boundaries" of the signal range
            ax_sig.axhline(s_min, color='#4B0082', linestyle=':', alpha=0.4)
            ax_sig.axhline(s_max, color='#4B0082', linestyle=':', alpha=0.4)
            
            # Draw "Dead Sensor Limit" (Centered for visual comparison) - Only if req > 0
            min_req = info['req']
            if min_req > 0:
                s_mid = (s_max + s_min) / 2
                ax_sig.axhline(s_mid + min_req/2, color='#7f7f7f', linestyle='--', alpha=0.5)
                ax_sig.axhline(s_mid - min_req/2, color='#7f7f7f', linestyle='--', alpha=0.5)
            
            # Annotate the Range vs Threshold
            req_text = f"{min_req} {info['unit']}" if min_req > 0 else "N/A"
            range_info = f"Signal Range: {s_range:.4f} {info['unit']}\nMin Req: {req_text}"
            
            ax_sig.text(0.02, 0.05, range_info, transform=ax_sig.transAxes, 
                        fontsize=8, verticalalignment='bottom', color='#4B0082',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e0e0e0', boxstyle='round,pad=0.2'))

            ax_sig.set_ylabel(f"{info['label']} ({info['unit']})")
            ax_sig.set_title(f"{title} - {info['label']} Trace")
            ax_sig.grid(True)
        else:
            ax_sig.text(0.5, 0.5, f"No {signal_key} Data", ha='center')

        # 2. Plot ACC Magnitude (Motion Proxy)
        if 'ACC_x' in window:
            acc_x = np.array(window['ACC_x'])
            acc_y = np.array(window['ACC_y'])
            acc_z = np.array(window['ACC_z'])
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            
            # Subtract gravity (1.0) to show dynamic component (centered at 0)
            acc_dynamic = acc_mag - 1.0
            
            ax_acc.plot(acc_dynamic, color="#FFA500")
            
            # Add "Safe Corridor" visual guide
            ax_acc.axhline(0.5, color='#7f7f7f', linestyle='--', alpha=0.5)
            ax_acc.axhline(-0.5, color='#7f7f7f', linestyle='--', alpha=0.5)
            
            # Add simple text for threshold
            ax_acc.text(0.02, 0.05, "Noise Limit: +/- 0.5g", transform=ax_acc.transAxes,
                        fontsize=8, verticalalignment='bottom', color='#7f7f7f',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#e0e0e0', boxstyle='round,pad=0.2'))
            
            ax_acc.set_ylabel("Dynamic Accel (g)")
            
            # 3. Add Annotation Box with Scores (Upper Right of Both Plots)
            score_text = "SQI Scores:\n" + "\n".join([f"{k}: {v:.2f}" for k, v in scores.items()])
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            
            # Place on Signal
            ax_sig.text(0.98, 0.95, score_text, transform=ax_sig.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right', bbox=props)
            # Place on ACC
            ax_acc.text(0.98, 0.95, score_text, transform=ax_acc.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right', bbox=props)
            
            # 4. Contextual Title Logic
            # Check Signal Quality (Generic check for key ending in _sqi)
            sig_sqi_key = f"{signal_key.lower()}_sqi"
            warning_color = '#C75B1F'
            
            if sig_sqi_key in scores and scores[sig_sqi_key] < 0.5:
                 ax_sig.set_title(f"{title} - {signal_key} [ANOMALY DETECTED]", color=warning_color, fontweight='bold')
            else:
                ax_sig.set_title(f"{title} - {signal_key}")
            
            # Check Motion Quality
            if 'motion_score' in scores and scores['motion_score'] < 0.5:
                 ax_acc.set_title(f"{title} - Motion [HIGH NOISE]", color=warning_color, fontweight='bold')
            else:
                 ax_acc.set_title(f"{title} - Motion")

    title = f"Signal Quality Diagnostics: {signal_key} vs Motion"
    plt.suptitle(title, fontsize=16, y=1.05)
    _save_plot(fig, title, save_folder)
    return fig

def plot_eda_decomposition(
    time_axis: np.ndarray,
    raw: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    peaks: np.ndarray,
    threshold: float = 0.01,
    motion_signal: Optional[np.ndarray] = None,
    save_folder: str = None
):
    """
    Visualizes EDA decomposition into Tonic and Phasic components.
    
    Args:
        time_axis: Time vector in seconds.
        raw: Raw EDA signal.
        tonic: Extracted Tonic component.
        phasic: Extracted Phasic component.
        peaks: Indices of detected SCR peaks.
        threshold: Phasic threshold used for detection.
        motion_signal: Optional accelerometer signal for context.
    """
    set_plot_style()
    colors = sns.color_palette()
    primary = colors[0] # #4B0082
    contrast = colors[1] # #FF8C00
    
    n_plots = 3 if motion_signal is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots), sharex=True)
    
    # 1. Raw vs Tonic
    ax = axes[0]
    ax.plot(time_axis, raw, color=primary, label='Raw EDA')
    ax.plot(time_axis, tonic, color=contrast, linestyle='--', linewidth=2, label='Tonic (Low-pass)')
    ax.set_title("EDA Decomposition: Raw vs Tonic")
    ax.legend(loc='upper right')
    ax.set_ylabel("uS")
    ax.grid(True)
    
    # 2. Phasic & Peaks
    ax = axes[1]
    ax.plot(time_axis, phasic, color=primary, label='Phasic (Residual)')
    
    # Updated Peak Visualization: Transparent with outline using plt.scatter
    if len(peaks) > 0:
        ax.scatter(
            time_axis[peaks], 
            phasic[peaks], 
            color=contrast, 
            s=80, 
            alpha=0.6, 
            edgecolor='white', 
            linewidth=1.5, 
            zorder=3,
            label=f'Detected SCRs (n={len(peaks)})'
        )
    
    ax.axhline(threshold, color='gray', linestyle=':', alpha=0.5, label=f'Threshold ({threshold} uS)')
    ax.set_title("Phasic Component & SCR Events")
    ax.legend(loc='upper right')
    ax.set_ylabel("uS")
    ax.grid(True)
    
    # 3. Motion (Optional)
    if motion_signal is not None:
        ax = axes[2]
        # Use a lighter/grey shade for context
        ax.plot(time_axis, motion_signal, color='#7f7f7f', alpha=0.6, label='Motion (ACC X)')
        ax.set_title("Motion Context")
        ax.set_xlabel("Time (s)")
        ax.legend(loc='upper right')
        ax.set_ylabel("Acc (g)")
        ax.grid(True)
    else:
        axes[1].set_xlabel("Time (s)")
        
    plt.tight_layout()
    _save_plot(fig, "EDA_Decomposition", save_folder)
    return fig

def plot_ecg_audit(
    time_axis: np.ndarray,
    clean_ecg: np.ndarray,
    peaks: np.ndarray,
    threshold: float,
    title: str = "ECG Audit",
    save_folder: str = None
):
    """
    Visualizes ECG signal and detected R-peaks.
    
    Args:
        time_axis: Time vector in seconds.
        clean_ecg: Processed ECG signal.
        peaks: Indices of detected R-peaks.
        threshold: Detection threshold used.
    """
    set_plot_style()
    colors = sns.color_palette()
    primary = colors[0] # #4B0082
    contrast = colors[1] # #FF8C00
    
    fig = plt.figure(figsize=(12, 4))
    plt.plot(time_axis, clean_ecg, color=primary, label='Processed ECG')
    plt.plot(time_axis[peaks], clean_ecg[peaks], "o", color=contrast, markersize=6, label='R-Peaks')
    plt.axhline(threshold, color='gray', linestyle=':', label='Threshold')
    
    # Calculate BPM for title
    if len(peaks) > 1:
        duration_min = (time_axis[-1] - time_axis[0]) / 60
        bpm_val = len(peaks) / duration_min
        title = f"{title}: Found {len(peaks)} beats (~{bpm_val:.0f} BPM)"
        
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def _get_feature_unit(feature_name: str) -> str:
    """Helper to deduce unit from feature name substring."""
    f = feature_name.lower()
    if 'eda' in f: return '($\mu S$)'
    if 'temp' in f:
        if 'slope' in f: return '($^\circ C/s$)'
        return '($^\circ C$)'
    if 'acc' in f: return '($g$ or $m/s^2$)'
    if 'ecg' in f:
        if 'hr' in f or 'bpm' in f: return '(bpm)'
        if 'rmssd' in f or 'sdnn' in f: return '(s)'
        return '(mV)'
    if 'resp' in f:
        if 'rate_bpm' in f: return '(bpm)'
        if 'rate_hz' in f: return '(Hz)'
        return '(a.u.)'
    if 'bvp' in f:
        if 'hr' in f: return '(bpm)'
        return '(a.u.)'
    return ''

def _create_balanced_grid(n_plots: int, title: str = "", base_size: float = 4.0):
    """
    Creates a mosaic grid (rows with different column counts) to ensure:
    1. No empty slots (all rows full).
    2. Consistent aspect ratio (plot shape preserved, though size may vary).
    """
    # 1. Solve for row configuration: x rows of c cols, y rows of c+1 cols
    # Objective: Minimize total rows, prefer 3-5 columns.
    best_sol = None # (x, y, c)
    min_rows = float('inf')
    
    # Range of column counts to consider
    candidates = [3, 4, 5]
    if n_plots < 3: candidates = [n_plots]
    
    for c in candidates:
        if n_plots == c:
            best_sol = (1, 0, c)
            break
            
        c_next = c + 1
        # Solve x*c + y*(c+1) = n_plots
        max_y = n_plots // c_next
        for y in range(max_y + 1):
            remainder = n_plots - (y * c_next)
            if remainder >= 0 and remainder % c == 0:
                x = remainder // c
                total = x + y
                if total < min_rows:
                    min_rows = total
                    best_sol = (x, y, c)
    
    # Fallback if no clean mosaic found (should handle large N, but for small N just grid)
    if best_sol is None:
        cols = 4 if n_plots > 4 else n_plots
        rows = int(np.ceil(n_plots / cols))
        # Default behavior: Just standard grid (stretch handled previously, or gaps)
        # We revert to gaps if solver fails (unlikely for N > 5)
        fig = plt.figure(figsize=(base_size * cols, base_size * 0.8 * rows))
        return fig, [fig.add_subplot(rows, cols, i+1) for i in range(n_plots)]

    x, y, c = best_sol
    # Configuration: x rows of c, y rows of c+1
    row_config = [c] * x + [c + 1] * y
    # Optional: Sort rows so larger plots (fewer cols) are at top? Or bottom?
    # Current: Top=Smaller Density (Larger Plots), Bottom=Higher Density
    
    # 2. Calculate Row Heights to preserve Aspect Ratio (Height = Width * 0.75)
    # Figure Width is fixed constant relative to 'Base Size' (e.g. normalized to Max Cols)
    # Let Figure Width = base_size * Max(Cols)
    # Then Plot Width in row i = Figure Width / Cols_i
    # Plot Height in row i = Plot Width * 0.75
    
    max_cols = max(row_config)
    fig_width = base_size * max_cols
    
    row_heights = []
    for rc in row_config:
        w_plot = fig_width / rc
        h_plot = w_plot * 0.7 # Aspect ratio 0.7
        row_heights.append(h_plot)
        
    total_height = sum(row_heights)
    
    # 3. Create Figure using specific rects or GridSpecs
    fig = plt.figure(figsize=(fig_width, total_height))
    
    # Use nested GridSpec to handle varying columns per row
    # Outer GridSpec: Rows
    gs_outer = mpl.gridspec.GridSpec(len(row_config), 1, figure=fig, height_ratios=row_heights)
    
    axes = []
    for i, n_cols in enumerate(row_config):
        # Inner GridSpec: Columns for this row
        gs_inner = mpl.gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs_outer[i])
        for j in range(n_cols):
            axes.append(fig.add_subplot(gs_inner[0, j]))
            
    if title:
        # Adjust top margin
        fig.suptitle(title, y=1.0 + (0.2/total_height), fontsize=16)

    return fig, axes

def plot_feature_separability(
    df: pd.DataFrame, # type: ignore
    feature_cols: List[str],
    label_col: str,
    title: str = "Feature Separability",
    save_folder: str = None
):
    """
    Generates Violin plots + Strip plots to show feature distribution by class.
    """
    set_plot_style()
    n_cols = len(feature_cols)
    
    fig, axes = _create_balanced_grid(n_cols, title, base_size=3.5) # Reduced base size for mosaic
    
    for i, col_name in enumerate(feature_cols):
        ax = axes[i]
        if col_name not in df.columns:
            ax.text(0.5, 0.5, f"{col_name} not found", ha='center')
            continue
            
        n_classes = len(df[label_col].unique())
        current_palette = sns.color_palette()[:n_classes]

        sns.violinplot(
            x=label_col, 
            y=col_name, 
            data=df, 
            ax=ax, 
            hue=label_col,
            inner=None, 
            palette=current_palette, 
            legend=False,
            density_norm='width',
            linewidth=1.5,
            saturation=1.0, 
            alpha=0.8       
        )
        
        sns.boxplot(
            x=label_col,
            y=col_name,
            data=df,
            ax=ax,
            width=0.1, 
            showcaps=False,
            boxprops={'facecolor': 'white', 'edgecolor': '#333333', 'alpha': 0.4, 'zorder': 10},
            whiskerprops={'color': '#333333', 'linewidth': 2, 'zorder': 10},
            medianprops={'color': '#333333', 'linewidth': 2, 'zorder': 10},
            fliersize=0 
        )
        
        sns.stripplot(
            x=label_col, 
            y=col_name, 
            data=df, 
            ax=ax, 
            jitter=True, 
            alpha=0.7,   
            color='#202020', 
            size=4       
        )
        
        unit_str = _get_feature_unit(col_name)
        ax.set_title(f"{col_name} {unit_str}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def plot_feature_comparison_bars(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    label_col: str, 
    title: str = "Feature Comparison (Mean \u00B1 SE)",
    save_folder: str = None
):
    """
    Generates Bar plots with Standard Error bars.
    """
    set_plot_style()
    n_cols = len(feature_cols)
    
    fig, axes = _create_balanced_grid(n_cols, title, base_size=3.5)
    
    for i, col_name in enumerate(feature_cols):
        ax = axes[i]
        if col_name not in df.columns:
            ax.text(0.5, 0.5, f"{col_name} not found", ha='center')
            continue
            
        n_classes = len(df[label_col].unique())
        current_palette = sns.color_palette()[:n_classes]
        
        sns.barplot(
            x=label_col,
            y=col_name,
            data=df,
            ax=ax,
            palette=current_palette,
            errorbar='se', 
            capsize=0.1,
            err_kws={'linewidth': 2, 'color': '#333333'},
            edgecolor='#333333',
            linewidth=1.5,
            alpha=0.9
        )
        
        ax.set_title(col_name, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Mean Value")
        
    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def plot_feature_importance_cohens_d(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    label_col: str, 
    class_1: str = 'Stress',
    class_0: str = 'Baseline',
    title: str = "Feature Separability (Cohen's d)",
    save_folder: str = None
):
    """
    Calculates and plots Cohen's d effect size for each feature (Single Plot).
    X-axis: Cohen's d (Metric of separability).
    Y-axis: Features.
    
    Positive d: Higher in Stress.
    Negative d: Higher in Baseline.
    """
    set_plot_style()
    
    effect_sizes = []
    
    group1 = df[df[label_col] == class_1]
    group0 = df[df[label_col] == class_0]
    
    for feat in feature_cols:
        if feat not in df.columns: continue
        
        g1 = group1[feat].dropna().values
        g0 = group0[feat].dropna().values
        
        if len(g1) < 2 or len(g0) < 2:
            d = 0
        else:
            n1, n2 = len(g1), len(g0)
            var1, var2 = np.var(g1, ddof=1), np.var(g0, ddof=1)
            mean1, mean2 = np.mean(g1), np.mean(g0)
            
            # Pooled SD
            pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            if pooled_sd == 0:
                d = 0
            else:
                d = (mean1 - mean2) / pooled_sd
                
        effect_sizes.append({'Feature': feat, 'Cohens_d': d})
        
    df_effect = pd.DataFrame(effect_sizes)
    # Sort by absolute magnitude for visibility, or by signed value
    df_effect = df_effect.sort_values(by='Cohens_d', ascending=True)
    
    # Create Colors based on sign
    # Purple (#4B0082) for Negative (Higher in Baseline)
    # Orange (#FF8C00) for Positive (Higher in Stress)
    colors = ['#4B0082' if x < 0 else '#FF8C00' for x in df_effect['Cohens_d']]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.4)))
    
    bars = ax.barh(df_effect['Feature'], df_effect['Cohens_d'], color=colors, edgecolor='#333333', alpha=0.9)
    
    # Add vertical line at 0
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    
    # Add guidelines for Effect Size interpretation
    # Small=0.2, Medium=0.5, Large=0.8
    for d_val, style in zip([0.2, 0.5, 0.8, -0.2, -0.5, -0.8], [':', '--', '-.']*2):
        ax.axvline(d_val, color='gray', alpha=0.3, linestyle=style, zorder=0)
        
    ax.set_title(title)
    ax.set_xlabel(f"Cohen's d (Effect Size)\n← Higher in {class_0} | Higher in {class_1} →")
    
    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + np.sign(width) * 0.05
        ha = 'left' if width > 0 else 'right'
        
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                va='center', ha=ha, fontsize=9, fontweight='bold', color='#333333')

    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def plot_feature_pairplot(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    title: str = "Feature Pairwise Relationships",
    save_folder: str = None
):
    """
    Generates a Pair Plot (Scatter matrix) for selected features.
    
    Args:
        df: Dataframe containing features and label.
        feature_cols: List of features to plot. Limit this to 5-8 for readability.
        label_col: Column name to use for Hue (color coding).
        title: Figure title.
    """
    set_plot_style()
    
    # We explicitly define the palette dict to map the label strings to our colors
    # Assuming standard project labels if possible, but auto-detecting unique values
    unique_labels = sorted(df[label_col].unique())
    
    # Map 'Baseline' to Purple, 'Stress' to Orange if those exact strings exist
    # Otherwise fallback to the cycle
    palette_map = {}
    if 'Baseline' in unique_labels and 'Stress' in unique_labels:
        palette_map['Baseline'] = '#4B0082' # Deep Purple
        palette_map['Stress'] = '#FF8C00'   # Orange
    else:
        # Fallback to defaults
        colors = ["#4B0082", "#FF8C00", "#9370DB", "#FFA500"]
        palette_map = dict(zip(unique_labels, colors))

    # Create PairGrid
    g = sns.pairplot(
        df[feature_cols + [label_col]], 
        hue=label_col,
        palette=palette_map,
        diag_kind='kde',
        plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'white', 'linewidth': 0.5},
        diag_kws={'fill': True, 'alpha': 0.5},
        height=2.5
    )
    
    g.fig.suptitle(title, y=1.02, fontsize=16)
    
    _save_plot(g.fig, title, save_folder)
    return g

def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    title: str = "Feature Correlation Matrix",
    save_folder: str = None
):
    """
    Plots a heatmap of the correlation matrix for selected features.
    """
    set_plot_style()
    
    corr = df[feature_cols].corr()
    
    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set background to dark gray for the masked area
    ax.set_facecolor('#404040')
    
    # Diverging colormap (Purple - White - Orange)
    cmap = LinearSegmentedColormap.from_list(
        "custom_div", 
        ['#4B0082', '#FFFFFF', '#FF8C00']
    )
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f"
    )
    
    ax.set_title(title)
    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig

def get_top_discriminative_features(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    label_col: str, 
    n: int = 6,
    class_1: str = 'Stress',
    class_0: str = 'Baseline'
) -> List[str]:
    """
    Selects top features based on Cohen's d effect size, identifying diversity.
    Prioritizes the highest-ranked feature per modality if redundancy is detected.
    """
    effect_sizes = []
    
    group1 = df[df[label_col] == class_1]
    group0 = df[df[label_col] == class_0]
    
    for feat in feature_cols:
        # Skip redundant Hz features (keep bpm)
        if '_Hz' in feat: continue
        
        g1 = group1[feat].dropna().values
        g0 = group0[feat].dropna().values
        
        if len(g1) < 2 or len(g0) < 2:
            d = 0
        else:
            diff = np.mean(g1) - np.mean(g0)
            pooled_sd = np.sqrt((np.var(g1, ddof=1) + np.var(g0, ddof=1)) / 2)
            d = abs(diff / pooled_sd) if pooled_sd > 0 else 0
            
        effect_sizes.append((feat, d))
        
    # Sort by absolute effect size
    effect_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Filter for diversity
    # Strategy: Allow max 1 'temp', 2 'eda', 2 'resp', 2 'ecg', 2 'acc' unless we run out
    final_feats = []
    counts = {'temp': 0, 'eda': 0, 'resp': 0, 'ecg': 0, 'acc': 0, 'bvp': 0}
    
    for f, d in effect_sizes:
        # Identify type
        ftype = next((k for k in counts.keys() if k in f), 'other')
        
        # Hard limits per type to force diversity
        limit = 1 if ftype == 'temp' else 2
        
        if counts.get(ftype, 0) < limit:
            final_feats.append(f)
            if ftype in counts: counts[ftype] += 1
            
        if len(final_feats) >= n: break
            
    return final_feats

def plot_temp_audit(
    time_axis: np.ndarray,
    temp_signal: np.ndarray,
    slope: float = None,
    title: str = "Temperature Trend Audit",
    save_folder: str = None
) -> plt.Figure:
    """
    Visualizes Temperature signal and its linear trend.
    If 'slope' is provided (in units/sec), it is used for the trend line.
    Otherwise, the slope is calculated from the data.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Raw Signal
    ax.plot(time_axis, temp_signal, label='Raw Temp', color='#4B0082', linewidth=2)
    
    # Determine FS from time axis
    estimated_fs = 1.0 / np.mean(np.diff(time_axis))
    x = np.arange(len(temp_signal))
    
    if slope is not None:
        # feature 'slope' is per second. Convert to per sample.
        slope_per_sample = slope / estimated_fs
        
        # Calculate intercept (line passes through centroid)
        mean_x = np.mean(x)
        mean_y = np.mean(temp_signal)
        intercept = mean_y - slope_per_sample * mean_x
        
        label_text = f'Feature Slope: {slope:.4f}/s'
    else:
        # Recalculate
        slope_calc, intercept, _, _, _ = stats.linregress(x, temp_signal)
        slope_per_sample = slope_calc
        slope = slope_calc * estimated_fs
        label_text = f'Calc Slope: {slope:.4f}/s'
    
    fit_line = slope_per_sample * x + intercept
    
    ax.plot(time_axis, fit_line, label=label_text, 
            color='#FF8C00', linestyle='--', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    
    # Add annotation
    ax.text(0.05, 0.95, label_text, 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            
    _save_plot(fig, title, save_folder)
    return fig

def plot_resp_audit(
    time_axis: np.ndarray,
    resp_signal: np.ndarray,
    fs: float,
    detected_rate: float = None,
    title: str = "Respiration Rate Audit",
    save_folder: str = None
) -> plt.Figure:
    """
    Visualizes Respiration signal and its Power Spectral Density (PSD)
    to audit the dominant frequency (Respiration Rate).
    """
    set_plot_style()
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    
    # 1. Time Domain
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(time_axis, resp_signal, color='#4B0082', label='Resp Signal')
    ax0.set_title("Time Domain")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    
    # 2. Frequency Domain (PSD)
    ax1 = fig.add_subplot(gs[1])
    freqs, psd = signal.welch(resp_signal, fs=fs, nperseg=len(resp_signal), window='hann')
    
    # Focus on physiological range 0-1Hz
    mask = (freqs <= 1.0)
    ax1.plot(freqs[mask], psd[mask], color='#FF8C00', label='PSD')
    
    # Find peak in valid range (0.1 - 0.5 Hz)
    valid_mask = (freqs >= 0.1) & (freqs <= 0.5)
    
    if detected_rate is not None:
        # If rate is provided (from robust extractor), visualize that specific point
        # We find the PSD value at that frequency (interpolation or nearest)
        idx = (np.abs(freqs - detected_rate)).argmin()
        peak_freq = freqs[idx]
        peak_pow = psd[idx]
        
        ax1.plot(peak_freq, peak_pow, 'o', color="black", label=f'Detected: {peak_freq:.2f}Hz')
        ax1.axvline(peak_freq, color="black", linestyle=':', alpha=0.5)
        ax1.text(peak_freq + 0.02, peak_pow, f"Rate: {peak_freq*60:.1f} bpm",
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                 
    elif np.any(valid_mask):
        # Fallback to simple argmax if no rate provided
        f_valid = freqs[valid_mask]
        p_valid = psd[valid_mask]
        peak_idx = np.argmax(p_valid)
        peak_freq = f_valid[peak_idx]
        peak_pow = p_valid[peak_idx]
        
        ax1.plot(peak_freq, peak_pow, 'ro', label=f'Peak: {peak_freq:.2f}Hz')
        ax1.axvline(peak_freq, color='r', linestyle=':', alpha=0.5)
        ax1.text(0.5, 0.9, f"Rate: {peak_freq*60:.1f} bpm", transform=ax1.transAxes, ha='center',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    ax1.set_title("Frequency Domain (PSD)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Density")
    ax1.legend()
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig


def _plot_styled_violin(ax, data, y, color):
    """
    Internal helper to plot a consistent 'Stress-style' violin with inner boxplot.
    Does NOT plot the strip points (handled separately).
    """
    # 1. Violin (Outer Shape)
    sns.violinplot(
        y=data[y],
        ax=ax,
        color=color,
        inner=None, 
        density_norm='width',
        linewidth=1.5,
        saturation=1.0,
        alpha=0.8,
        width=0.7
    )
    
    # 2. Boxplot (Inner Stats)
    sns.boxplot(
        y=data[y],
        ax=ax,
        width=0.1,
        showcaps=False,
        boxprops={'facecolor': 'white', 'edgecolor': '#333333', 'alpha': 0.4, 'zorder': 10},
        whiskerprops={'color': '#333333', 'linewidth': 2, 'zorder': 10},
        medianprops={'color': '#333333', 'linewidth': 2, 'zorder': 10},
        fliersize=0
    )

def plot_model_diagnostics(results_df: pd.DataFrame, save_folder: str = None, title_prefix: str = ""):
    """
    Generates a 2x2 diagnostic panel for model verification (LOSO).
    
    Panels:
    1. Global Confusion Matrix (Counts)
    2. ROC Curves (Global + Per-Subject)
    3. Mean Subject Confusion Matrix (Normalized +/- SE)
    4. Accuracy Distribution (Violin + Strip)
    """
    subjects = sorted(results_df['subject_id'].unique())
    # Color Constants
    DEEP_PURPLE = "#4B0082"
    DARK_ORANGE = "#FF8C00"
    # Revert to LinearSegmentedColormap for better contrast control
    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", DEEP_PURPLE])
    
    # Setup Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    
    # --- 1. Global Confusion Matrix (Top-Left) ---
    ax_cm = axes[0, 0]
    # Always show both classes (0, 1) for binary stress classification
    cm = confusion_matrix(results_df['true'], results_df['pred'], labels=[0, 1])
    acc_pooled = accuracy_score(results_df['true'], results_df['pred'])
    
    # Disable annot=True, use manual
    sns.heatmap(cm, annot=False, cmap=cmap_purple, ax=ax_cm, cbar=False,
                xticklabels=['Baseline', 'Stress'], yticklabels=['Baseline', 'Stress'])
    _add_heatmap_annotations(ax_cm, cm, fmt='d', threshold=0.3)
    
    ax_cm.set_title(f'Global CM (Pooled) - Acc: {acc_pooled:.1%}', fontsize=16)
    ax_cm.set_ylabel('True Label', fontsize=12)
    ax_cm.set_xlabel('Predicted Label', fontsize=12)
    
    # --- 2. ROC Curves (Top-Right) ---
    ax_roc = axes[0, 1]
    subject_palette = sns.color_palette("plasma", n_colors=len(subjects))
    subj_color_map = dict(zip(subjects, subject_palette))
    
    for sub in subjects:
        sub_df = results_df[results_df['subject_id'] == sub]
        if len(sub_df['true'].unique()) < 2: continue
        
        fpr_s, tpr_s, _ = roc_curve(sub_df['true'], sub_df['prob'])
        ax_roc.plot(fpr_s, tpr_s, color=subj_color_map[sub], alpha=0.5, lw=2)
        
        # Add labels roughly at the "elbow"
        mid_idx = len(fpr_s) // 2
        ax_roc.text(fpr_s[mid_idx], tpr_s[mid_idx], sub, 
                   fontsize=8, color=subj_color_map[sub], fontweight='bold')
                   
    # Global
    fpr, tpr, _ = roc_curve(results_df['true'], results_df['prob'])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=DARK_ORANGE, lw=5, label=f'Global (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_title('ROC Curves (Subject Variability)', fontsize=16)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    
    # --- 3. Mean Subject Confusion Matrix (Bottom-Left) ---
    ax_mcm = axes[1, 0]
    
    # Collect normalized CMs and Accuracies
    cms = []
    subject_accuracies = []
    for sub in subjects:
        sub_df = results_df[results_df['subject_id'] == sub]
        # Normalize by true (Rows sum to 1) -> Recall per class
        # Labels=[0,1] ensures shape (2,2) even if 1 class missing
        c = confusion_matrix(sub_df['true'], sub_df['pred'], labels=[0, 1], normalize='true')
        cms.append(c)
        acc = accuracy_score(sub_df['true'], sub_df['pred'])
        subject_accuracies.append({'Subject': sub, 'Accuracy': acc})
        
    cms = np.array(cms) # Shape (n_subjects, 2, 2)
    mean_cm = np.mean(cms, axis=0)
    se_cm = stats.sem(cms, axis=0)
    
    df_acc = pd.DataFrame(subject_accuracies).sort_values('Accuracy', ascending=False)
    mean_subject_acc = df_acc['Accuracy'].mean()
    
    # Custom annotations
    annot = np.empty_like(mean_cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{mean_cm[i, j]:.2f}\n±{se_cm[i, j]:.2f}"
            
    sns.heatmap(mean_cm, annot=False, cmap=cmap_purple, ax=ax_mcm, cbar=True,
                xticklabels=['Baseline', 'Stress'], yticklabels=['Baseline', 'Stress'])
    # Pass the formatted 'annot' array to the manual function
    _add_heatmap_annotations(ax_mcm, mean_cm, fmt=annot, threshold=0.3)
    
    ax_mcm.set_title(f'Mean Subject CM (Subject-Normalized) - Avg Acc: {mean_subject_acc:.1%}', fontsize=16)
    ax_mcm.set_ylabel('True Label', fontsize=12)
    ax_mcm.set_xlabel('Predicted Label', fontsize=12)
    
    # --- 4. Accuracy Distribution (Bottom-Right) ---
    ax_acc = axes[1, 1]
    

    # Violin Plot (Refactored to match Notebook 2 style)
    _plot_styled_violin(ax_acc, df_acc, 'Accuracy', DARK_ORANGE)

    # Strip Plot: Manual Scatter to ensure visibility and valid zorder
    np.random.seed(42) # Consistent look
    x_jitter = np.random.uniform(-0.04, 0.04, size=len(df_acc))

    # Extract colors for each point based on subject map (same as ROC)
    point_colors = [subj_color_map[sub] for sub in df_acc['Subject']]

    # Scatter: Subject-Colored dots with Dark edges
    ax_acc.scatter(x_jitter, df_acc['Accuracy'], 
                   c=point_colors,               # Colored center
                   edgecolors='#202020',         # Dark border
                   linewidths=2.0,               
                   s=65,                         
                   zorder=200, 
                   alpha=1.0)

    # Place subject id label in the subject's color, very close to the dot (slightly above)
    for i, (idx, row) in enumerate(df_acc.iterrows()):
        ax_acc.text(x_jitter[i], row['Accuracy'] + 0.015, f"{row['Subject']}",
                   color=subj_color_map[row['Subject']],
                   fontsize=10,
                   fontweight='bold',
                   va='bottom', ha='center', zorder=201)

    ax_acc.axhline(df_acc['Accuracy'].mean(), color=DEEP_PURPLE, linestyle='--', linewidth=2, label='Mean')
    ax_acc.set_title('LOSO Accuracy Distribution', fontsize=16)
    
    # Standard 0-1 range + top headroom
    ax_acc.set_ylim(0.0, 1.1)
    
    # Focus X-axis
    ax_acc.set_xlim(-0.5, 0.5)
    
    # Add Tick on X-axis (Categorical)
    ax_acc.set_xticks([0])
    ax_acc.set_xticklabels(["All Subjects"], fontsize=12, fontweight='bold')
    ax_acc.set_xlabel("")
    
    ax_acc.grid(axis='y', alpha=0.3)
    ax_acc.grid(axis='y', alpha=0.3)
    
    save_name = f"{title_prefix}Model_Diagnostics_Panel" if title_prefix else "Model_Diagnostics_Panel"
    _save_plot(fig, save_name, save_folder)
    return fig

def plot_timeline_segmentation(df: pd.DataFrame, title: str = "Timeline Segmentation", save_folder: str = None):
    """
    Visualizes the timeline segmentation (Stress vs Baseline vs Excluded).
    Useful for verifying data integrity and gap patterns.
    
    Args:
        df: DataFrame containing "target" column (1=Stress, 0=Baseline), indexed by window ID.
    """
    set_plot_style()
    
    plt.figure(figsize=(15, 3))

    max_idx = df.index.max()
    full_timeline = np.arange(max_idx + 1)

    # Create masks
    is_stress = np.zeros(len(full_timeline))
    is_baseline = np.zeros(len(full_timeline))

    # WESAD Specific: target 1 is Stress, 0 is Baseline (remapped) or strict labels
    stress_indices = df[df["target"] == 1].index
    baseline_indices = df[df["target"] == 0].index

    is_stress[stress_indices] = 1
    is_baseline[baseline_indices] = 1
    is_excluded = 1 - (is_stress + is_baseline)

    # Plot
    plt.fill_between(full_timeline, is_stress, step="mid", color="#FF8C00", alpha=0.9, label="Stress (Retained)")
    plt.fill_between(full_timeline, is_baseline, step="mid", color="#4B0082", alpha=0.9, label="Baseline (Retained)")
    plt.fill_between(full_timeline, is_excluded, step="mid", color="#4A4A4A", alpha=0.9, label="Excluded (Amusement / Artifact)")

    plt.title(f"{title}: {len(df)} windows retained")
    plt.xlabel("Original Window Index")
    plt.yticks([])
    plt.xlim(0, max_idx)
    plt.legend(loc="upper right", ncol=3)
    
    _save_plot(plt.gcf(), title, save_folder)
    return plt.gcf()


def plot_multiscale_heatmap(
    tensor_norm: np.ndarray, 
    df: pd.DataFrame,
    channels: List[str], 
    title_prefix: str = "Feature Intensity",
    save_folder: str = None,
    tensor_micro: np.ndarray = None
):
    """
    Generates a Macro-View (entire session) and Micro-View (single window) heatmap.
    
    Args:
        tensor_norm: Standardized Tensor (N, C, T) used for Macro View (and Micro if tensor_micro is None).
        df: DataFrame matching the tensor (for indices and labels).
        channels: List of channel names.
        title_prefix: Title prefix.
        save_folder: Folder to save plot.
        tensor_micro: Optional specific tensor for the Micro View (e.g. Instance Normalized).
    """
    set_plot_style()
    import matplotlib.patches as patches
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.4})

    # --- Prepare Data ---
    # Aggregate for Macro View (Mean over time dimension T)
    # tensor_norm is (N, C, T) -> (N, C)
    heatmap_matrix = tensor_norm.mean(axis=2)
    viz_df = pd.DataFrame(heatmap_matrix, columns=channels, index=df.index)
    
    full_idx = np.arange(df.index.max() + 1)
    viz_df_full = viz_df.reindex(full_idx) # Fill missing indices with NaN for gaps

    # Define Colormap (Deep Purple -> White -> Dark Orange)
    colors = ["#4B0082", "white", "#FF8C00"]
    cmap = LinearSegmentedColormap.from_list("custom_purple_orange", colors)

    # --- Plot 1: Macro View (Timeline) ---
    sns.heatmap(
        viz_df_full.T, 
        cmap=cmap, 
        center=0, 
        robust=True, 
        cbar_kws={"label": "Z-Score"},
        xticklabels=100,
        ax=ax1
    )

    # Highlight Gaps (NaNs) with a dark hatched pattern
    ax1.set_facecolor("#4A4A4A") # Dark gray background

    ax1.set_title(f"Macro-View (Global Normalization)")
    ax1.set_xlabel("Original Window Index (Gray = Excluded)")
    ax1.set_ylabel("Channel")

    # --- Plot 2: Micro View (Single Window Detail) ---
    # Pick a representative Stress Window from the middle of the available stress set
    stress_subset = df[df["target"] == 1]
    if not stress_subset.empty:
        zoom_idx_loc = len(stress_subset) // 2
        original_idx = stress_subset.index[zoom_idx_loc]
        # Find integer location in N dimension
        # df.index.get_loc might return slice if duplicates, but valid_indices should be unique
        # We need the index "i" such that df.iloc[i] corresponds to original_idx
        # Simpler:
        target_iloc = df.index.get_loc(original_idx)
        if isinstance(target_iloc, slice): target_iloc = target_iloc.start # Safety
        
        # Select tensor for Micro View
        micro_source = tensor_micro if tensor_micro is not None else tensor_norm
        window_data = micro_source[target_iloc]
        
        sns.heatmap(
            window_data, 
            cmap=cmap, 
            center=0, 
            robust=True,
            cbar_kws={"label": "Z-Score"},
            yticklabels=channels,
            xticklabels=350, # Mark every 10 seconds (35Hz * 10)
            ax=ax2
        )
        # Dynamic Title based on tensor source
        micro_title_suffix = "(Instance Normalization)" if tensor_micro is not None else ""
        ax2.set_title(f"Micro-View {micro_title_suffix}: Detail of Window #{original_idx}")
        ax2.set_xlabel("Time Samples (0-2100 @ 35Hz)")

        # Highlight the selected window on the Macro plot
        rect = patches.Rectangle((original_idx, 0), width=5, height=len(channels), linewidth=2, edgecolor="#DAA520", facecolor="none")
        ax1.add_patch(rect)
        ax1.annotate("Zoomed Window", xy=(original_idx, len(channels)), xytext=(original_idx, len(channels)+1.5),
                     arrowprops=dict(facecolor="#DAA520", shrink=0.05), color="#DAA520", ha="center", weight="bold")

    else:
        ax2.text(0.5, 0.5, "No Stress Windows Found", ha="center", va="center")

    _save_plot(fig, title_prefix + "_Heatmap", save_folder)
    return fig

def plot_learning_curves(history: dict, save_folder: str = None, title_prefix: str = ""):
    """
    Plots Training and Validation Loss/Accuracy curves aggregated across LOSO folds.
    
    Args:
        history (dict): Dictionary where keys are 'Fold_X' and values are dicts with 'train_loss', 'val_loss', 'val_acc'.
        save_folder (str): Folder to save output.
        title_prefix (str): Prefix for the saved filename.
    """
    if not history:
        print("No training history available to plot.")
        return None

    # Color Constants
    DEEP_PURPLE = "#4B0082"
    DARK_PURPLE = "#4B0082" # Standardizing name
    DARK_ORANGE = "#FF8C00"

    # Aggregate Data
    train_losses_list = []
    val_losses_list = []
    val_accs_list = []
    
    max_epochs = 0
    for fold_id, metrics in history.items():
        t_loss = metrics['train_loss']
        v_loss = metrics['val_loss']
        v_acc = metrics['val_acc']
        
        train_losses_list.append(t_loss)
        val_losses_list.append(v_loss)
        val_accs_list.append(v_acc)
        
        max_epochs = max(max_epochs, len(t_loss))
        
    # Convert to Padding Arrays (Folds x Max_Epochs) with NaN
    n_folds = len(train_losses_list)
    
    train_losses = np.full((n_folds, max_epochs), np.nan)
    val_losses = np.full((n_folds, max_epochs), np.nan)
    val_accs = np.full((n_folds, max_epochs), np.nan)
    
    for i in range(n_folds):
        curr_len = len(train_losses_list[i])
        train_losses[i, :curr_len] = train_losses_list[i]
        val_losses[i, :curr_len] = val_losses_list[i]
        val_accs[i, :curr_len] = val_accs_list[i]
    
    epochs = np.arange(1, max_epochs + 1)
    
    # Calculate Stats (Ignorning NaNs from early stopping)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        train_mean = np.nanmean(train_losses, axis=0)
        train_std = np.nanstd(train_losses, axis=0)
        
        val_mean = np.nanmean(val_losses, axis=0)
        val_std = np.nanstd(val_losses, axis=0)
        
        acc_mean = np.nanmean(val_accs, axis=0)
        acc_std = np.nanstd(val_accs, axis=0)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Loss Curve
    # Plot Individual Folds (Thin Dashed)
    for i in range(n_folds):
        # We only plot validation loss of individual folds to show generalization gap
        # Training loss variance is usually less interesting, but we can plot it very faintly if needed.
        # Let's stick to user request: "curves of each indivisual subject"
        # We'll plot both Train and Val for individuals but extremely thin
        ax1.plot(epochs, train_losses[i], color=DARK_PURPLE, alpha=0.4, linewidth=0.8, linestyle='-')
        ax1.plot(epochs, val_losses[i], color=DARK_ORANGE, alpha=0.4, linewidth=0.8, linestyle='-')

    # Training Mean
    ax1.plot(epochs, train_mean, label='Training Loss', color=DARK_PURPLE, linewidth=2.5) # Thicker solid
    
    # Validation Mean
    ax1.plot(epochs, val_mean, label='Validation Loss', color=DARK_ORANGE, linewidth=2.5, linestyle='-') # Thicker solid (changed from dashed)
    
    ax1.set_title("Learning Dynamics (Loss)", fontsize=16) # Removed Bold
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Curve
    # Plot Individual Folds
    for i in range(n_folds):
         ax2.plot(epochs, val_accs[i], color=DARK_ORANGE, alpha=0.4, linewidth=0.8, linestyle='-')

    ax2.plot(epochs, acc_mean, label='Validation Accuracy', color=DARK_ORANGE, linewidth=2.5) # Thicker solid
    
    ax2.set_title("Generalization (Validation Accuracy)", fontsize=16) # Removed Bold
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.4, 1.0) # Focus on relevant range
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Random Guess')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_folder:
        save_name = f"{title_prefix}learning_curves" if title_prefix else "learning_curves"
        _save_plot(fig, save_name, save_folder)
        
    return fig

def visualize_inference(window_row: pd.Series, predictor, save_folder: str = None):
    """
    1. Prepares data for inference (extracts channels).
    2. Runs predictions via StressPredictor.
    3. Plots Instance-Normalized Heatmap + Prediction Info using the standard purple-orange palette.
    
    Args:
        window_row (pd.Series): Row containing signal channels (ACC_x, etc.) and labels.
        predictor (StressPredictor): Initialized inference wrapper.
        save_folder (str, optional): Folder name to save the figure (under notebooks/outputs/).
        
    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    
    # --- 1. Data Prep ---
    channels = ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EDA', 'RESP', 'TEMP']
    
    # Extract Time-Series structure
    ts_data = {}
    for c in channels:
        if c in window_row:
             ts_data[c] = window_row[c]
        else:
             # Basic handling for missing channels in visualization (fill 0)
             # Ideally this shouldn't happen if validation was passed
             ts_data[c] = np.zeros(2100) # Assumes 35Hz * 60s
        
    # Create Time x Channels DataFrame (60s input)
    input_df = pd.DataFrame(ts_data)
    
    # Inference
    result = predictor.predict(input_df)
    
    # --- 2. Visualization Prep (Instance Norm for Plotting) ---
    # We replicate the model's view: Instance Normalization
    X_viz = input_df.values.T # (C, T)
    mean = X_viz.mean(axis=1, keepdims=True)
    std = X_viz.std(axis=1, keepdims=True) + 1e-6
    X_norm = (X_viz - mean) / std
    
    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Define Colormap (Deep Purple -> White -> Dark Orange) matching project style
    colors = ["#4B0082", "white", "#FF8C00"]
    cmap = LinearSegmentedColormap.from_list("custom_purple_orange", colors)
    
    sns.heatmap(
        X_norm, 
        cmap=cmap, 
        center=0, 
        robust=True, # Robust quantile based contrast
        cbar_kws={"label": "Z-Score (Instance Norm)"},
        yticklabels=channels,
        xticklabels=350, # Mark every 10s (35Hz) for cleaner axis
        ax=ax
    )
    
    # Format X-axis
    ax.set_xlabel("Time Samples (60s @ 35Hz)")
    
    # Annotations
    # Handle label mapping if 'label_str' exists, else map integer
    true_label = "Unknown"
    if 'label_str' in window_row:
        true_label = window_row['label_str']
    else:
        # Try numeric
        l = window_row.get('label')
        if l is not None:
             label_map = {1: 'Baseline', 2: 'Stress'}
             true_label = label_map.get(l, str(l))

    pred_label = result['prediction']
    conf = result['confidence']
    status = result['status']
    
    # Color code title based on correctness
    # Deep Purple for Correct, Dark Red for Error, Gray for Abstain
    if status == 'Abstained':
        title_color = 'gray'
        res_text = f"ABSTAINED (Conf: {conf:.1%})"
    elif pred_label == true_label:
        title_color = '#4B0082' # Deep Purple
        res_text = f"CORRECT: {pred_label} (Conf: {conf:.1%})"
    else:
        title_color = 'darkred' 
        res_text = f"INCORRECT: Pred {pred_label} vs True {true_label} (Conf: {conf:.1%})"

    ax.set_title(f"Inference Audit | Ground Truth: {true_label} | Result: {res_text}", 
                 color=title_color, fontsize=14, fontweight='bold', pad=15)
    
# Add props text box removal logic
    # The text box was deemed clutter. If we need stats, we should print them in the notebook
    # or add them to the title concisely.
    # For now, we simply remove the ax.text block.
    
    # We also remove the stats_lines construction since it is no longer used, unless we return it.
    
    plt.tight_layout()
    
    if save_folder:
        # Sanitize filename for _save_plot
        fname = f"inference_audit_True_{true_label}_Pred_{pred_label}_{conf:.2f}"
        _save_plot(fig, fname, save_folder)
    
    return fig
