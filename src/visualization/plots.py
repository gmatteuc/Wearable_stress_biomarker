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

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import itertools
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

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

def plot_raw_signals(
    time_axis: np.ndarray, 
    signals: Dict[str, np.ndarray], 
    title: str = "Raw Sensor Signals"
):
    """
    Generates a multi-row subplot for synchronized physiological signals.
    
    Args:
        time_axis: Common time axis in seconds.
        signals: Dictionary mapping modality name (str) to signal array (np.ndarray).
        title: Overall figure title.
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
    return fig

def plot_rel_diagram(y_true, y_prob, title="Reliability Diagram"):
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
    
    return fig

def plot_resampling_comparison(
    raw_t: np.array, 
    raw_signal: np.array, 
    res_t: np.array, 
    res_signal: np.array,
    raw_fs: int,
    target_fs: int,
    title: str = "Resampling Check: Signal Fidelity"
) -> plt.Figure:
    """
    Plots an overlay of the original high-frequency signal and the resampled/windowed version
    to demonstrate signal fidelity.

    Args:
        raw_t: Time axis for raw signal.
        raw_signal: Amplitude of raw signal.
        res_t: Time axis for resampled signal.
        res_signal: Amplitude of resampled signal.
        raw_fs: Sampling rate of raw signal (Hz).
        target_fs: Sampling rate of resampled signal (Hz).
        title: Plot title.
    
    Returns:
        plt.Figure: The generated figure.
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Plot Raw as a thicker, semi-transparent line (Primary Theme Color)
    plt.plot(
        raw_t, 
        raw_signal, 
        label=f'Raw Original ({raw_fs} Hz)', 
        alpha=0.6, 
        linewidth=3, 
        color='#4B0082'
    )
    
    # Plot Resampled as a dashed contrast line on top (Contrast Theme Color)
    plt.plot(
        res_t, 
        res_signal, 
        label=f'Resampled Window ({target_fs} Hz)', 
        linewidth=2, 
        linestyle='--', 
        color='#FF8C00', 
        marker='o', 
        markersize=4
    )
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    
    return fig

def plot_resampling_verification_grid(
    chest_data: Dict[str, np.ndarray],
    sample_window: Dict[str, np.ndarray],
    start_time: float,
    raw_fs: int = 700,
    target_fs: int = 35,
    plot_duration: int = 30
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
    plt.suptitle(f"Pipeline Verification: Signal Fidelity (Aligned at t={start_time:.1f}s)", y=1.02, fontsize=16)
    plt.tight_layout()
    return fig

def plot_sqi_comparison(scenarios: List[tuple], signal_key: str = 'EDA', thresholds: Dict[str, float] = None) -> plt.Figure:
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

    plt.suptitle(f"Signal Quality Diagnostics: {signal_key} vs Motion", fontsize=16, y=1.05)
    return fig
