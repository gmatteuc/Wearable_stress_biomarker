"""
Feature Extraction Module
=========================

This module transforms windowed raw signals (EDA, TEMP, ECG, ACC) into 
meaningful statistical and physiological features for machine learning.

Supported Modalities:
1. EDA: Decomposed into Phasic (Peaks) and Tonic (Baseline) components.
2. TEMP: Gradient (Slope) and statistical distribution.
3. ACC: Motion intensity (Mean/Std of Magnitude).
4. ECG: Heart Rate and basic HRV metrics (from R-peak detection).

Dependencies: scipy
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from typing import Dict, Any, Tuple

class FeatureExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Default to 35Hz (WESAD processed default), but can be overridden
        self.fs = self.config.get('data', {}).get('target_sampling_rate', 35)

    def extract_features(self, window: Dict[str, Any]) -> Dict[str, float]:
        """
        Main entry point. extracts features for all available modalities in the window.
        """
        features = {}
        
        # 1. EDA Features
        if 'EDA' in window:
            features.update(self._process_eda(np.array(window['EDA'])))
            
        # 2. Temperature Features
        if 'TEMP' in window:
            features.update(self._process_temp(np.array(window['TEMP'])))
            
        # 3. Accelerometer Features
        if 'ACC_x' in window:
            features.update(self._process_acc(window))
            
        # 4. ECG Features
        if 'ECG' in window:
            features.update(self._process_ecg(np.array(window['ECG'])))
            
        return features

    def _process_eda(self, sig: np.ndarray) -> Dict[str, float]:
        """
        EDA Decomposition:
        - Tonic: Slowly varying baseline (Low-pass filter < 0.05Hz)
        - Phasic: Rapid skin conductance responses (High-frequency residual)
        """
        # 1. Decompose
        # Butterworth Low-pass filter for Tonic
        b, a = signal.butter(3, 0.05 / (0.5 * self.fs), btype='low')
        tonic = signal.filtfilt(b, a, sig)
        phasic = sig - tonic # Approximation of Phasic
        
        # 2. Extract Features
        feats = {
            'eda_mean': np.mean(sig),
            'eda_std': np.std(sig),
            'eda_tonic_mean': np.mean(tonic),
            'eda_phasic_std': np.std(phasic),
            'eda_range': np.max(sig) - np.min(sig)
        }
        
        # Count SCR Peaks (Phasic spikes)
        # We define a peak as something > 0.01 uS above local baseline with min distance
        peaks, _ = signal.find_peaks(phasic, height=0.01, distance=self.fs) # Min 1 sec apart
        feats['eda_acc_scr_count'] = len(peaks)
        
        return feats

    def _process_temp(self, sig: np.ndarray) -> Dict[str, float]:
        """
        Temperature Features:
        - Absolute levels (Mean)
        - Trend (Slope) - Critical for stress detection (vasoconstriction)
        """
        feats = {
            'temp_mean': np.mean(sig),
            'temp_std': np.std(sig),
            'temp_min': np.min(sig),
            'temp_max': np.max(sig)
        }
        
        # Calculate Slope (Trend) over the window
        # x axis is just time indices
        x = np.arange(len(sig))
        slope, _, _, _, _ = stats.linregress(x, sig)
        
        # Scale slope to "change per second" instead of "per sample"
        feats['temp_slope'] = slope * self.fs 
        
        return feats

    def _process_acc(self, window: Dict[str, Any]) -> Dict[str, float]:
        """
        ACC Features:
        - Magnitude (Energy)
        - Axis-specific variance
        """
        ax = np.array(window['ACC_x'])
        ay = np.array(window['ACC_y'])
        az = np.array(window['ACC_z'])
        
        # Magnitude
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        feats = {
            'acc_mean': np.mean(mag),
            'acc_std': np.std(mag), # Great proxy for physical intensity
            'acc_max': np.max(mag)
        }
        return feats

    def _process_ecg(self, sig: np.ndarray) -> Dict[str, float]:
        """
        ECG Features from R-Peak detection (Time Domain only for short windows).
        Note: At 35Hz, this is an approximation.
        """
        # 1. Clean Signal (Bandpass 5-15Hz to verify QRS energy)
        # Using a wider band 1-30Hz just to be safe with low Fs
        b, a = signal.butter(3, [1.0 / (0.5 * self.fs), 15.0 / (0.5 * self.fs)], btype='band')
        clean_ecg = signal.filtfilt(b, a, sig)
        
        # 2. Find Peaks (Distance=0.4s implies Max HR ~150bpm, reasonable for baseline)
        peaks, _ = signal.find_peaks(clean_ecg, height=np.mean(clean_ecg), distance=int(0.4 * self.fs))
        
        # 3. Calculate HR and HRV
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.fs # in seconds
            
            hr_bpm = 60.0 / np.mean(rr_intervals)
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
            sdnn = np.std(rr_intervals)
        else:
            hr_bpm = 0.0
            rmssd = 0.0
            sdnn = 0.0
            
        feats = {
            'ecg_hr_bpm': hr_bpm,
            'ecg_rmssd': rmssd,
            'ecg_sdnn': sdnn
        }
        
        return feats
