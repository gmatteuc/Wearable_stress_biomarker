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
        # Load EDA threshold
        self.eda_threshold = self.config.get('features', {}).get('eda', {}).get('scr_threshold', 0.015)

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
            
        # 5. Respiration Features (Chest Only)
        if 'RESP' in window:
            features.update(self._process_resp(np.array(window['RESP'])))

        # 6. BVP Features (Wrist Only)
        if 'BVP' in window:
            features.update(self._process_bvp(np.array(window['BVP'])))

        return features

    def process_windows(self, windows_data) -> "pd.DataFrame":
        """
        Process a batch of windows (DataFrame or list of dicts).
        Returns a DataFrame of features.
        """
        import pandas as pd
        
        # Convert DataFrame to list of dicts if needed
        if isinstance(windows_data, pd.DataFrame):
            windows_list = windows_data.to_dict('records')
        else:
            windows_list = windows_data
            
        features_list = []
        for w in windows_list:
            # Extract features
            feats = self.extract_features(w)
            
            # Preserve metadata
            meta = {k: v for k, v in w.items() if k in ['subject_id', 'label', 'session', 'start_idx']}
            
            features_list.append({**meta, **feats})
            
        return pd.DataFrame(features_list)

    def get_eda_components(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Public helper to decompose EDA for visualization/audit.
        Returns: (tonic, phasic, peaks_indices)
        """
        b, a = signal.butter(3, 0.05 / (0.5 * self.fs), btype='low')
        tonic = signal.filtfilt(b, a, sig)
        phasic = sig - tonic 
        
        # Lowered prominence threshold to capture smaller SCRs (configured in default.yaml)
        peaks, _ = signal.find_peaks(phasic, prominence=self.eda_threshold, distance=self.fs)
        return tonic, phasic, peaks

    def _process_eda(self, sig: np.ndarray) -> Dict[str, float]:
        """
        EDA Decomposition:
        - Tonic: Slowly varying baseline (Low-pass filter < 0.05Hz)
        - Phasic: Rapid skin conductance responses (High-frequency residual)
        """
        # Use helper to ensure consistency with audits
        tonic, phasic, peaks = self.get_eda_components(sig)
        
        # 2. Extract Features
        feats = {
            'eda_mean': np.mean(sig),
            'eda_std': np.std(sig),
            'eda_tonic_mean': np.mean(tonic),
            'eda_phasic_std': np.std(phasic),
            'eda_range': np.max(sig) - np.min(sig)
        }
        
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
        - Dynamic features only (Std, Range) to remove static gravity/posture bias.
        """
        ax = np.array(window['ACC_x'])
        ay = np.array(window['ACC_y'])
        az = np.array(window['ACC_z'])
        
        # Magnitude (Modulus)
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        feats = {
            # We explicitly REMOVE 'acc_mean' because it captures the static gravity vector (1g),
            # which is sensitive to sensor calibration and posture (tilt) rather than stress/activity.
            
            'acc_std': np.std(mag),     # Best proxy for Activity Intensity (AI)
            'acc_range': np.max(mag) - np.min(mag), # Peak-to-peak amplitude
            'acc_energy': np.sum(np.square(mag - np.mean(mag))) / len(mag) # Dynamic Energy
        }
        return feats

    def get_ecg_peaks(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Public helper to get ECG peaks for visualization/audit.
        Returns: (clean_ecg, peaks_indices, threshold_used)
        Only detects positive R-peaks (no negative artifacts).
        """
        b, a = signal.butter(3, [1.0 / (0.5 * self.fs), 15.0 / (0.5 * self.fs)], btype='band')
        clean_ecg = signal.filtfilt(b, a, sig)
        # Dynamic threshold: mean + 1.5*std (positive peaks only)
        # Increased from 0.5 to 1.5 to reduce false positives (T-waves/noise)
        peak_threshold = np.mean(clean_ecg) + 1.5 * np.std(clean_ecg)
        peaks, _ = signal.find_peaks(clean_ecg, height=peak_threshold, distance=int(0.35 * self.fs))
        return clean_ecg, peaks, peak_threshold

    def _process_ecg(self, sig: np.ndarray) -> Dict[str, float]:
        """
        ECG Features from R-Peak detection (Time Domain only for short windows).
        Note: At 35Hz, this is an approximation.
        """
        # Use helper to ensure consistency
        _, peaks, _ = self.get_ecg_peaks(sig)
        
        # 4. Calculate HR and HRV
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

    def _process_resp(self, sig: np.ndarray) -> Dict[str, float]:
        """
        Respiration Features (Chest):
        - Statistical moments.
        - Estimation of Respiration Rate via FFT (Dominant frequency in 0.1-0.5Hz).
        """
        feats = {
            'resp_mean': np.mean(sig),
            'resp_std': np.std(sig),
            'resp_range': np.max(sig) - np.min(sig)
        }
        
        # Estimate Respiration Rate using Power Spectral Density
        # Hanning window to reduce leakage
        f, Pxx = signal.welch(sig, fs=self.fs, nperseg=len(sig), window='hann')
        
        # Focus on physiological range: 0.1Hz (6 bpm) to 0.5Hz (30 bpm)
        valid_mask = (f >= 0.1) & (f <= 0.5)
        if np.any(valid_mask):
            f_valid = f[valid_mask]
            Pxx_valid = Pxx[valid_mask]
            
            # Robust Peak Detection
            # We look for peaks with at least 20% of the max power to be candidates
            max_power = np.max(Pxx_valid)
            peak_indices, properties = signal.find_peaks(Pxx_valid, height=max_power * 0.20)
            
            if len(peak_indices) > 0:
                peak_freqs = f_valid[peak_indices]
                peak_heights = properties['peak_heights']
                
                # Start with the highest peak as the candidate
                best_idx = np.argmax(peak_heights)
                dominant_f = peak_freqs[best_idx]
                
                # Check for "Octave Errors" (Subharmonics)
                # If the dominant peak is actually the 2nd harmonic, 
                # there should be a significant peak at roughly 1/2 frequency.
                # We check if any other candidate peak is close to dominant_f / 2
                potential_fundamental = dominant_f / 2.0
                
                # Log debug info if needed (print for notebook audit)
                # print(f"Peaks: {peak_freqs}, Heights: {peak_heights}")
                
                for i, f_p in enumerate(peak_freqs):
                    if i == best_idx: continue
                    
                    # Check if within 10% tolerance or 0.04Hz of expected fundamental
                    # Using a slightly wider window to catch spectral leakage shifts
                    if abs(f_p - potential_fundamental) < max(0.04, 0.1 * potential_fundamental):
                        # Verify the harmonic is significant enough (at least 40% of the dominant peak)
                        # The 0.20 threshold above already filters for 20% of max, so it's a candidate.
                        # But we double check relative strength.
                        if peak_heights[i] > 0.4 * peak_heights[best_idx]:
                            dominant_f = f_p
                            break
                            
                # Check for 3rd Harmonic case (less common in Resp but possible)
                # e.g., 0.33Hz is 3rd harmonic of 0.11Hz
                potential_fundamental_3 = dominant_f / 3.0
                 # Simple check if there is a peak there
                for i, f_p in enumerate(peak_freqs):
                     if abs(f_p - potential_fundamental_3) < 0.03 and peak_heights[i] > 0.3 * peak_heights[best_idx]:
                        dominant_f = f_p
                        break

                feats['resp_rate_Hz'] = dominant_f
                feats['resp_rate_bpm'] = dominant_f * 60.0
            else:
                # Fallback to simple argmax
                dominant_freq = f_valid[np.argmax(Pxx_valid)]
                feats['resp_rate_Hz'] = dominant_freq
                feats['resp_rate_bpm'] = dominant_freq * 60.0
        else:
            feats['resp_rate_Hz'] = 0.0
            feats['resp_rate_bpm'] = 0.0
            
        return feats

    def _process_bvp(self, sig: np.ndarray) -> Dict[str, float]:
        """
        BVP Features (Wrist):
        - Similar to ECG, BVP measures pulse.
        - We extract statistical approximations.
        """
        # BVP is often quite clean. Peak detection for HR.
        # Bandpass filter for PPG: 0.5Hz - 8Hz
        b, a = signal.butter(3, [0.5 / (0.5 * self.fs), 8.0 / (0.5 * self.fs)], btype='band')
        clean_bvp = signal.filtfilt(b, a, sig)
        
        # Stat features
        feats = {
            'bvp_mean': np.mean(sig),
            'bvp_std': np.std(sig)
        }
        
        # Peak Detection for HR estimate (Wrist)
        # Squaring for robustness
        sq_bvp = clean_bvp ** 2
        peak_threshold = np.mean(sq_bvp) * 1.5 # BVP peaks are less sharp than ECG
        peaks, _ = signal.find_peaks(sq_bvp, height=peak_threshold, distance=int(0.35 * self.fs))
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.fs
            feats['bvp_hr_est'] = 60.0 / np.mean(rr_intervals)
        else:
            feats['bvp_hr_est'] = 0.0
            
        return feats

