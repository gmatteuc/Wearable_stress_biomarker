"""
Signal Quality Index (SQI) Module
=================================

This module implements rule-based and statistical checks to assess the quality
of wearable sensor data (EDA, ACC, etc.). It flags windows with missing data,
flatlines (sensor disconnect), or high-intensity motion artifacts.
"""

import numpy as np
from typing import Dict, Any

class SignalQualityIndex:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Load SQI settings from config if available (features -> sqi)
        sqi_cfg = self.config.get('features', {}).get('sqi', {})
        
        self.acc_noise_threshold = sqi_cfg.get('motion_threshold_g', 0.5)
        
        # 1. Dead Sensor (Flatline) Thresholds
        # Each signal has a different unit/scale, so we need specific thresholds.
        defaults_flatline = {
            'EDA': 0.005,   # uS (MicroSiemens)
            'ECG': 0.05,    # mV (Millivolts)
            'RESP': 0.5,    # Raw ADC
            'TEMP': 0.01    # °C
        }
        self.flatline_thresholds = {**defaults_flatline, **sqi_cfg.get('flatline_thresholds', {})}
        
        # 2. Physiological Plausibility Range
        # Used to detect if sensor is just outputting zeros or rails
        defaults_ranges = {
            'TEMP': [20.0, 45.0], # Degrees Celsius
            'EDA':  [0.01, 100.0] # uS
        }
        self.valid_ranges = {**defaults_ranges, **sqi_cfg.get('valid_ranges', {})}
        
        # Overall Acceptance Threshold
        self.acceptance_threshold = sqi_cfg.get('acceptance_threshold', 0.5)
    
    def compute_sqi(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute SQI scores for a window.
        Returns a dict of scores (0 to 1, where 1 is best) or flags.
        """
        sqi_scores = {}
        
        # 1. Missing data check
        sqi_scores['completeness'] = 1.0
        for key, sig in window_data.items():
            if isinstance(sig, list) and not sig:
                 sqi_scores['completeness'] = 0.0
                 break
            if isinstance(sig, list) and np.isnan(sig).any():
                 sqi_scores['completeness'] = 0.0
                 break

        # 2. Signal-Specific Quality Checks
        # Connectivity (Flatline) & Range Checks
        
        # EDA Check
        if 'EDA' in window_data and sqi_scores['completeness'] == 1.0:
            eda = np.array(window_data['EDA'])
            eda_range = np.max(eda) - np.min(eda)
            if eda_range < self.flatline_thresholds['EDA']:
                sqi_scores['eda_sqi'] = 0.0
            elif np.mean(eda) < self.valid_ranges['EDA'][0] or np.mean(eda) > self.valid_ranges['EDA'][1]:
                 sqi_scores['eda_sqi'] = 0.0
            else:
                sqi_scores['eda_sqi'] = 1.0

        # ECG Check
        if 'ECG' in window_data and sqi_scores['completeness'] == 1.0:
            ecg = np.array(window_data['ECG'])
            ecg_range = np.max(ecg) - np.min(ecg)
            if ecg_range < self.flatline_thresholds['ECG']:
                sqi_scores['ecg_sqi'] = 0.0
            else:
                sqi_scores['ecg_sqi'] = 1.0

        # Temperature Check
        if 'TEMP' in window_data and sqi_scores['completeness'] == 1.0:
             temp = np.array(window_data['TEMP'])
             temp_range = np.max(temp) - np.min(temp)
             
             # Check 1: Range (Is it disconnected/stuck?)
             if temp_range < self.flatline_thresholds['TEMP']:
                 sqi_scores['temp_sqi'] = 0.0
             # Check 2: Plausible Biological Range (20-45C)
             elif np.mean(temp) < self.valid_ranges['TEMP'][0] or np.mean(temp) > self.valid_ranges['TEMP'][1]:
                 sqi_scores['temp_sqi'] = 0.0
             else:
                 sqi_scores['temp_sqi'] = 1.0
                
        # 3. Motion Artifacts from ACC
        if 'ACC_x' in window_data and sqi_scores['completeness'] == 1.0:
            acc_x = np.array(window_data['ACC_x'])
            acc_y = np.array(window_data['ACC_y'])
            acc_z = np.array(window_data['ACC_z'])
            
            # Magnitude
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            acc_std = np.std(acc_mag)
            
            # High variance = high motion
            # Heuristic map: std > threshold -> low quality for EDA/ECG analysis
            sqi_scores['motion_score'] = max(0.0, 1.0 - (acc_std / self.acc_noise_threshold))
        
        # Aggregate SQI (simple geometric mean or min)
        # We define 'is_good' boolean
        scores = [v for k, v in sqi_scores.items()]
        sqi_scores['overall_sqi'] = np.mean(scores) if scores else 0.0
        
        return sqi_scores

    def is_acceptable(self, window_data: Dict[str, Any], threshold: float = None) -> bool:
        scores = self.compute_sqi(window_data)
        limit = threshold if threshold is not None else self.acceptance_threshold
        return scores['overall_sqi'] >= limit
