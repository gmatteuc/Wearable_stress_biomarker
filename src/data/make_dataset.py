"""
WESAD Data Preprocessing Module
===============================

This module handles the ingestion, parsing, and windowing of the WESAD dataset.
It normalizes sampling rates, synchronizes modalities, and segments continuous
recordings into fixed-length windows for downstream machine learning tasks.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import signal
from typing import Dict, List, Tuple
import joblib

from src.config import load_config, PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WESADPreprocessor:
    def __init__(self):
        self.config = load_config()
        self.raw_path = PROJECT_ROOT / self.config['data']['raw_path'] / "WESAD"
        self.processed_path = PROJECT_ROOT / self.config['data']['processed_path']
        self.target_fs = self.config['data']['target_sampling_rate']
        self.window_size = self.config['data']['window_size_sec']
        self.overlap = self.config['data']['window_overlap_sec']
        self.target_labels = self.config['data']['target_labels']
        self.subjects = [f"S{i}" for i in range(2, 18)]
        
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_subject(self, subject_id: str) -> Dict:
        file_path = self.raw_path / subject_id / f"{subject_id}.pkl"
        logger.info(f"Loading {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def resample_signal(self, sig: np.array, original_fs: int) -> np.array:
        """Resample signal to target_fs using FFT."""
        if original_fs == self.target_fs:
            return sig
        
        num_samples = int(len(sig) * self.target_fs / original_fs)
        return signal.resample(sig, num_samples)

    def process_subject(self, subject_id: str) -> List[Dict]:
        data = self.load_subject(subject_id)
        location = self.config['data']['sensor_location']
        labels = data['label'] # Always 700 Hz
        
        signals = {}
        
        if location == 'chest':
            source_data = data['signal']['chest']
            # Chest Modalities: All 700Hz
            # ACC(3), ECG(1), EDA(1), RESP(1), TEMP(1)
            
            def process_chest(key, dims):
                raw = source_data[key].astype(float)
                if dims == 1:
                    return self.resample_signal(raw.flatten(), 700)
                else:
                    return np.stack([self.resample_signal(raw[:, i], 700) for i in range(dims)], axis=1)

            signals['ACC_x'] = process_chest('ACC', 3)[:, 0]
            signals['ACC_y'] = process_chest('ACC', 3)[:, 1]
            signals['ACC_z'] = process_chest('ACC', 3)[:, 2]
            signals['ECG']   = process_chest('ECG', 1)
            signals['EDA']   = process_chest('EDA', 1)
            signals['RESP']  = process_chest('Resp', 1)
            signals['TEMP']  = process_chest('Temp', 1)
            
        elif location == 'wrist':
            source_data = data['signal']['wrist']
            # Wrist Modalities: Different Sampling Rates
            # ACC(32), BVP(64), EDA(4), TEMP(4)
            
            def process_wrist(key, dims, original_fs):
                raw = source_data[key].astype(float)
                if dims == 1:
                    return self.resample_signal(raw.flatten(), original_fs)
                else:
                    return np.stack([self.resample_signal(raw[:, i], original_fs) for i in range(dims)], axis=1)

            # Note: WESAD Wrist dictionary keys might differ slightly (BVP vs bvp?), normally Caps in WESAD pickle
            # Based on standard WESAD structure: 'ACC', 'BVP', 'EDA', 'TEMP'
            
            signals['ACC_x'] = process_wrist('ACC', 3, 32)[:, 0]
            signals['ACC_y'] = process_wrist('ACC', 3, 32)[:, 1]
            signals['ACC_z'] = process_wrist('ACC', 3, 32)[:, 2]
            signals['BVP']   = process_wrist('BVP', 1, 64)
            signals['EDA']   = process_wrist('EDA', 1, 4)
            signals['TEMP']  = process_wrist('TEMP', 1, 4)
            
        else:
            raise ValueError(f"Unknown sensor location: {location}")
        
        # Resample labels (using nearest to keep integer classes)
        # labels are 700Hz.
        num_samples = len(signals['ACC_x'])
        # Map labels to 0..N indices correctly time-aligned
        # We can implement nearest neighbor via simple indexing
        original_indices = np.linspace(0, len(labels)-1, num_samples).astype(int)
        resampled_labels = labels[original_indices]
        
        # Windowing
        windows = []
        window_len = self.target_fs * self.window_size
        step = self.target_fs * (self.window_size - self.overlap)
        
        for start in range(0, num_samples - window_len, int(step)):
            end = start + window_len
            seg_label = pd.Series(resampled_labels[start:end]).mode()[0]
            
            if seg_label not in self.target_labels:
                continue
                
            window_dict = {
                'subject_id': subject_id,
                'label': int(seg_label),
                'start_idx': start
            }
            
            for k, v in signals.items():
                window_dict[k] = v[start:end].tolist()
                
            windows.append(window_dict)
            
        logger.info(f"Subject {subject_id}: {len(windows)} windows created.")
        return windows

    def run(self):
        all_windows = []
        for subject in self.subjects:
            try:
                windows = self.process_subject(subject)
                all_windows.extend(windows)
            except Exception as e:
                logger.error(f"Failed to process {subject}: {e}")
        
        df = pd.DataFrame(all_windows)
        output_file = self.processed_path / "windows.parquet"
        
        # Save as parquet (pyarrow handles array columns)
        logger.info(f"Saving {len(df)} windows to {output_file}")
        df.to_parquet(output_file, index=False)
        logger.info("Processing complete.")

if __name__ == "__main__":
    WESADPreprocessor().run()
