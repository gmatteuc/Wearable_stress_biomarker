import pytest
import numpy as np
from src.models.train import Trainer

def test_loso_split_no_leakage():
    # Mock data
    groups = np.array(['S1']*10 + ['S2']*10 + ['S3']*10)
    
    # Instantiate trainer (mock config)
    trainer = Trainer(model_type='logistic', split_type='loso')
    
    train_idx, test_idx = trainer.get_split(groups)
    
    train_subjects = np.unique(groups[train_idx])
    test_subjects = np.unique(groups[test_idx])
    
    # Intersection must be empty
    intersection = np.intersect1d(train_subjects, test_subjects)
    assert len(intersection) == 0, f"Subject leakage detected: {intersection}"
