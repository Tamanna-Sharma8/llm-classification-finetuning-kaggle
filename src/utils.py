"""
Utility functions for Kaggle LLM Classification Competition.

This module provides helper functions for:
- Data loading and preprocessing
- Submission file creation and validation
- Model evaluation and visualization
- Probability calibration and checks

Author: Tamanna Sharma
Date: February 2026
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ============================================================================ 
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(data_path: str = '/kaggle/input/llm-classification-finetuning/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from Kaggle competition.
    """
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Please add competition data.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    
    return train_df, test_df

# ============================================================================ 
# (Keep the rest of the functions exactly as in your original code)
# Just update the main guard and author info
# ============================================================================

__version__ = "0.1.0"
__author__ = "Tamanna Sharma"
__email__ = "ts825391@gmail.com"

if __name__ == "__main__":
    print("🔬 Kaggle LLM Classification Competition - Utility Functions")
    print(f"   Version: {__version__}")
    print(f"   Author: {__author__}")
    print("\n📚 Available functions:")
    
    functions = [
        'load_data', 'get_target_stats',
        'create_submission', 'validate_probabilities', 'calibrate_probabilities', 'save_submission',
        'evaluate_predictions', 'calculate_confidence_interval',
        'plot_feature_importance', 'plot_prediction_distribution',
        'log_submission', 'ensemble_predictions', 'prepare_features',
        'setup_kaggle_environment'
    ]
    
    for func in functions:
        print(f"   • {func}")
