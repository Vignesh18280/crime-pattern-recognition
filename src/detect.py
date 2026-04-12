"""
detect.py
-----------
This module provides functions for anomaly detection in network traffic logs.
It uses statistical methods to identify and isolate suspicious connections
from a mixed stream of normal and attack traffic.
"""

import pandas as pd
import numpy as np

def detect_suspicious_connections(df: pd.DataFrame, n_std: float = 2.0) -> pd.DataFrame:
    """
    Identifies suspicious connections in a DataFrame using statistical anomaly detection.

    A connection is flagged as suspicious if any of its numerical feature values
    fall outside the range of [mean - n_std * std_dev, mean + n_std * std_dev].

    Args:
        df (pd.DataFrame): The input DataFrame containing network traffic data.
        n_std (float): The number of standard deviations to use as the threshold for outliers.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows flagged as suspicious.
    """
    if df.empty:
        return pd.DataFrame()

    # Select only numeric features for statistical analysis
    df_numeric = df.select_dtypes(include=[np.number])

    # Drop columns that are numeric but shouldn't be used for anomaly detection
    # (e.g., identifiers or binary flags that are already encoded)
    # Based on dataset.py, 'id' and 'is_sm_ips_ports' might be irrelevant.
    # For a general purpose function, we may need to be more sophisticated,
    # but for this project, let's assume most numeric cols are relevant.
    cols_to_ignore = ['id', 'is_sm_ips_ports', 'label']
    cols_to_analyse = [c for c in df_numeric.columns if c not in cols_to_ignore]
    
    if not cols_to_analyse:
        print("[detect] No numeric columns found to analyze for anomalies.")
        return pd.DataFrame()
        
    df_analyze = df_numeric[cols_to_analyse]

    # Calculate mean and standard deviation for each feature
    mean = df_analyze.mean()
    std = df_analyze.std()

    # Identify outliers - any feature that is more than n_std from the mean
    # The .any(axis=1) flags a row if at least one feature is an outlier.
    is_outlier = (np.abs(df_analyze - mean) > n_std * std).any(axis=1)

    suspicious_df = df[is_outlier]

    print(f"[detect] Found {len(suspicious_df)} suspicious connections out of {len(df)} total.")

    return suspicious_df
