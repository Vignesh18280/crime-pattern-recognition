"""
pattern_generator.py
--------------------
Generates multiple overlapping 5-event crime patterns from suspicious connections.
"""

import numpy as np
import pandas as pd

def generate_multiple_crime_patterns(suspicious_df, window_size=5, stride=1):
    """
    Generate overlapping 5-event patterns from suspicious data.
    """
    patterns = []
    num_events = len(suspicious_df)
    
    if num_events < window_size:
        return []
    
    events_array = suspicious_df.values
    
    for i in range(0, num_events - window_size + 1, stride):
        pattern = events_array[i : i + window_size]
        patterns.append(pattern)
            
    return patterns
