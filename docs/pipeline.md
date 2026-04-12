# Processing Pipeline
### End-to-End Data Flow from Upload to Verdict

---

This document explains the complete processing pipeline — how data flows from a user's uploaded incident log file to the final similarity verdict.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Step 1: File Upload](#2-step-1-file-upload)
3. [Step 2: Anomaly Detection](#3-step-2-anomaly-detection)
4. [Step 3: Pattern Generation](#4-step-3-pattern-generation)
5. [Step 4: Preprocessing](#5-step-4-preprocessing)
6. [Step 5: Tensor Conversion](#6-step-5-tensor-conversion)
7. [Step 6: Best Match Search](#7-step-6-best-match-search)
8. [Step 7: Verdict Generation](#8-step-7-verdict-generation)
9. [Step 8: SHAP Explanation](#9-step-8-shap-explanation)
10. [Complete Flow Diagram](#10-complete-flow-diagram)

---

## 1. Pipeline Overview

The processing pipeline transforms raw incident logs into a similarity verdict:

```
Incident Log A     Incident Log B
      │                  │
      ▼                  ▼
File Upload ──► Anomaly Detection ──► Pattern Generation ──► Preprocessing
                                                                   │
                                    All Combinations ◄─────────────┘
                                                                   │
                                    Best Match ◄─────────────────────┘
                                                                   │
                                    Verdict Generation ◄──────────────┘
                                                                   │
                                    SHAP Explanation ◄──────────────┘
                                                                   │
                                    Final Result
```

---

## 2. Step 1: File Upload

### 2.1 What Happens

The user uploads two CSV files containing network traffic logs. Each file represents a different cyber incident.

### 2.2 Expected File Format

```csv
# Sample incident log format (CSV)
# Each row represents one network connection/event

dur,sbytes,dbytes,sttl,dttl,spkts,dpkts,tcprtt,synack,ackdat,...
0.0,264,0,64,64,4,4,0.0,0.0,0.0,...
0.1,1500,0,64,64,3,3,0.0,0.0,0.0,...
0.2,2000,500,64,64,5,4,0.1,0.05,0.02,...
...
```

### 2.3 Server-Side Handling

```python
# From app.py

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload.
    Accepts CSV or Parquet files.
    Saves to temporary directory.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.parquet')):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save file
    file.save(filepath)
    
    return jsonify({"filename": filename})
```

### 2.4 Output

| Output | Description |
|---|---|
| Saved file path | `data/temp_uploads/{uuid}_{filename}` |
| Returned | `{filename: "uuid_filename.csv"}` |

---

## 3. Step 2: Anomaly Detection

### 3.1 What Happens

From all connections in the incident log, we identify which ones are **suspicious** (potentially part of an attack) versus normal traffic.

### 3.2 Why This Matters

Not all connections in a log are attack-related. We need to isolate the malicious activity to generate meaningful crime patterns.

### 3.3 Detection Algorithm

We use **statistical anomaly detection** based on standard deviation thresholds:

```python
# From src/detect.py

def detect_suspicious_connections(df: pd.DataFrame, n_std: float = 2.0) -> pd.DataFrame:
    """
    Identifies suspicious connections using statistical thresholds.
    
    A connection is flagged as suspicious if any of its numerical 
    feature values fall outside [mean - n_std * std, mean + n_std * std].
    
    Args:
        df: Input DataFrame with network traffic
        n_std: Number of standard deviations for threshold
    
    Returns:
        DataFrame containing only suspicious rows
    """
    # Select numeric columns only
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Columns to ignore (identifiers, not features)
    cols_to_ignore = ['id', 'is_sm_ips_ports', 'label']
    cols_to_analyse = [c for c in df_numeric.columns if c not in cols_to_ignore]
    
    if not cols_to_analyse:
        return pd.DataFrame()
    
    suspicious_mask = pd.Series([False] * len(df), index=df.index)
    
    # Check each feature
    for col in cols_to_analyse:
        if df_numeric[col].nunique() <= 1:
            continue  # Skip constant features
        
        mean = df_numeric[col].mean()
        std = df_numeric[col].std()
        
        if std > 0:
            # Flag if outside n_std standard deviations
            lower = mean - n_std * std
            upper = mean + n_std * std
            
            col_suspicious = (df_numeric[col] < lower) | (df_numeric[col] > upper)
            suspicious_mask = suspicious_mask | col_suspicious
    
    return df[suspicious_mask]
```

### 3.4 Visual Example

```
Original connections (100 total):
┌────────────────────────────────────────────────────────────────┐
│ Connection 1:  dur=0.0,  sbytes=264     ← suspicious │
│ Connection 2:  dur=0.1,  sbytes=1500   ← suspicious │
│ Connection 3:  dur=1.5,  sbytes=500    ← normal     │
│ Connection 4:  dur=2.0,  sbytes=800    ← normal     │
│ Connection 5:  dur=0.0,  sbytes=9999    ← suspicious │
│ ...                                           │
└────────────────────────────────────────────────────────────────┘
                    │
                    ▼ Statistical threshold: mean ± 2 std
                    ▼
Suspicious connections (15 total):
┌────────────────────────────────────────────────────────────────┐
│ Connection 1, 2, 5, 12, 34, 56, 78, 89, 95, ...
└────────────────────────────────────────────────────────────────┘
```

### 3.5 Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_std` | 2.0 | Standard deviations for threshold |
| Minimum connections | 5 | At least 5 suspicious needed |

---

## 4. Step 3: Pattern Generation

### 4.1 What Happens

From the suspicious connections, we generate **overlapping 5-event patterns** — groups of 5 consecutive network events that represent a multi-stage attack sequence.

### 4.2 Why 5 Events?

5 events capture a typical attack chain:
1. Initial access
2. Execution
3. Persistence
4. Lateral movement (optional)
5. Objective (data theft, damage)

This is enough to reveal the attacker's MO without being too long.

### 4.3 Generation Algorithm

```python
# From src/pattern_generator.py

def generate_multiple_crime_patterns(suspicious_df, window_size=5, stride=1):
    """
    Generate overlapping 5-event patterns from suspicious data.
    
    Args:
        suspicious_df: DataFrame of suspicious connections
        window_size: Events per pattern (default: 5)
        stride: Slide by this many events (default: 1)
    
    Returns:
        List of pattern arrays, each shape (window_size, num_features)
    """
    patterns = []
    num_events = len(suspicious_df)
    
    if num_events < window_size:
        return []  # Not enough data
    
    events_array = suspicious_df.values
    
    # Generate overlapping windows
    for i in range(0, num_events - window_size + 1, stride):
        pattern = events_array[i : i + window_size]
        patterns.append(pattern)
    
    return patterns
```

### 4.4 Visual Example

```
Suspicious connections (20 events):
E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20

Generated patterns (window=5, stride=1):
┌────────────────────────────────────────────────────────┐
│ Pattern 1:  [E1, E2, E3, E4, E5]             │
│ Pattern 2:  [E2, E3, E4, E5, E6]             │
│ Pattern 3:  [E3, E4, E5, E6, E7]             │
│ Pattern 4:  [E4, E5, E6, E7, E8]             │
│ Pattern 5:  [E5, E6, E7, E8, E9]             │
│ ...                                          │
│ Pattern 16: [E16, E17, E18, E19, E20]            │
└────────────────────────────────────────────────────────┘

Total: 16 patterns from 20 events
Formula: (num_events - window_size + 1) = 20 - 5 + 1 = 16
```

### 4.5 Parameters

| Parameter | Default | Description |
|---|---|---|
| `window_size` | 5 | Events per pattern |
| `stride` | 1 | Slide by 1 for max overlap |
| Minimum | 5 | Need at least 5 suspicious events |

---

## 5. Step 4: Preprocessing

### 5.1 What Happens

Each raw pattern is preprocessed: scaled using the fitted StandardScaler, ensuring consistent feature ranges.

### 5.2 Why This Matters

The scaler was fit on training data to normalize all features to the same scale (mean=0, std=1). Using the same scaler ensures new data is in the same "space" as training data.

### 5.3 Scaling Process

```python
# From app.py - process_incident_log function

def process_incident_log(file_path):
    """
    Load, process, and convert incident log to pattern tensors.
    """
    # Load the file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)
    
    # Step 1: Detect suspicious connections
    suspicious_df = detect_suspicious_connections(df)
    
    if len(suspicious_df) < 5:
        return None, ["Need at least 5 suspicious connections"]
    
    # Step 2: Generate patterns
    raw_patterns = generate_multiple_crime_patterns(suspicious_df, window_size=5, stride=1)
    
    if not raw_patterns:
        return None, ["Could not generate patterns"]
    
    # Step 3: Process each pattern through scaler
    processed_patterns = []
    feature_names = []
    
    for raw_pattern in raw_patterns:
        # raw_pattern shape: (5, num_features) e.g., (5, 31)
        
        # Apply saved scaler (transform, not fit_transform!)
        # This uses the same scaling as training data
        pattern_scaled = scaler.transform(raw_pattern.astype(np.float32))
        
        # Convert to tensor
        pattern_tensor = torch.tensor(pattern_scaled, dtype=torch.float32).unsqueeze(0)
        # Shape: (1, 5, 31) - batch_size=1 for single pattern
        
        processed_patterns.append(pattern_tensor)
        
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(31)]
    
    return processed_patterns, feature_names
```

### 5.4 Visual: Scaling Transformation

```
Raw Pattern (before scaling):
┌─────────────────────────────────────────────────────────────────┐
│ Feature:    dur   sbytes   dbytes   tcprtt   ...        │
│ Value:     0.0   1000     0        0.0            │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼ scaler.transform()
                    │
Scaled Pattern (after scaling):
┌─────────────────────────────────────────────────────────────────┐
│ Feature:    dur   sbytes   dbytes   tcprtt   ...        │
│ Value:    -0.5   1.2     -0.3     -0.8            │
└─────────────────────────────────────────────────────────────────┘

Why scale? Ensures all features contribute equally to similarity
No single feature dominates due to different scales
```

---

## 6. Step 5: Tensor Conversion

### 6.1 What Happens

Each pattern is converted from NumPy array to PyTorch tensor, with proper shape and device placement.

### 6.2 Tensor Shape Requirements

The model expects input in this format:

| Dimension | Size | Description |
|---|---|---|
| Batch | 1 | Single pattern at a time |
| Sequence | 5 | Events per pattern |
| Features | 31 | Network features |

### 6.3 Code

```python
# Convert to tensor with proper shape
pattern_tensor = torch.tensor(pattern_scaled, dtype=torch.float32).unsqueeze(0)
# Before: (5, 31)
# After:  (1, 5, 31) ← (batch=1, seq=5, features=31)
```

---

## 7. Step 6: Best Match Search

### 7.1 What Happens

We compare **every pattern from Incident 1** with **every pattern from Incident 2** to find the best match.

### 7.2 Why All Combinations?

Different parts of each incident might match. We need to find the maximum similarity across all possible pairs.

### 7.3 Algorithm

```python
# From app.py - predict function

# Initialize best match tracking
max_similarity_score = -1.0
best_p1_tensor = None
best_p2_tensor = None
best_p1_idx = 0
best_p2_idx = 0

# Compare all combinations
with torch.no_grad():
    for idx1, p1 in enumerate(pattern1):
        for idx2, p2 in enumerate(pattern2):
            # Move to device
            p1dev = p1.to(DEVICE)
            p2dev = p2.to(DEVICE)
            
            # Compute similarity using model
            score = model.compute_similarity(p1dev, p2dev).item()
            
            # Update best match if better
            if score > max_similarity_score:
                max_similarity_score = score
                best_p1_tensor = p1
                best_p2_tensor = p2
                best_p1_idx = idx1
                best_p2_idx = idx2
```

### 7.4 Visual Example

```
Incident 1: P1, P2, P3, P4, P5 (5 patterns)
Incident 2: Q1, Q2, Q3, Q4 (4 patterns)

All combinations (5 × 4 = 20):
┌──────────────────────────────────────────────────┐
│ P1 vs Q1:  0.92  ← Same MO!                     │
│ P1 vs Q2:  0.45                               │
│ P1 vs Q3:  0.38                               │
│ P1 vs Q4:  0.21                               │
│ P2 vs Q1:  0.55                               │
│ P2 vs Q2:  0.88                               │
│ P2 vs Q3:  0.31                               │
│ ...                                           │
│ P4 vs Q2:  0.95  ← BEST MATCH!                │
│ ...                                           │
└──────────────────────────────────────────────────┘

Best match: P4 vs Q2 = 0.95 (highest similarity)
```

### 7.5 Complexity

| Metric | Value |
|---|---|
| Patterns incident 1 | N |
| Patterns incident 2 | M |
| Comparisons | N × M |
| Time complexity | O(N × M) |

---

## 8. Step 7: Verdict Generation

### 8.1 What Happens

Based on the best similarity score, we generate a human-readable verdict.

### 8.2 Verdict Logic

```python
# Threshold: 0.5 (50%)
# >= 0.5: SAME MO (same attacker)
# < 0.5: DIFFERENT MO (unrelated)

verdict = "SAME MO" if max_similarity_score >= 0.5 else "DIFFERENT MO"
```

### 8.3 Why 0.5?

The threshold is chosen to balance:
- **Precision**: Not too strict (miss real matches)
- **Recall**: Not too lenient (false positives)

0.5 (50%) means "more similarities than differences" — a balanced trade-off.

### 8.4 Output Format

```python
return jsonify({
    "verdict": "SAME MO",
    "similarityScore": 0.95,
    "incident1": "incident_dos_a.csv",
    "incident2": "incident_dos_b.csv",
    "best_patterns": {
        "pattern1_index": 4,
        "pattern2_index": 2,
        "pattern1_data": [0.12, 0.85, -0.33, ...],
        "pattern2_data": [0.10, 0.82, -0.30, ...]
    },
    "explanation": [
        {"feature": "sbytes", "importance": 0.35},
        {"feature": "tcprtt", "importance": -0.22},
        ...
    ]
})
```

---

## 9. Step 8: SHAP Explanation

### 9.1 What Happens

We compute SHAP values to explain which features drove the similarity decision.

### 9.2 Why This Matters

Forensically, we need to explain WHY the model made this decision — not just WHAT it decided.

### 9.3 SHAP Process

```python
# From src/explain.py

import shap

# Create SHAP explainer
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values for the best pattern pair
shap_values = explainer.shap_values([best_p1_tensor, best_p2_tensor])

# Extract feature importance
importance = []
for i, (feat, val) in enumerate(zip(feature_names, shap_values)):
    importance.append({
        "feature": feat,
        "importance": float(val)
    })

# Sort by absolute importance
importance = sorted(importance, key=lambda x: abs(x['importance']), reverse=True)
```

### 9.4 Example Output

```json
{
  "explanation": [
    {"feature": "sbytes", "importance": 0.35},
    {"feature": "tcprtt", "importance": -0.22},
    {"feature": "synack", "importance": 0.15},
    {"feature": "dmeansz", "importance": 0.08},
    {"feature": "rate", "importance": -0.05}
  ]
}
```

### 9.5 Interpreting SHAP

| Feature | Impact | Interpretation |
|---|---|---|
| sbytes (+0.35) | High | Source bytes pattern very similar |
| tcprtt (-0.22) | Negative | TCP timing differs |
| synack (+0.15) | Positive | SYN-ACK pattern similar |

---

## 10. Complete Flow Diagram

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  USER UPLOADS TWO INCIDENT LOG FILES                                               │
│  ┌─────────────────────┐          ┌─────────────────────┐                            │
│  │ incident_log_A.csv │          │ incident_log_B.csv │                            │
│  └─────────┬───────────┘          └─────────┬───────────┘                            │
│            │                               │                                       │
│            ▼                               ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 2: ANOMALY DETECTION                                    │               │
│  │  • Load CSV                                               │               │
│  │  • Statistical thresholding (mean ± 2 std)                │               │
│  │  • Extract suspicious connections                       │               │
│  └────────────────────────────┬────────────────────────────┘               │
│                               │                                               │
│                               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 3: PATTERN GENERATION                                 │               │
│  │  • Window size: 5                                         │               │
│  │  • Stride: 1 (overlapping)                               │               │
│  │  • Multiple patterns per incident                         │               │
│  └────────────────────────────┬────────────────────────────┘               │
│                               │                                               │
│                               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 4: PREPROCESSING                                       │               │
│  │  • Apply StandardScaler (fitted on training data)              │               │
│  │  • Transform (not fit!)                                 │               │
│  │  • Convert to PyTorch tensor                            │               │
│  └────────────────────────────┬────────────────────────────┘               │
│                               │                                               │
│                               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 5: BEST MATCH SEARCH                                 │               │
│  │  • Compare all (pattern_A × pattern_B) pairs             │               │
│  │  • Compute similarity for each                         │               │
│  │  • Return maximum similarity                           │               │
│  └────────────────────────────┬────────────────────────────┘               │
│                               │                                               │
│                               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 6: VERDICT GENERATION                               │               │
│  │  • if similarity >= 0.5: SAME MO                      │               │
│  │  • else: DIFFERENT MO                                  │               │
│  └────────────────────────────┬────────────────────────────┘               │
│                               │                                               │
│                               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 7: SHAP EXPLANATION                                │               │
│  │  • Compute SHAP values                                │               │
│  │  • Rank features by importance                        │               │
│  │  • Return explanation                              │               │
│  └──��─��───────────────────────┬────────────────────────────┘               │
│                               │                                               │
│                               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ RETURN RESULT                                           │               │
│  │  • verdict: "SAME MO" or "DIFFERENT MO"                │               │
│  │  • similarityScore: 0.95                             │               │
│  │  • best_patterns: {...}                              │               │
│  │  • explanation: [...]                               │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

- **[Architecture Deep Dive](architecture.md)**: CNN, Bi-LSTM, Siamese details
- **[API Reference](api.md)**: Flask API endpoints
- **[Frontend Guide](frontend.md)**: React components

---

<div align="center">
  <sub>Part of the Intelligent Crime Pattern Recognition System</sub>
</div>