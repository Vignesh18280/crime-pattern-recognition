# Intelligent Crime Pattern Recognition
### Hybrid Siamese · CNN · Bidirectional LSTM Architecture for Cybercrime MO Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-86.09%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-purple)
![XAI](https://img.shields.io/badge/XAI-SHAP-yellow)

---

> 📖 **Detailed Documentation**: For comprehensive technical documentation, see the [`docs/`](docs/README.md) folder.
> - [Architecture Deep Dive](docs/architecture.md) - Full CNN + Bi-LSTM + Siamese explanation
> - [Dataset Reference](docs/dataset.md) - UNSW-NB15 features and attack categories
> - [Processing Pipeline](docs/pipeline.md) - End-to-end data flow
> - [API Reference](docs/api.md) - Flask API endpoints
> - [Frontend Guide](docs/frontend.md) - React components

---

## Quick Start

```bash
# Terminal 1: Backend (Port 8080)
python3 app.py

# Terminal 2: Frontend (Port 5173)
cd frontend && npm run dev
```

Then open **http://localhost:5173**

---

## Overview

This project implements an end-to-end intelligent system for **cybercrime pattern recognition** using a hybrid deep learning architecture that combines **Convolutional Neural Networks (CNN)**, **Bidirectional Long Short-Term Memory (Bi-LSTM)** networks, and **Siamese Neural Networks (SNN)**.

The system is capable of determining whether two cybercrime incidents share the same **Modus Operandi (MO)** — identifying if they were carried out by the same threat actor — even when surface-level details like IP addresses, file hashes, or email content differ completely.

---

## Architecture

```
Crime Pattern A                        Crime Pattern B
[5 network events]                     [5 network events]
       │                                       │
       ▼                                       ▼
┌─────────────────────────────────────────────────────┐
│              TWIN SUBNETWORK (Shared Weights)        │
│                                                     │
│  Event 1 ──► CNN ──► fingerprint vector             │
│  Event 2 ──► CNN ──► fingerprint vector             │
│  Event 3 ──► CNN ──► fingerprint vector             │
│  Event 4 ──► CNN ──► fingerprint vector             │
│  Event 5 ──► CNN ──► fingerprint vector             │
│                    │                                │
│                    ▼                                │
│          Bi-LSTM (forward + backward)               │
│                    │                                │
│                    ▼                                │
│              MO Vector (128-dim)                    │
└─────────────────────────────────────────────────────┘
       │                                       │
    MO_Vec_A                               MO_Vec_B
       │                                       │
       └──────── |absolute difference| ────────┘
                          │
                          ▼
                 Similarity Score
                    (0.0 → 1.0)
              
         ≥ 0.5  →  SAME MO (same attacker)
         < 0.5  →  DIFFERENT MO (unrelated)
```

### Component Roles

| Component | Role | Why |
|---|---|---|
| **CNN** | Extracts spatial features from each network event | Learns packet patterns, byte distributions, protocol signatures |
| **Bi-LSTM** | Models the temporal attack sequence | Understands attack chain: recon → exploit → C2 comms |
| **Siamese Network** | Compares two crime patterns | Shared weights ensure both crimes measured by same standard |
| **SHAP** | Explains model decisions | Makes AI evidence legally defensible |

---

## Technical Deep Dive

This section explains every component of the system in depth. By reading this, you should understand how the entire pipeline works without looking at a single line of code.

---

### 1. The Problem: Cybercrime Pattern Recognition

In digital forensics, investigators often encounter multiple cyberattacks that may (or may not) be related. Traditional investigation relies on **Modus Operandi (MO)** — the distinct methodology a threat actor uses. Two seemingly different attacks (different IP addresses, different targets) could be carried out by the same attacker using the same technique.

The challenge: **How do we determine if two cyber incidents share the same MO?**

This is a **similarity problem**, not a classification problem. We're not trying to label an incident as "DoS" or "Backdoor" — we're trying to answer: *"Did the same person carry out both attacks?"*

---

### 2. Why Deep Learning?

Traditional machine learning requires:
- Large labeled datasets for each attack type
- Manual feature engineering (choosing which features matter)
- Cannot generalize to new, unknown attacks

Deep learning offers:
- Automatic feature learning from raw data
- One-shot learning capability (learn from few examples)
- Captures non-linear relationships in data

Specifically, we use three deep learning components working together:
1. **CNN** (Convolutional Neural Network) - Spatial feature extraction
2. **Bi-LSTM** (Bidirectional Long Short-Term Memory) - Temporal sequence modeling
3. **Siamese Network** - Similarity learning

---

### 3. CNN: Spatial Feature Extraction

#### What is a CNN?

A **Convolutional Neural Network (CNN)** is a neural network that uses convolutional layers to automatically learn spatial patterns from data. While commonly used for images (2D grids of pixels), CNNs can also work on 1D data — like network traffic features.

#### How CNNs Work in This Project

Each network event in our data is a vector of 31 features:
```
[duration, src_bytes, dst_bytes, src_pkts, dst_pkts, ..., tcp_rtt, syn_ack]
```

Think of this as a "1D image" with 31 "pixels". The CNN applies **filters** (also called kernels) that slide across these features:

```
Input (31 features):  [f1, f2, f3, ..., f31]
                         ↓ CNN Convolution
Filter (size 3):      [w1, w2, w3]
                         ↓
Output:               [g1, g2, g3, ..., g29]  (smaller due to convolution)
```

The CNN learns different filters to detect different patterns:
- Filter 1: Detects high bandwidth transfers (src_bytes + dst_bytes)
- Filter 2: Detects TCP handshake patterns (syn_ack + ack)
- Filter 3: Detects packet rate anomalies (src_pkts / duration)
- And so on...

After convolution, we apply **pooling** (typically max-pooling) to reduce dimensions while keeping the most important features:

```
Input:   [1, 5, 3, 8, 2]  →  Max Pool (2):  [5, 8]
```

#### Why CNN for Network Data?

Network traffic has inherent spatial structure:
- Certain features naturally group together (e.g., all "bytes" features)
- Attack signatures often appear as specific combinations (e.g., high src_bytes + low dst_bytes = data exfiltration)
- CNNs automatically learn these local patterns without manual feature engineering

#### Code-Level Understanding

```python
# Simplified CNN layer
Conv1d(in_channels=31, out_channels=64, kernel_size=3)
# Takes 31 input features → Produces 64 output features
# Each output feature is a learned combination of 3 input features
```

---

### 4. Bi-LSTM: Temporal Sequence Modeling

#### What is an LSTM?

**Long Short-Term Memory (LSTM)** is a type of recurrent neural network (RNN) designed to handle sequences and remember long-term dependencies. Standard RNNs suffer from the "vanishing gradient problem" — they can't learn relationships between events far apart in time. LSTMs solve this with a special "memory cell" architecture:

```
         ┌─────────────────────────────────────┐
         │          LSTM Cell                 │
         │  ┌───────┐    ┌───────┐         │
Input ──▶│  │ Forget│    │ Input │         │
         │  │ Gate  │    │ Gate  │         │
         │  └───────┘    └───────┘         │
         │       │           │              │
         │       ▼           ▼              │
         │  ┌───────────────────────┐      │
         │  │     Memory Cell       │      │
         │  │   (long-term memory)  │      │
         │  └───────────────────────┘      │
         │                │                │
         │                ▼                │
         │  ┌────────────────────────┐     │
         │  │       Output Gate       │     │
         │  └────────────────────────┘     │
         │                │                │
         └───────────────▶│──▶ Output
```

The three gates:
1. **Forget Gate**: What old information to discard
2. **Input Gate**: What new information to store
3. **Output Gate**: What information to output

#### What is Bidirectional LSTM?

A standard LSTM reads sequences only in one direction (forward). A **Bidirectional LSTM (Bi-LSTM)** processes the same sequence twice:
- Once forward (Event 1 → Event 2 → Event 3...)
- Once backward (Event 5 → Event 4 → Event 3...)

The outputs are then concatenated:

```
Forward Pass:  [h1_forward, h2_forward, h3_forward, h4_forward, h5_forward]
Backward Pass: [h5_backward, h4_backward, h3_backward, h2_backward, h1_backward]
                    │
                    ▼
Combined:     [h1_fwd, h5_bwd, h2_fwd, h4_bwd, ...]  (each event knows both past and future context)
```

#### Why Bi-LSTM for Attack Sequences?

Cyberattacks are **multi-stage sequences**:

1. **Reconnaissance** — Port scanning, vulnerability discovery
2. **Initial Access** — Exploiting a vulnerability
3. **Privilege Escalation** — Gaining higher privileges
4. **Lateral Movement** — Spreading to other systems
5. **Data Exfiltration** — Stealing data / Causing damage

Understanding an attack requires knowing both:
- What happened **before** (context of the attack)
- What happens **after** (the goal of the attack)

Example: A single TCP connection might look innocent in isolation. But if we know it came **after** a port scan and **before** data transfer, it becomes clear this is part of an attack chain.

Bi-LSTM gives every event access to the full attack context.

#### Code-Level Understanding

```python
# Simplified LSTM layer
LSTM(input_size=64, hidden_size=128, bidirectional=True)
# Takes 64 features per event → Produces 128*2 = 256 features (bidirectional)
# Processes sequence of 5 events → Outputs 5 * 256 features
```

---

### 5. Siamese Network: Learning Similarity

#### What is a Siamese Network?

A **Siamese Network** is a neural network architecture that compares two inputs and outputs a **similarity score**. The key idea: twins with shared weights.

```
              ┌─────────────────┐
   Input A ──▶│  Twin Network 1  │──▶ Embedding A
              └─────────────────┘
                    │
                (Shared Weights)
                    │
              ┌─────────────────┐
   Input B ──▶│  Twin Network 2  │──▶ Embedding B
              └─────────────────┘
                    │
         ┌──────────┴──────────┐
         │  Similarity Measure  │
         │   (Absolute Diff)     │
         └──────────┬──────────┘
                    │
                    ▼
            Similarity Score
               (0.0 to 1.0)
```

**Key property**: Both twin networks have **identical, shared weights**. This ensures both inputs are measured by the exact same "ruler."

#### Why Siamese for MO Detection?

1. **Same Standard**: Both crime patterns are encoded by the same network, so differences come from the data, not the encoder
2. **One-Shot Learning**: Can detect new attack types from a single example
3. **Similarity, Not Classification**: Answers "how alike?" not "what is this?"

#### How Similarity is Computed

1. Both inputs pass through identical twin networks → Get two embedding vectors (e.g., 128 dimensions each)
2. Compute absolute difference: `|embedding_A - embedding_B|`
3. Pass through a final dense layer with sigmoid activation
4. Output is a single value between 0.0 and 1.0

```python
# Similarity computation (simplified)
embedding_a = twin_network(input_a)   # Shape: (128,)
embedding_b = twin_network(input_b)  # Shape: (128,)

difference = torch.abs(embedding_a - embedding_b)  # Shape: (128,)

similarity = sigmoid(  # Flattens to single value
    torch.matmul(difference, weights) + bias
)
# Output: 0.7368 (73.68% similar)
```

---

### 6. The Complete Architecture

Putting it all together:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Crime Pattern A (5 events)                          │
│  Event 1: [31 features] ──┐                                           │
│  Event 2: [31 features] ──┤                                           │
│  Event 3: [31 features] ──┼──▶ CNN + Bi-LSTM──▶ 128-dim MO Vector   │
│  Event 4: [31 features] ──┤                                           │
│  Event 5: [31 features] ──┘                                           │
└──────────────────────────────────────────────────────────────���─���───────────┘
                                    │
                                    │  (Twin Network: Shared Weights)
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     Crime Pattern B (5 events)                          │
│  Event 1: [31 features] ──┐                                           │
│  Event 2: [31 features] ──┤                                           │
│  Event 3: [31 features] ──┼──▶ CNN + Bi-LSTM──▶ 128-dim MO Vector   │
│  Event 4: [31 features] ──┤                                           │
│  Event 5: [31 features] ──┘                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ |Embed A - Embed B│
                           │    (Absolute)    │
                           └────────┬────────┘
                                    │
                                    ▼
                            Similarity Score
                               (0.0-1.0)
```

**Step-by-step data flow**:

1. **Input**: 5 consecutive network events (5 × 31 = 155 features)
2. **CNN**: Each event → 64 spatial features (5 × 64 = 320 features)
3. **Bi-LSTM**: Sequence → 128-dim embedding (single vector)
4. **Siamese**: Two embeddings → Similarity score

---

### 7. Data Processing Pipeline

#### Step 1: Raw Network Data

Raw data from UNSW-NB15 looks like:

```
dur, sbytes, dbytes, spkts, dpkts, tcprtt, synack, ackdat, ...
0.0,  264,    0,     4,    4,   0.0,   0.0,    0.0,   ...
0.0,  1500,   0,     3,    3,   0.0,   0.0,    0.0,   ...
0.0,  2000,   500,   5,    4,   0.1,   0.05,   0.02,  ...
```

#### Step 2: Anomaly Detection

We first detect suspicious connections from normal traffic. This uses statistical thresholding:

```python
# Simplified anomaly detection
for connection in connections:
    if connection.duration > HIGH_THRESHOLD:
        mark_suspicious(connection)
    if connection.bytes > HIGH_THRESHOLD:
        mark_suspicious(connection)
    # ... other rules
```

The result: A subset of connections flagged as potentially part of an attack.

#### Step 3: Crime Pattern Generation

From suspicious connections, we generate **5-event patterns**:

```
Window Size = 5 (5 consecutive events)
Stride = 1 (slide by 1 event each time)

Events: [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]

Pattern 1: [e1, e2, e3, e4, e5]  ← Events 1-5
Pattern 2: [e2, e3, e4, e5, e6]  ← Events 2-6
Pattern 3: [e3, e4, e5, e6, e7]  ← Events 3-7
Pattern 4: [e4, e5, e6, e7, e8]  ← Events 4-8
Pattern 5: [e5, e6, e7, e8, e9]  ← Events 5-9
Pattern 6: [e6, e7, e8, e9, e10] ← Events 6-10
```

Why stride=1? This generates overlapping patterns to capture different parts of the attack sequence.

#### Step 4: Preprocessing

1. **Padding**: Ensure all patterns have exactly 5 events (if shorter, pad with zeros)
2. **Scaling**: Normalize features using StandardScaler (z-score normalization):

```python
# Z-score normalization
z = (x - mean) / std

# Example: src_bytes = 1000
# mean = 500, std = 250
# z = (1000 - 500) / 250 = 2.0 (2 standard deviations above mean)
```

3. **Tensor conversion**: Convert to PyTorch tensors with shape `(1, 5, 31)`

---

### 8. Training Process

#### Why Training Matters

The Siamese network learns to produce **similar embeddings for same-MO pairs** and **different embeddings for different-MO pairs**.

#### Training Data: Positive and Negative Pairs

We create pairs from the training data:

| Pair Type | How Created | Label |
|---|---|---|
| **Same MO** | Two patterns from the same attack category (e.g., two DoS attacks) | 1.0 (positive) |
| **Different MO** | Two patterns from different categories (e.g., DoS vs Backdoor) | 0.0 (negative) |

For example:
- Pattern A (DoS) + Pattern B (DoS) → Same MO → Label = 1.0
- Pattern A (DoS) + Pattern C (Backdoor) → Different MO → Label = 0.0

#### Loss Function: Contrastive Loss

During training, we use **Contrastive Loss** (also called Siamese Loss):

```
Loss = (1 - label) * max(0, margin - similarity)^2 + label * (similarity)^2

Where:
- label = 1.0 for Same MO, 0.0 for Different MO
- margin = 0.5 (threshold for negative pairs)
- similarity = computed similarity score (0.0 to 1.0)
```

What this does:
- **Same MO pairs** (label=1): Minimize similarity → 0
- **Different MO pairs** (label=0): Maximize similarity → < margin (default 0.5)

```python
# Simplified training loop
for epoch in range(30):
    for pair_a, pair_b, label in training_pairs:
        # Forward pass
        embedding_a = twin_network(pair_a)
        embedding_b = twin_network(pair_b)
        similarity = compute_similarity(embedding_a, embedding_b)
        
        # Compute loss
        loss = contrastive_loss(similarity, label)
        
        # Backward pass (update weights)
        loss.backward()
        optimizer.step()
```

#### Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 30 |
| Margin (for contrastive loss) | 0.5 |
| Optimizer | Adam |

---

### 9. Inference: How Predictions Work

After training, making predictions is straightforward:

#### Step 1: Load New Incidents

Upload two new incident log files (CSV format).

#### Step 2: Process Each Incident

```python
# Process each incident log
incident1_patterns = generate_multiple_crime_patterns(incident1_log)
# Output: List of 5-event pattern tensors
# Example: [pattern_1, pattern_2, pattern_3, ..., pattern_N]

incident2_patterns = generate_multiple_crime_patterns(incident2_log)
# Output: List of 5-event pattern tensors
```

#### Step 3: Compare All Combinations

We compare **every pattern from Incident 1** with **every pattern from Incident 2**:

```python
max_similarity = -1.0
best_match = (None, None)

for p1 in incident1_patterns:
    for p2 in incident2_patterns:
        similarity = model.compute_similarity(p1, p2)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = (p1, p2)
```

Why all combinations? Because:
- Different parts of Incident 1 might match different parts of Incident 2
- We want the best possible match

#### Step 4: Generate Verdict

Based on the maximum similarity:

```python
if max_similarity >= 0.5:
    verdict = "SAME MO"  # Same attacker
else:
    verdict = "DIFFERENT MO"  # Unrelated
```

Threshold 0.5 is chosen because:
- Below 50% similarity means more differences than similarities
- This is a balanced trade-off (not too strict, not too lenient)

---

### 10. Pattern Generation: Deep Dive

The pattern generator is crucial for the system. Here's exactly how it works:

```python
def generate_multiple_crime_patterns(suspicious_df, window_size=5, stride=1):
    """
    Generate multiple 5-event patterns from suspicious connections.
    
    Args:
        suspicious_df: DataFrame of suspicious connections
        window_size: Number of events per pattern (default: 5)
        stride: Slide by this many events (default: 1)
    
    Returns:
        List of pattern tensors, each shape (1, 5, 31)
    """
    patterns = []
    
    # Flatten to event sequences
    events = suspicious_df.values
    
    # Generate overlapping windows
    for i in range(0, len(events) - window_size + 1, stride):
        window = events[i:i + window_size]
        
        # Skip if not enough events
        if len(window) < window_size:
            continue
        
        # Pad if needed (ensure 31 features)
        if window.shape[1] < 31:
            window = pad_to_31(window)
        
        # Scale the pattern
        window_scaled = scaler.transform(window)
        
        # Convert to tensor
        pattern_tensor = torch.tensor(window_scaled).unsqueeze(0)
        patterns.append(pattern_tensor)
    
    return patterns
```

**Example output**:
- Input: 20 suspicious events
- Window size: 5, Stride: 1
- Output: 16 patterns (20 - 5 + 1 = 16)

---

### 11. Why CNN + Bi-LSTM + Siamese?

Each component serves a specific purpose:

| Component | What It Captures | Example |
|---|---|---|
| **CNN** | Local spatial patterns | TCP handshake irregularities, byte distribution spikes |
| **Bi-LSTM** | Sequential attack behavior | Recon → Exploit → C2 → Exfil chain |
| **Siamese** | Comparative similarity | Same attacker methodology |

**Why combine them?**

- CNN alone: Can't capture sequence (event order matters in attacks)
- Bi-LSTM alone: Can't compare two incidents
- Siamese alone: Needs embeddings to compare (CNN+Bi-LSTM provides embeddings)

---

### 12. SHAP: Explainable AI

A critical requirement: **AI in forensics must be explainable.**

This system uses **SHAP (SHapley Additive exPlanations)**:

```python
import shap

# Create explainer
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values for a pair
shap_values = explainer.shap_values([pattern_a, pattern_b])

# Output shows exactly which features drove the decision
# Example output:
#  src_bytes:    +0.35  (increases similarity)
#  tcp_rtt:     -0.22  (decreases similarity)  
#  syn_ack:     +0.15  (increases similarity)
```

Why SHAP for forensics:
- **Legally defensible**: Can explain in court why the model made this decision
- **Feature-level**: Tells investigators exactly which network features mattered
- **Directional**: Shows whether each feature increased or decreased similarity

---

### 13. Complete End-to-End Data Flow

Here's the entire pipeline from raw data to prediction:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Upload Incident Logs                                           │
│                                                                         │
│   incident_log_1.csv  ───┐                                             │
│                            │                                             │
│   incident_log_2.csv  ───┘                                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Anomaly Detection                                              │
│                                                                         │
│   Scan for suspicious connections:                                      │
│   - High duration, High bytes, Anomalous packet counts, etc.           │
│                                                                         │
│   Output: suspicious_connections_list                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Pattern Generation                                            │
│                                                                         │
│   Extract 5-event windows with stride=1:                                 │
│   Pattern 1: [e1,e2,e3,e4,e5]                                        │
│   Pattern 2: [e2,e3,e4,e5,e6]                                         │
│   ...                                                                   │
│                                                                         │
│   Output: List[pattern_tensor]                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Preprocessing                                                  │
│                                                                         │
│   1. Scale with StandardScaler (z-score normalization)               │
��   2. Convert to PyTorch tensor (shape: 1, 5, 31)                       │
│   3. Move to device (CPU or GPU)                                       │
│                                                                         │
│   Output: List[torch.Tensor]                                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Best Match Search                                              │
│                                                                         │
│   For each pattern in incident1:                                       │
│       For each pattern in incident2:                                    │
│           Compute similarity score                                      │
│           Keep maximum                                                  │
│                                                                         │
│   Output: max_similarity_score, best_pattern_pair                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Generate Verdict                                               │
│                                                                         │
│   if max_similarity >= 0.5:                                           │
│       verdict = "SAME MO"                                              │
│   else:                                                                 │
│       verdict = "DIFFERENT MO"                                          │
│                                                                         │
│   Output: {verdict, similarity_score, ...}                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Explainability (SHAP)                                           │
│                                                                         │
│   Compute SHAP values to explain:                                       │
│   - Which features drove the similarity?                               │
│   - Did they increase or decrease similarity?                         │
│                                                                         │
│   Output: List[{feature: str, importance: float}]                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 14. Key Parameters Summary

| Parameter | Value | Purpose |
|---|---|---|
| `window_size` | 5 | Events per pattern |
| `stride` | 1 | Pattern overlap |
| `feature_dim` | 31 | UNSW-NB15 features |
| `embedding_dim` | 128 | MO vector size |
| `similarity_threshold` | 0.5 | SAME MO vs DIFFERENT MO |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer |
| `epochs` | 30 | Training iterations |

---

## Results

| Metric | Value |
|---|---|
| Best Validation Accuracy | **86.09%** |
| Final Training Accuracy | **90.04%** |
| Training Time (CPU) | **6.2 minutes** |
| Dataset | **UNSW-NB15 (175,341 real records)** |
| Attack Classes | **10 categories** |
| Total Parameters | **716,481** |

### Sample Predictions

```
SAME MO PAIR (DoS → DoS):
  Similarity Score : 0.7368
  Verdict          : SAME MO — Likely same threat actor
  Confidence       : 73.68%

DIFFERENT MO PAIR (DoS → Backdoor):
  Similarity Score : 0.0000
  Verdict          : DIFFERENT MO — Unrelated crimes
  Confidence       : 100.00%
```

---

## Dataset

**UNSW-NB15** — Created by the Cyber Range Lab, University of New South Wales

| Property | Value |
|---|---|
| Total Records | 175,341 (training set) |
| Features | 31 numeric network features |
| Attack Categories | Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms |
| Source | Real network traffic captured in a controlled lab environment |
| Format | Parquet |

The dataset is processed into **crime pattern sequences** — groups of 5 consecutive network events of the same attack type — which simulate a multi-stage attack chain. These patterns are then paired to create **positive pairs** (same MO) and **negative pairs** (different MO) for Siamese training.

---

## Project Structure

```
crime-pattern-recognition/
│
├── src/
│   ├── model.py         # CNN + Bi-LSTM + Siamese architecture
│   ├── dataset.py       # UNSW-NB15 loading + crime pattern pairing
│   ├── train.py         # Training loop + visualization
│   └── explain.py       # SHAP XAI explanations
│
├── results/
│   ├── training_curves.png          # Accuracy + loss over 30 epochs
│   ├── similarity_distribution.png  # Same MO vs Different MO scores
│   ├── feature_importance.png       # Top 20 SHAP features
│   └── shap_summary.png             # Feature impact direction
│
├── data/                            # Dataset files (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/crime-pattern-recognition
cd crime-pattern-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# Requires Kaggle API setup
kaggle datasets download -d dhoogla/unswnb15 -p data/
cd data && unzip unswnb15.zip
```

Or download manually from:
[https://www.kaggle.com/datasets/dhoogla/unswnb15](https://www.kaggle.com/datasets/dhoogla/unswnb15)

---

## Running the Project

All commands should be run from the **project root directory**.

### Step 1 — Train the model
```bash
python3 src/train.py
```
Expected output:
```
Epoch 01/30  →  Train Acc: 59.24%  Val Acc: 72.37%
Epoch 10/30  →  Train Acc: 87.97%  Val Acc: 80.12%
Epoch 30/30  →  Train Acc: 90.04%  Val Acc: 84.93%
Best val accuracy: 86.09%
```

### Step 2 — Run XAI explanations
```bash
python3 src/explain.py
```
Expected output:
```
Similarity Score : 0.7368
Verdict          : SAME MO — Likely same threat actor
Confidence       : 73.68%
```

### Step 3 — View results
```
results/training_curves.png
results/similarity_distribution.png
results/feature_importance.png
results/shap_summary.png
```

---

## Full System: API + Frontend

The project includes a **Flask API** and **React Frontend** for end-to-end crime pattern analysis.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   React UI       │────▶│   Flask API      │
│   (Port 5173)   │     │   (Port 8080)   │
└─────────────────┘     └─────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │  Siamese Model  │
                      │  + Pattern Gen │
                      └─────────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Upload incident log (CSV/Parquet) |
| `/api/predict` | POST | Compare two incidents, return similarity + patterns |
| `/api/ready` | GET | Check if model is loaded |

### How Pattern Matching Works

1. **Upload** → Raw incident log (CSV) is uploaded to the server
2. **Anomaly Detection** → Suspicious connections are detected from network logs
3. **Pattern Generation** → Multiple 5-event patterns are extracted with stride=1
4. **Comparison** → All pattern pairs are evaluated by the Siamese network
5. **Best Match** → The pair with highest similarity score is returned
6. **Visualization** → Frontend displays verdict, scores, trend charts, and data table

```python
# Best-match comparison logic (from app.py)
max_similarity_score = -1.0

with torch.no_grad():
    for idx1, p1 in enumerate(pattern1):
        for idx2, p2 in enumerate(pattern2):
            score = model.compute_similarity(p1, p2).item()
            if score > max_similarity_score:
                max_similarity_score = score
                best_p1_idx = idx1
                best_p2_idx = idx2

# Verdict: ≥0.5 = SAME MO, <0.5 = DIFFERENT MO
verdict = "SAME MO" if max_similarity_score >= 0.5 else "DIFFERENT MO"
```

### Running the Full System

#### 1. Start Backend (Port 8080)
```bash
python3 app.py
```

#### 2. Start Frontend (Port 5173)
```bash
cd frontend
npm install
npm run dev
```

#### 3. Usage
1. Open `http://localhost:5173` in browser
2. Upload two incident log CSV files
3. Click "Analyze Incidents"
4. View results: verdict, similarity score, trend charts, and feature comparison

### Frontend Features

| Component | Description |
|---|---|
| **Verdict Card** | Shows SAME MO / DIFFERENT MO with similarity score |
| **Explanation Chart** | Bar chart of feature importance |
| **Stats Grid** | Similarity %, Verdict, Feature count, Best pattern index |
| **Summary Text** | Plain English analysis summary |
| **Trend Chart** | Line chart comparing features across incidents |
| **Data Table** | Feature-by-feature comparison with differences |

### Sample Incident Files

Sample CSV files for testing are in `data/sample_incidents/`:
- `incident_dos_a.csv` — DoS attack (A)
- `incident_dos_b.csv` — DoS attack (B) — same MO as A
- `incident_fuzzers_a.csv` — Fuzzers attack — different MO

---

## Explainability (XAI)

A critical requirement for AI in digital forensics is **explainability** — the ability to justify why the model reached a conclusion. Without this, AI evidence cannot be presented in a court of law.

This system uses **SHAP (SHapley Additive exPlanations)** to:
- Identify which network features drove the similarity decision
- Show the direction of each feature's impact (increases/decreases similarity)
- Provide per-pair explanations that investigators can validate

```
Top features driving MO similarity (sample output):
  src_load        →  highest impact  (network load pattern)
  dst_load        →  second impact   (destination load)
  tcp_rtt         →  third impact    (round-trip time signature)
  syn_ack         →  fourth impact   (TCP handshake pattern)
  trans_depth     →  fifth impact    (transaction depth)
```

---

## Key Concepts for Understanding

**Why Siamese Network?**
Traditional classifiers assign a label to a single input. A Siamese Network learns a *similarity function* — it answers "how alike are these two things?" This enables **one-shot learning**: detecting a new, previously unseen attack type from a single example without retraining.

**Why Bi-LSTM over regular LSTM?**
A standard LSTM only reads sequences forward — it knows the past but not the future. A Bi-LSTM reads both forward and backward, giving every event full context. In attack sequences, a later event can retroactively change the meaning of an earlier one.

**Why CNN for network features?**
CNNs extract spatial hierarchies of patterns. Applied to network features (treated as a 1D signal), CNNs learn increasingly abstract representations — from raw bytes to protocol behaviors to attack signatures.

---

## Challenges & Limitations

| Challenge | Description | Mitigation |
|---|---|---|
| Black Box Problem | Deep learning models are opaque | SHAP XAI provides feature-level explanations |
| Adversarial Attacks | Attackers can manipulate traffic to evade detection | Dropout regularization improves robustness |
| Dataset Imbalance | Some attack classes have far fewer samples | Balanced positive/negative pair generation |
| Overfitting | Train acc (90%) > Val acc (86%) | Dropout, weight decay, early checkpointing |

---

## Technologies Used

### Backend
| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| PyTorch 2.0 | Deep learning framework |
| Flask | REST API |
| SHAP | Explainable AI |
| Pandas / NumPy | Data processing |
| Scikit-learn | Preprocessing, metrics |

### Frontend
| Technology | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool |
| Recharts | Charts (bar, line) |
| React Dropzone | File uploads |
| Lucide React | Icons |
| CSS Modules | Styling |

---

<div align="center">
  <sub> Real UNSW-NB15 Data · 86.09% Validation Accuracy</sub>
</div>
