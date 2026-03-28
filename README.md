# Intelligent Crime Pattern Recognition
### Hybrid Siamese · CNN · Bidirectional LSTM Architecture for Cybercrime MO Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-86.09%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-purple)
![XAI](https://img.shields.io/badge/XAI-SHAP-yellow)

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

| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| PyTorch 2.0 | Deep learning framework |
| SHAP | Explainable AI |
| Pandas / NumPy | Data processing |
| Scikit-learn | Preprocessing, metrics |
| Matplotlib / Seaborn | Visualization |
| Kaggle API | Dataset download |

---

## Academic Context

This implementation is a **B.Tech major project** prototype demonstrating the feasibility of hybrid deep learning architectures for cybercrime pattern recognition. It is based on published research in the field of AI-driven digital forensics and implements the core concepts described in the referenced paper.

The system is a **proof-of-concept** and is not intended for production forensic use without further validation, legal review, and adversarial robustness testing.

---

<div align="center">
  <sub> Real UNSW-NB15 Data · 86.09% Validation Accuracy</sub>
</div>