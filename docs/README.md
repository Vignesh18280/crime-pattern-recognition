# Intelligent Crime Pattern Recognition System
### Hybrid Siamese · CNN · Bidirectional LSTM Architecture for Cybercrime MO Detection

---

## Quick Overview

This project implements an **end-to-end intelligent system for cybercrime pattern recognition** using a hybrid deep learning architecture combining:

- **Convolutional Neural Networks (CNN)** — For spatial feature extraction from network traffic
- **Bidirectional LSTM (Bi-LSTM)** — For temporal sequence modeling of attack chains
- **Siamese Neural Networks (SNN)** — For learning and measuring Modus Operandi (MO) similarity

The system determines whether two cybercrime incidents were carried out by the **same threat actor** — even when surface-level details like IP addresses differ completely.

---

## Quick Start

### Running the Full System

```bash
# Terminal 1: Start Backend (Port 8080)
python3 app.py

# Terminal 2: Start Frontend (Port 5173)
cd frontend && npm run dev
```

Then open **http://localhost:5173** in your browser.

### Training the Model

```bash
python3 src/train.py
```

---

## Key Statistics

| Metric | Value |
|---|---|
| Validation Accuracy | **86.09%** |
| Training Accuracy | **90.04%** |
| Training Time (CPU) | **6.2 minutes** |
| Dataset | **UNSW-NB15 (175,341 records)** |
| Attack Categories | **10** |
| Model Parameters | **716,481** |

---

## Documentation Structure

This documentation is organized into detailed technical guides. Click any topic to dive deeper:

### Core Concepts

| Topic | Description | Estimated Reading Time |
|---|---|---|
| **[Architecture Deep Dive](architecture.md)** | Complete explanation of CNN, Bi-LSTM, and Siamese architecture | 20-25 min |
| **[Dataset Reference](dataset.md)** | UNSW-NB15 dataset, all 31 features, attack categories | 15-20 min |
| **[Processing Pipeline](pipeline.md)** | End-to-end data flow from upload to verdict | 10-15 min |

### System Components

| Topic | Description | Estimated Reading Time |
|---|---|---|
| **[API Reference](api.md)** | Flask API endpoints, request/response formats | 5-10 min |
| **[Frontend Guide](frontend.md)** | React components, visualizations, UI | 5-10 min |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Crime Pattern A (5 events)                         │
│  Event 1: [31 network features] ──┐                                 │
│  Event 2: [31 network features] ──┤                                 │
│  Event 3: [31 network features] ──┼──▶ CNN + Bi-LSTM ─▶ MO Vector  │
│  Event 4: [31 network features] ──┤            (128-dim)          │
│  Event 5: [31 network features] ──┘                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                           ┌────────┴────────┐
                           │ Twin Network    │
                           │ (Shared Weights)│
                           └────────┬────────┘
                                    │
┌──────────────────────────────────────────────────────────��──────────┐
│                    Crime Pattern B (5 events)                         │
│  Event 1: [31 network features] ──┐                                 │
│  Event 2: [31 network features] ──┤                                 │
│  Event 3: [31 network features] ──┼──▶ CNN + Bi-LSTM ─▶ MO Vector  │
│  Event 4: [31 network features] ──┤                                 │
│  Event 5: [31 network features] ──┘                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           Similarity Score
                              (0.0-1.0)
                              
                    ≥ 0.5 → SAME MO (same attacker)
                    < 0.5 → DIFFERENT MO (unrelated)
```

---

## What is Modus Operandi (MO)?

**Modus Operandi** (Latin for "method of operating") is a Latin term used in forensics to describe the distinctive methodology a criminal uses to commit offenses.

In cybercrime, MO includes:
- **Attack vectors** (phishing, exploit, brute force)
- **Patterns in network traffic** (timing, packet sizes, protocols)
- **Tool signatures** (特定 tools, malware behavior)
- **Post-attack behaviors** (data exfiltration patterns)

Two attacks that appear different on the surface (different target IPs, different times) might share the same underlying MO — indicating the same threat actor.

---

## Why Deep Learning for MO Detection?

Traditional forensic methods:
- Rely on manual expert analysis
- Cannot scale to large datasets
- Miss subtle patterns

Deep learning offers:
- **Automatic feature learning** — No manual feature engineering
- **Pattern detection** — Learns complex non-linear relationships
- **Scalability** — Works with millions of records
- **One-shot learning** — Can detect new attacks from few examples

---

## File Structure

```
crime-pattern-recognition/
│
├── docs/                        # Detailed documentation
│   ├── README.md              # This file
│   ├── architecture.md       # CNN + Bi-LSTM + Siamese deep dive
│   ├── dataset.md          # UNSW-NB15 dataset reference
│   ├── pipeline.md        # End-to-end processing pipeline
│   ├── api.md            # Flask API reference
│   └── frontend.md       # React frontend guide
│
├── src/                      # Core source code
│   ├── model.py            # CNN + Bi-LSTM + Siamese architecture
│   ├── dataset.py         # UNSW-NB15 loading + pattern generation
│   ├── train.py           # Training loop
│   ├── detect.py         # Anomaly detection
│   ├── pattern_generator.py  # 5-event pattern extraction
│   └── explain.py        # SHAP XAI
│
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── lib/        # API client
│   │   └── App.tsx     # Main app
│   └── package.json
│
├── data/                    # Dataset files
│   ├── sample_incidents/  # Test CSV files
│   └── UNSW_NB15_*.parquet
│
├── results/                # Model checkpoints + visualizations
├── app.py               # Flask API server
└── README.md           # Main project README
```

---

## Sample Usage Flow

### 1. Upload Two Incident Logs

```bash
# Using the frontend at http://localhost:5173
# Upload two CSV files representing cyber incidents
```

### 2. System Processes Each Incident

```
Raw CSV
    │
    ▼
Anomaly Detection (find suspicious connections)
    │
    ▼
Pattern Generation (extract 5-event windows)
    │
    ▼
Preprocessing (scale with StandardScaler)
    │
    ▼
Tensor conversion
```

### 3. Compare and Get Result

```
Pattern A vs Pattern B ──▶ Siamese Network ──▶ Similarity Score
                                                   │
                                          ┌────────┴────────┐
                                          ▼                 ▼
                                    SAME MO      DIFFERENT MO
                              (≥0.5 score)    (<0.5 score)
```

### 4. View Explanation

SHAP values show exactly which features drove the similarity decision — legally defensible AI for court.

---

## Common Workflows

### Training New Model

```bash
python3 src/train.py
```

Results saved to `results/best_model.pth`

### Running Predictions

```bash
# Start backend
python3 app.py

# Start frontend
cd frontend && npm run dev
```

Open http://localhost:5173

### Understanding Predictions

See **[SHAP Explainability](architecture.md#shap-explainable-ai)** in architecture.md

---

## Next Steps: Deep Dive

Choose a topic based on your interest:

1. **New to the project?** Start with **[Architecture Deep Dive](architecture.md)**
2. **Want to understand the data?** Read **[Dataset Reference](dataset.md)**
3. **Need to customize the pipeline?** See **[Processing Pipeline](pipeline.md)**
4. **Building on the API?** Check **[API Reference](api.md)**
5. **Modifying the UI?** See **[Frontend Guide](frontend.md)**

---

## License & Acknowledgments

- Dataset: **UNSW-NB15** by Cyber Range Lab, University of New South Wales
- Framework: **PyTorch 2.0**
- Architecture: Hybrid CNN + Bi-LSTM + Siamese Network

This is a **B.Tech major project** proof-of-concept. Not intended for production forensic use without further validation.

---

<div align="center">
  <sub>Real UNSW-NB15 Data · 86.09% Validation Accuracy</sub>
</div>