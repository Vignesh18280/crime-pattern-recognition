"""
dataset.py
----------
Handles loading of real UNSW-NB15 parquet dataset
and generating crime pattern pairs for Siamese Network training.

UNSW-NB15 Attack Categories (our "crime types"):
    0 = Normal
    1 = Fuzzers
    2 = Backdoor
    3 = DoS
    4 = Exploits
    5 = Reconnaissance
    6 = Shellcode
    7 = Worms
    8 = Analysis
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DATA_DIR       = "data"
PARQUET_PATH   = "data/UNSW_NB15_training-set.parquet"  # real dataset
SYNTHETIC_PATH = "data/UNSW_NB15_training.csv"          # fallback
SEQUENCE_LEN   = 5
FEATURE_DIM    = 32
NUM_CLASSES    = 9


# ─── DATASET LOADER ───────────────────────────────────────────────────────────

def download_dataset():
    """
    Priority order:
    1. Real UNSW-NB15 parquet file (from Kaggle download)
    2. Existing synthetic CSV
    3. Generate new synthetic data
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Priority 1 — real parquet file
    if os.path.exists(PARQUET_PATH):
        print(f"[dataset] ✅ Found real UNSW-NB15 dataset: {PARQUET_PATH}")
        return PARQUET_PATH

    # Priority 2 — existing synthetic CSV
    if os.path.exists(SYNTHETIC_PATH):
        print(f"[dataset] Found synthetic dataset: {SYNTHETIC_PATH}")
        return SYNTHETIC_PATH

    # Priority 3 — generate synthetic
    print("[dataset] No dataset found — generating synthetic...")
    return generate_synthetic_dataset(SYNTHETIC_PATH)


# ─── SYNTHETIC FALLBACK ───────────────────────────────────────────────────────

def generate_synthetic_dataset(save_path):
    """
    Each attack class has a DISTINCT statistical signature.
    Small noise within class = similar patterns (same MO).
    Large distance between classes = different MOs.
    """
    np.random.seed(42)
    n_per_class = 800
    n_features  = 32

    attack_types = [
        "Normal", "Fuzzers", "Backdoor", "DoS",
        "Exploits", "Reconnaissance", "Shellcode", "Worms", "Analysis"
    ]

    class_means = {
        "Normal":         np.array([0.1, 0.1, 0.2, 0.1] + [0.15] * 28),
        "DoS":            np.array([0.9, 0.8, 0.1, 0.9] + [0.85] * 28),
        "Backdoor":       np.array([0.1, 0.1, 0.9, 0.2] + [0.15] * 28),
        "Exploits":       np.array([0.5, 0.6, 0.4, 0.5] + [0.55] * 28),
        "Reconnaissance": np.array([0.3, 0.2, 0.3, 0.8] + [0.25] * 28),
        "Fuzzers":        np.array([0.7, 0.5, 0.6, 0.4] + [0.65] * 28),
        "Shellcode":      np.array([0.4, 0.8, 0.2, 0.6] + [0.45] * 28),
        "Worms":          np.array([0.8, 0.3, 0.7, 0.3] + [0.75] * 28),
        "Analysis":       np.array([0.2, 0.4, 0.5, 0.2] + [0.30] * 28),
    }

    noise_std = 0.05
    all_features, all_labels = [], []

    for attack in attack_types:
        mean    = class_means[attack]
        samples = mean + np.random.randn(n_per_class, n_features) * noise_std
        samples = np.clip(samples, 0, 1)
        all_features.append(samples)
        all_labels.extend([attack] * n_per_class)

    X    = np.vstack(all_features)
    cols = [f"feature_{i}" for i in range(n_features)]
    df   = pd.DataFrame(X, columns=cols)
    df["label"] = all_labels
    df   = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(save_path, index=False)

    print(f"[dataset] Synthetic dataset → {save_path} ({len(df)} samples, {n_features} features)")
    return save_path


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def load_and_preprocess(data_path):
    """
    Loads parquet or CSV, cleans, encodes labels, scales features.
    Handles both real UNSW-NB15 and synthetic datasets automatically.
    """
    print(f"[dataset] Loading {data_path}...")

    # Load based on file type
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        print(f"[dataset] Loaded real UNSW-NB15 parquet file")
    else:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"[dataset] Loaded CSV file")

    print(f"[dataset] Raw shape: {df.shape}")
    print(f"[dataset] Columns sample: {list(df.columns[:8])}...")

    # Identify label column
    if "attack_cat" in df.columns:
        label_col = "attack_cat"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError("No label column found. Expected 'attack_cat' or 'label'.")

    # Drop non-feature columns
    drop_cols = [
        label_col, "id", "srcip", "sport", "dstip", "dsport",
        "proto", "state", "service", "attack_cat", "is_sm_ips_ports"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X_df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric features
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Encode attack labels to integers
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str).str.strip())

    # Normalize features
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_df.values).astype(np.float32)

    classes = list(le.classes_)
    print(f"[dataset] {len(X)} samples | {X.shape[1]} features | {len(classes)} classes")
    print(f"[dataset] Classes: {classes}")

    return X, y, classes


# ─── CRIME PATTERN CREATION ───────────────────────────────────────────────────

def create_crime_patterns(X, y, seq_len=SEQUENCE_LEN):
    """
    Groups individual network events into crime patterns (sequences).
    Each crime pattern = seq_len consecutive real events of same attack type.

    This simulates a multi-stage attack chain:
        event_1 (initial access) → event_2 (execution) → ... → event_5 (C2)
    """
    print(f"[dataset] Creating crime patterns (seq_len={seq_len})...")
    patterns, labels = [], []

    for cls in np.unique(y):
        idx  = np.where(y == cls)[0]
        data = X[idx]
        for i in range(0, len(data) - seq_len, 2):
            patterns.append(data[i : i + seq_len])
            labels.append(cls)

    patterns = np.array(patterns, dtype=np.float32)
    labels   = np.array(labels,   dtype=np.int64)
    print(f"[dataset] Created {len(patterns)} crime patterns")
    return patterns, labels


# ─── SIAMESE PAIR GENERATION ──────────────────────────────────────────────────

def create_pairs(patterns, labels, pairs_per_class=300):
    """
    Creates balanced positive and negative pairs for Siamese training.

    Positive pair (label=1): same attack type  = same MO
    Negative pair (label=0): diff attack type  = different MO
    """
    print("[dataset] Generating Siamese pairs...")
    pairs, targets = [], []

    for cls in np.unique(labels):
        same_idx = np.where(labels == cls)[0]
        diff_idx = np.where(labels != cls)[0]
        n_pos    = min(len(same_idx) - 1, pairs_per_class)

        # Positive pairs — same MO
        for i in range(n_pos):
            j = np.random.randint(0, len(same_idx))
            while j == i:
                j = np.random.randint(0, len(same_idx))
            pairs.append((patterns[same_idx[i]], patterns[same_idx[j]]))
            targets.append(1)

        # Negative pairs — different MO
        for i in range(n_pos):
            j = np.random.choice(diff_idx)
            pairs.append((patterns[same_idx[i % len(same_idx)]], patterns[j]))
            targets.append(0)

    combined = list(zip(pairs, targets))
    np.random.shuffle(combined)
    pairs, targets = zip(*combined)

    pos = sum(targets)
    neg = len(targets) - pos
    print(f"[dataset] {len(pairs)} pairs → positive: {pos}, negative: {neg}")
    return list(pairs), list(targets)


# ─── PYTORCH DATASET ──────────────────────────────────────────────────────────

class CrimePairDataset(Dataset):
    """
    PyTorch Dataset for Siamese pairs.
    Each item: (pattern_A, pattern_B, label)
        pattern shape : (seq_len, feature_dim)
        label         : 1.0 = same MO, 0.0 = different MO
    """
    def __init__(self, pairs, targets):
        self.pairs   = pairs
        self.targets = targets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        return (
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def build_dataloaders(batch_size=32, seq_len=SEQUENCE_LEN, test_size=0.2):
    """
    Full pipeline:
    load → preprocess → crime patterns → pairs → DataLoaders
    """
    # 1. Load dataset
    data_path = download_dataset()

    # 2. Preprocess
    X, y, class_names = load_and_preprocess(data_path)

    # 3. Fix to FEATURE_DIM
    if X.shape[1] < FEATURE_DIM:
        pad = np.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=np.float32)
        X   = np.hstack([X, pad])
    else:
        X   = X[:, :FEATURE_DIM]

    feature_dim = X.shape[1]

    # 4. Crime patterns
    patterns, labels = create_crime_patterns(X, y, seq_len=seq_len)

    # 5. Siamese pairs
    pairs, targets = create_pairs(patterns, labels)

    # 6. Split
    split    = int(len(pairs) * (1 - test_size))
    train_ds = CrimePairDataset(pairs[:split],  targets[:split])
    val_ds   = CrimePairDataset(pairs[split:],  targets[split:])

    # 7. DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n[dataset] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"[dataset] Feature dim  : {feature_dim} | Classes: {len(class_names)}\n")

    return train_loader, val_loader, feature_dim, len(class_names), class_names


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, feat_dim, n_cls, cls_names = build_dataloaders()
    a, b, lbl = next(iter(train_loader))
    print(f"Batch → A: {a.shape}  B: {b.shape}  Labels: {lbl.shape}")
    print(f"Sample label → {lbl[0].item()} (1=same MO, 0=different MO)")