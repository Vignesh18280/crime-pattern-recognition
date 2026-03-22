"""
dataset.py
----------
Handles downloading, preprocessing of UNSW-NB15 dataset
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
"""

import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DATA_DIR       = "data"
DATASET_URL    = "https://raw.githubusercontent.com/datasets/unsw-nb15/main/UNSW_NB15_training-set.csv"
FALLBACK_URL   = "https://zenodo.org/record/4394993/files/UNSW_NB15_training-set.csv"
SEQUENCE_LEN   = 5      # number of artifacts per crime pattern
FEATURE_DIM    = 64     # feature vector size per artifact
NUM_CLASSES    = 9      # attack categories in UNSW-NB15


# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

def download_dataset():
    """
    Downloads UNSW-NB15 training set CSV into data/ folder.
    Falls back to synthetic data if download fails.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, "UNSW_NB15_training.csv")

    if os.path.exists(save_path):
        print(f"[dataset] Found existing dataset at {save_path}")
        return save_path

    print("[dataset] Downloading UNSW-NB15 dataset...")
    try:
        response = requests.get(DATASET_URL, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="UNSW-NB15"
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"[dataset] Saved to {save_path}")
        return save_path

    except Exception as e:
        print(f"[dataset] Download failed: {e}")
        print("[dataset] Falling back to synthetic dataset...")
        return generate_synthetic_dataset(save_path)


# ─── SYNTHETIC FALLBACK ───────────────────────────────────────────────────────

def generate_synthetic_dataset(save_path):
    """
    Generates a realistic synthetic cybercrime dataset if download fails.
    Each row = one network event with attack category label.

    Attack types mirror UNSW-NB15 categories so the rest of
    the pipeline works identically whether data is real or synthetic.
    """
    np.random.seed(42)
    n_samples = 5000

    attack_labels = [
        "Normal", "Fuzzers", "Backdoor", "DoS",
        "Exploits", "Reconnaissance", "Shellcode", "Worms", "Analysis"
    ]

    # Simulate realistic network features
    data = {
        "dur":       np.random.exponential(1.5, n_samples),
        "proto":     np.random.randint(0, 10, n_samples),
        "spkts":     np.random.poisson(5, n_samples),
        "dpkts":     np.random.poisson(8, n_samples),
        "sbytes":    np.random.randint(100, 50000, n_samples),
        "dbytes":    np.random.randint(100, 80000, n_samples),
        "rate":      np.random.exponential(500, n_samples),
        "sttl":      np.random.choice([64, 128, 255], n_samples),
        "dttl":      np.random.choice([64, 128, 255], n_samples),
        "sload":     np.random.exponential(1000, n_samples),
        "dload":     np.random.exponential(1000, n_samples),
        "sloss":     np.random.poisson(0.5, n_samples),
        "dloss":     np.random.poisson(0.5, n_samples),
        "sinpkt":    np.random.exponential(0.1, n_samples),
        "dinpkt":    np.random.exponential(0.1, n_samples),
        "sjit":      np.random.exponential(5, n_samples),
        "djit":      np.random.exponential(5, n_samples),
        "swin":      np.random.choice([0, 255], n_samples),
        "dwin":      np.random.choice([0, 255], n_samples),
        "ct_srv_src":np.random.randint(1, 20, n_samples),
        "label":     np.random.choice(attack_labels, n_samples,
                                      p=[0.4,0.1,0.05,0.1,0.15,0.1,0.04,0.03,0.03])
    }

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"[dataset] Synthetic dataset saved to {save_path} ({n_samples} samples)")
    return save_path


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path):
    """
    Loads CSV, cleans it, encodes labels, scales features.
    Returns:
        X      : np.ndarray of shape (N, num_features)
        y      : np.ndarray of integer class labels
        classes: list of class names
    """
    print(f"[dataset] Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Identify label column (UNSW-NB15 uses 'attack_cat' or 'label')
    if "attack_cat" in df.columns:
        label_col = "attack_cat"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError("No label column found in dataset.")

    # Drop non-numeric and identifier columns
    drop_cols = [label_col, "id", "srcip", "sport", "dstip", "dsport",
                 "proto", "state", "service", "attack_cat"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Keep only numeric features
    X_df = df.drop(columns=drop_cols, errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.fillna(0)

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str).str.strip())

    # Scale features
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_df.values)

    classes = list(le.classes_)
    print(f"[dataset] Loaded {len(X)} samples | {X.shape[1]} features | {len(classes)} classes")
    print(f"[dataset] Classes: {classes}")

    return X, y, classes


# ─── CRIME PATTERN CREATION ───────────────────────────────────────────────────

def create_crime_patterns(X, y, seq_len=SEQUENCE_LEN):
    """
    Groups individual network events into crime patterns (sequences).
    Each crime pattern = seq_len consecutive events of the same attack type.
    This simulates a multi-artifact attack chain:
        event_1 (phishing) → event_2 (malware) → ... → event_n (C2 comms)

    Returns:
        patterns : np.ndarray of shape (N, seq_len, feature_dim)
        labels   : np.ndarray of shape (N,) — attack category per pattern
    """
    print(f"[dataset] Creating crime patterns (seq_len={seq_len})...")
    patterns, labels = [], []

    unique_classes = np.unique(y)
    for cls in unique_classes:
        idx  = np.where(y == cls)[0]
        data = X[idx]

        # Slide a window of seq_len over this class's events
        for i in range(0, len(data) - seq_len, seq_len):
            seq = data[i : i + seq_len]               # (seq_len, features)
            patterns.append(seq)
            labels.append(cls)

    patterns = np.array(patterns, dtype=np.float32)
    labels   = np.array(labels,   dtype=np.int64)

    print(f"[dataset] Created {len(patterns)} crime patterns")
    return patterns, labels


# ─── SIAMESE PAIR GENERATION ──────────────────────────────────────────────────

def create_pairs(patterns, labels):
    """
    Creates positive and negative pairs for Siamese Network training.

    Positive pair (same_label=1): two patterns with the SAME attack category
        → model should output HIGH similarity (same MO)

    Negative pair (same_label=0): two patterns with DIFFERENT attack categories
        → model should output LOW similarity (different MO)

    Returns:
        pairs  : list of (pattern_A, pattern_B) tuples
        targets: list of 1 (same MO) or 0 (different MO)
    """
    print("[dataset] Generating Siamese pairs...")
    pairs, targets = [], []
    unique_classes = np.unique(labels)

    for cls in unique_classes:
        same_idx = np.where(labels == cls)[0]
        diff_idx = np.where(labels != cls)[0]

        # Positive pairs — same attack category
        for i in range(min(len(same_idx) - 1, 200)):
            pairs.append((patterns[same_idx[i]], patterns[same_idx[i + 1]]))
            targets.append(1)

        # Negative pairs — different attack categories
        for i in range(min(len(same_idx), 200)):
            j = np.random.choice(diff_idx)
            pairs.append((patterns[same_idx[i]], patterns[j]))
            targets.append(0)

    # Shuffle
    combined = list(zip(pairs, targets))
    np.random.shuffle(combined)
    pairs, targets = zip(*combined)

    print(f"[dataset] Total pairs: {len(pairs)} "
          f"(positive: {sum(targets)}, negative: {len(targets)-sum(targets)})")
    return list(pairs), list(targets)


# ─── PYTORCH DATASET ──────────────────────────────────────────────────────────

class CrimePairDataset(Dataset):
    """
    PyTorch Dataset wrapping our Siamese pairs.
    Each item returns:
        pattern_a : Tensor (seq_len, features)
        pattern_b : Tensor (seq_len, features)
        label     : Tensor scalar — 1.0 (same MO) or 0.0 (different MO)
    """

    def __init__(self, pairs, targets):
        self.pairs   = pairs
        self.targets = targets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b  = self.pairs[idx]
        label = self.targets[idx]
        return (
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def build_dataloaders(batch_size=32, seq_len=SEQUENCE_LEN, test_size=0.2):
    """
    Full pipeline:
        download → preprocess → create patterns → create pairs → split → DataLoaders

    Returns:
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        feature_dim  : int — number of features per timestep
        num_classes  : int — number of attack categories
        class_names  : list of class name strings
    """
    # 1. Download
    csv_path = download_dataset()

    # 2. Preprocess
    X, y, class_names = load_and_preprocess(csv_path)

    # 3. Pad or truncate features to fixed FEATURE_DIM
    if X.shape[1] < FEATURE_DIM:
        pad  = np.zeros((X.shape[0], FEATURE_DIM - X.shape[1]), dtype=np.float32)
        X    = np.hstack([X, pad])
    else:
        X    = X[:, :FEATURE_DIM]

    feature_dim = X.shape[1]

    # 4. Create crime pattern sequences
    patterns, labels = create_crime_patterns(X, y, seq_len=seq_len)

    # 5. Create Siamese pairs
    pairs, targets = create_pairs(patterns, labels)

    # 6. Train / val split
    split       = int(len(pairs) * (1 - test_size))
    train_pairs = pairs[:split];  train_targets = targets[:split]
    val_pairs   = pairs[split:];  val_targets   = targets[split:]

    # 7. Wrap in DataLoaders
    train_ds     = CrimePairDataset(train_pairs, train_targets)
    val_ds       = CrimePairDataset(val_pairs,   val_targets)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n[dataset] Train batches : {len(train_loader)}")
    print(f"[dataset] Val   batches : {len(val_loader)}")
    print(f"[dataset] Feature dim   : {feature_dim}")
    print(f"[dataset] Classes       : {len(class_names)}\n")

    return train_loader, val_loader, feature_dim, len(class_names), class_names


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, feat_dim, n_cls, cls_names = build_dataloaders()
    a, b, lbl = next(iter(train_loader))
    print(f"Batch shapes  → A: {a.shape}  B: {b.shape}  Labels: {lbl.shape}")
    print(f"Sample label  → {lbl[0].item()} (1=same MO, 0=different MO)")