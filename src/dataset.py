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
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ExifTags
from torchvision import transforms


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# --- Foundational ---
DATA_DIR = "data"
PARQUET_PATH = "data/UNSW_NB15_training-set.parquet"
SYNTHETIC_PATH = "data/UNSW_NB15_training.csv"
NUM_CLASSES = 9

# --- Modality-specific ---
LOG_SEQ_LEN = 5
LOG_FEATURE_DIM = 32
IMG_MAX_TIMED = 10  # Max number of timestamped images per incident
IMG_MAX_STATIC = 10   # Max number of static images per incident
IMG_SIZE = 128      # All images will be resized to this
BIN_IMG_SIZE = 32   # Binaries converted to images will be this size




# ─── IMAGE & BINARY PREPROCESSING ─────────────────────────────────────────────

def get_image_transforms(size):
    """Returns a composition of transforms for image processing."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_binary_transforms(size):
    """Returns transforms for binary-as-image processing (grayscale)."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def get_timestamp_from_image(image_path):
    """Tries to extract a timestamp from image EXIF data."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            # Tag 36867 is DateTimeOriginal
            return exif_data.get(36867, None)
    except Exception:
        return None
    return None

def process_incident_images(image_paths, transform):
    """
    Loads images, separates them into timed and static groups,
    and applies transformations.
    """
    timed_images, static_images = [], []
    timed_paths = []

    for path in image_paths:
        timestamp = get_timestamp_from_image(path)
        if timestamp:
            timed_paths.append((path, timestamp))
        else:
            img = Image.open(path).convert("RGB")
            static_images.append(transform(img))

    # Sort timestamped images chronologically
    timed_paths.sort(key=lambda x: x[1])
    for path, _ in timed_paths:
        img = Image.open(path).convert("RGB")
        timed_images.append(transform(img))

    return timed_images, static_images

def process_binary_as_image(binary_path, transform):
    """Reads a binary file, converts it to a square image, and transforms it."""
    if not os.path.exists(binary_path):
        return torch.zeros(1, BIN_IMG_SIZE, BIN_IMG_SIZE)
        
    with open(binary_path, 'rb') as f:
        byte_data = f.read()
    
    # Pad or truncate to a fixed size for consistent image dimensions
    size = BIN_IMG_SIZE * BIN_IMG_SIZE
    if len(byte_data) < size:
        byte_data += b'\x00' * (size - len(byte_data))
    else:
        byte_data = byte_data[:size]
        
    img_array = np.frombuffer(byte_data, dtype=np.uint8).reshape(BIN_IMG_SIZE, BIN_IMG_SIZE)
    img = Image.fromarray(img_array, 'L') # L for grayscale
    return transform(img)


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

def create_crime_patterns(X, y, seq_len=LOG_SEQ_LEN):
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


# ─── MULTI-MODAL SIAMESE DATASET ──────────────────────────────────────────

class MultiModalCrimeDataset(Dataset):
    """
    PyTorch Dataset for multi-modal Siamese pairs.
    Each item is a tuple containing data for two incidents and a label.
    (incident_A_data, incident_B_data, label)
    """
    def __init__(self, pairs, manifest_df, data_dir, scaler=None):
        self.pairs = pairs
        self.manifest = manifest_df.set_index('incident_id')
        self.data_dir = data_dir
        self.scaler = scaler
        self.img_transform = get_image_transforms(IMG_SIZE)
        self.bin_transform = get_binary_transforms(BIN_IMG_SIZE)

    def _get_incident_data(self, incident_id):
        """Loads all data for a single incident."""
        incident_info = self.manifest.loc[incident_id]

        # 1. Load Log Data from real UNSW-NB15 parquet data
        attack_type = incident_info.get('attack_type', 'Normal')
        log_data = self._generate_log_from_unsk(attack_type)

        # 2. Load Image Data
        image_folder = os.path.join(self.data_dir, incident_info['image_folder'])
        image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
        timed_imgs, static_imgs = process_incident_images(image_paths, self.img_transform)

        # 3. Load Binary Data
        binary_data = torch.zeros(1, BIN_IMG_SIZE, BIN_IMG_SIZE)
        binary_path_info = incident_info.get('binary_path')
        if isinstance(binary_path_info, str) and binary_path_info:
            binary_path = os.path.join(self.data_dir, binary_path_info)
            binary_data = process_binary_as_image(binary_path, self.bin_transform)

        # 4. Pad all data to max length
        padded_timed = torch.zeros(IMG_MAX_TIMED, 3, IMG_SIZE, IMG_SIZE)
        if timed_imgs:
            num_to_pad = min(IMG_MAX_TIMED, len(timed_imgs))
            padded_timed[:num_to_pad] = torch.stack(timed_imgs[:num_to_pad])
        
        padded_static = torch.zeros(IMG_MAX_STATIC, 3, IMG_SIZE, IMG_SIZE)
        if static_imgs:
            num_to_pad = min(IMG_MAX_STATIC, len(static_imgs))
            padded_static[:num_to_pad] = torch.stack(static_imgs[:num_to_pad])

        return log_data, padded_timed, padded_static, binary_data

    def _generate_log_from_unsk(self, attack_type):
        """Generate log sequence from real UNSW-NB15 data for given attack type."""
        parquet_path = os.path.join(self.data_dir, "UNSW_NB15_training-set.parquet")
        
        try:
            df = pd.read_parquet(parquet_path)
            attack_df = df[df['attack_cat'] == attack_type]
            
            if len(attack_df) == 0:
                attack_df = df[df['attack_cat'] != 'Normal'].sample(min(100, len(df[df['attack_cat'] != 'Normal'])))
            
            # Drop non-numeric columns
            drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'service', 'attack_cat', 'label', 'is_sm_ips_ports']
            feature_cols = [c for c in attack_df.columns if c not in drop_cols and attack_df[c].dtype in ['int64', 'float64']]
            
            # Sample rows
            if len(attack_df) >= LOG_SEQ_LEN:
                sampled = attack_df[feature_cols].sample(LOG_SEQ_LEN, random_state=42)
            else:
                sampled = attack_df[feature_cols].sample(min(LOG_SEQ_LEN, len(attack_df)), random_state=42)
            
            X = sampled.fillna(0).values.astype(np.float32)
            
            # Ensure correct shape
            if X.shape[0] < LOG_SEQ_LEN:
                padding = np.zeros((LOG_SEQ_LEN - X.shape[0], X.shape[1]), dtype=np.float32)
                X = np.vstack([X, padding])
            
            if X.shape[1] < LOG_FEATURE_DIM:
                padding = np.zeros((X.shape[0], LOG_FEATURE_DIM - X.shape[1]), dtype=np.float32)
                X = np.hstack([X, padding])
            elif X.shape[1] > LOG_FEATURE_DIM:
                X = X[:, :LOG_FEATURE_DIM]
            
            # Scale
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            return torch.tensor(X, dtype=torch.float32)
            
        except Exception as e:
            # Fallback to zeros if parquet loading fails
            return torch.zeros(LOG_SEQ_LEN, LOG_FEATURE_DIM)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        incident_id_a, incident_id_b, label = self.pairs[idx]
        
        data_a = self._get_incident_data(incident_id_a)
        data_b = self._get_incident_data(incident_id_b)
        
        return (*data_a, *data_b, torch.tensor(label, dtype=torch.float32))

def create_multi_modal_pairs(manifest_df, pairs_per_class=100):
    """Creates positive and negative pairs from the incident manifest."""
    pairs = []
    manifest_df['label'] = LabelEncoder().fit_transform(manifest_df['attack_type'])
    
    for label in manifest_df['label'].unique():
        same_class_incidents = manifest_df[manifest_df['label'] == label]['incident_id'].tolist()
        diff_class_incidents = manifest_df[manifest_df['label'] != label]['incident_id'].tolist()

        if len(same_class_incidents) < 2 or len(diff_class_incidents) < 1:
            continue
            
        # Create positive pairs
        for i in range(pairs_per_class):
            id_a, id_b = np.random.choice(same_class_incidents, 2, replace=False)
            pairs.append((id_a, id_b, 1.0))
            
        # Create negative pairs
        for i in range(pairs_per_class):
            id_a = np.random.choice(same_class_incidents)
            id_b = np.random.choice(diff_class_incidents)
            pairs.append((id_a, id_b, 0.0))
            
    np.random.shuffle(pairs)
    return pairs

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def build_dataloaders(batch_size=32, test_size=0.2):
    """
    Full pipeline for the multi-modal dataset.
    Loads real UNSW-NB15 data for training.
    """
    # First, load and scale the UNSW-NB15 data to create a scaler
    parquet_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.parquet")
    scaler = None
    
    try:
        print("[dataset] Loading UNSW-NB15 for scaler...")
        df = pd.read_parquet(parquet_path)
        drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'service', 'attack_cat', 'label', 'is_sm_ips_ports']
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['int64', 'float64']]
        X_all = df[feature_cols].fillna(0).values.astype(np.float32)
        
        scaler = StandardScaler()
        scaler.fit(X_all[:10000])  # Fit on subset for speed
        print("[dataset] Scaler created from UNSW-NB15")
    except Exception as e:
        print(f"[dataset] Warning: Could not create scaler: {e}")
    
    manifest_path = os.path.join(DATA_DIR, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        print("Please run generate_multimodal_dataset.py first.")
        return None, None, {}

    manifest_df = pd.read_csv(manifest_path)
    
    # Create pairs
    pairs = create_multi_modal_pairs(manifest_df)
    
    # Split data
    split_idx = int(len(pairs) * (1 - test_size))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create datasets with scaler
    train_ds = MultiModalCrimeDataset(train_pairs, manifest_df, data_dir=DATA_DIR, scaler=scaler)
    val_ds = MultiModalCrimeDataset(val_pairs, manifest_df, data_dir=DATA_DIR, scaler=scaler)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n[dataset] Multi-modal dataset loaded successfully.")
    print(f"[dataset] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    # Define a config dict for the model
    data_config = {
        'log_feature_dim': LOG_FEATURE_DIM,
        'log_seq_len': LOG_SEQ_LEN
    }

    return train_loader, val_loader, data_config


# ─── QUICK TEST ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, data_config = build_dataloaders()
    if train_loader:
        batch = next(iter(train_loader))
        print(f"Batch shapes: log={batch[0].shape}, timed={batch[1].shape}, static={batch[2].shape}, bin={batch[3].shape}")
        print(f"            log={batch[4].shape}, timed={batch[5].shape}, static={batch[6].shape}, bin={batch[7].shape}, label={batch[8].shape}")