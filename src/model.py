"""
model.py
--------
Hybrid Siamese + CNN + Bi-LSTM architecture for cybercrime
modus operandi (MO) recognition.

Architecture flow:
    Crime Pattern (seq_len, features)
        │
        ▼
    CNN Feature Extractor       ← extracts fingerprint from each artifact
        │
        ▼
    Bi-LSTM MO Modeler          ← understands the attack sequence
        │
        ▼
    MO Vector                   ← abstract representation of the crime's MO
        │
    [Twin A] ──distance── [Twin B]
        │
        ▼
    Similarity Score (0.0 - 1.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── CNN FEATURE EXTRACTOR ────────────────────────────────────────────────────

class CNNExtractor(nn.Module):
    """
    Convolutional Neural Network that processes a single artifact
    and produces a compact feature vector (fingerprint).

    Think of this as the component that:
    - Looks at a malware binary image and extracts its structural pattern
    - Reads a phishing email and extracts its linguistic fingerprint
    - Analyzes a network log plot and extracts its communication pattern

    Input  : (batch, seq_len, feature_dim)
    Output : (batch, seq_len, cnn_out_dim)
             — one fingerprint vector per artifact in the sequence
    """

    def __init__(self, feature_dim=64, cnn_out_dim=128):
        super(CNNExtractor, self).__init__()

        self.feature_dim = feature_dim
        self.cnn_out_dim = cnn_out_dim

        # 1D Convolutions treat each feature vector as a 1D signal
        # kernel_size=3 means each filter looks at 3 consecutive features
        self.conv1 = nn.Conv1d(
            in_channels  = 1,
            out_channels = 32,
            kernel_size  = 3,
            padding      = 1
        )
        self.conv2 = nn.Conv1d(
            in_channels  = 32,
            out_channels = 64,
            kernel_size  = 3,
            padding      = 1
        )
        self.conv3 = nn.Conv1d(
            in_channels  = 64,
            out_channels = 128,
            kernel_size  = 3,
            padding      = 1
        )

        # Batch normalization stabilizes training
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Adaptive pooling always outputs fixed size regardless of input length
        self.pool = nn.AdaptiveAvgPool1d(4)

        # Dropout prevents overfitting
        self.dropout = nn.Dropout(0.3)

        # Final fully connected layer maps to cnn_out_dim
        self.fc = nn.Linear(128 * 4, cnn_out_dim)

    def forward(self, x):
        """
        x shape: (batch * seq_len, feature_dim)
        We add a channel dim → (batch * seq_len, 1, feature_dim)
        then apply convolutions along the feature dimension.
        """
        x = x.unsqueeze(1)                          # add channel dim

        x = F.relu(self.bn1(self.conv1(x)))         # conv block 1
        x = F.relu(self.bn2(self.conv2(x)))         # conv block 2
        x = F.relu(self.bn3(self.conv3(x)))         # conv block 3

        x = self.pool(x)                            # fixed size pooling
        x = x.flatten(1)                            # flatten to vector
        x = self.dropout(x)
        x = F.relu(self.fc(x))                      # final fingerprint

        return x                                    # (batch*seq_len, cnn_out_dim)


# ─── BI-LSTM MO MODELER ───────────────────────────────────────────────────────

class BiLSTMModeler(nn.Module):
    """
    Bidirectional LSTM that reads a sequence of CNN fingerprints
    and produces a single MO vector representing the entire attack chain.

    Why Bidirectional?
    - Forward LSTM  : reads phishing → malware → C2 comms
    - Backward LSTM : reads C2 comms → malware → phishing
    - Combined      : full context — every event understood relative
                      to ALL other events in the attack chain

    Input  : (batch, seq_len, cnn_out_dim)
    Output : (batch, mo_dim)  ← the MO vector
    """

    def __init__(self, input_dim=128, hidden_dim=128, mo_dim=256, num_layers=2):
        super(BiLSTMModeler, self).__init__()

        self.hidden_dim = hidden_dim
        self.mo_dim     = mo_dim

        # bidirectional=True doubles the output size automatically
        # hidden_dim=128 → actual output = 128 * 2 = 256 (forward + backward)
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,       # ← this one word does the heavy lifting
            dropout       = 0.3
        )

        # Layer norm improves stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Project to final MO vector dimension
        self.fc = nn.Linear(hidden_dim * 2, mo_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)

        LSTM processes full sequence, we take the final hidden state
        which summarizes the entire attack chain — that IS the MO vector.
        """
        lstm_out, (hidden, _) = self.lstm(x)

        # Take last timestep output (encodes full sequence context)
        last_out = lstm_out[:, -1, :]               # (batch, hidden_dim*2)

        last_out = self.layer_norm(last_out)
        last_out = self.dropout(last_out)
        mo_vec   = F.relu(self.fc(last_out))        # (batch, mo_dim)

        return mo_vec                               # the MO vector


# ─── TWIN SUBNETWORK ──────────────────────────────────────────────────────────

class TwinSubNetwork(nn.Module):
    """
    One twin of the Siamese Network.
    Combines CNN + Bi-LSTM to convert a raw crime pattern
    into an abstract MO vector.

    Pipeline:
        crime pattern (seq_len, features)
            → CNN processes each artifact individually
            → sequence of fingerprints (seq_len, cnn_out_dim)
            → Bi-LSTM reads the full sequence
            → single MO vector (mo_dim,)
    """

    def __init__(self, feature_dim=64, cnn_out_dim=128, mo_dim=256):
        super(TwinSubNetwork, self).__init__()

        self.cnn  = CNNExtractor(feature_dim=feature_dim,
                                 cnn_out_dim=cnn_out_dim)
        self.lstm = BiLSTMModeler(input_dim=cnn_out_dim,
                                  mo_dim=mo_dim)

    def forward(self, x):
        """
        x shape: (batch, seq_len, feature_dim)
        """
        batch, seq_len, feat_dim = x.shape

        # ── Step 1: CNN processes each artifact independently ──
        # Reshape so CNN sees each artifact as its own sample
        x_flat = x.reshape(batch * seq_len, feat_dim)   # (batch*seq_len, feat_dim)

        # CNN extracts fingerprint from each artifact
        fingerprints = self.cnn(x_flat)                 # (batch*seq_len, cnn_out_dim)

        # Reshape back to sequence form
        fingerprints = fingerprints.reshape(batch, seq_len, -1)  # (batch, seq_len, cnn_out_dim)

        # ── Step 2: Bi-LSTM reads the sequence of fingerprints ──
        mo_vector = self.lstm(fingerprints)              # (batch, mo_dim)

        return mo_vector


# ─── SIAMESE NETWORK ──────────────────────────────────────────────────────────

class SiameseCrimeMatcher(nn.Module):
    """
    Full Siamese Network for cybercrime MO matching.

    Takes TWO crime patterns as input.
    Both pass through IDENTICAL twin subnetworks (shared weights).
    Distance between their MO vectors = similarity score.

    High similarity → same threat actor / same MO
    Low  similarity → unrelated crimes

    This is the main model class used for training and inference.
    """

    def __init__(self, feature_dim=64, cnn_out_dim=128, mo_dim=256):
        super(SiameseCrimeMatcher, self).__init__()

        # Single twin subnetwork — shared weights means we use
        # the SAME instance for both inputs (not two separate instances)
        self.twin = TwinSubNetwork(
            feature_dim  = feature_dim,
            cnn_out_dim  = cnn_out_dim,
            mo_dim       = mo_dim
        )

        # Final classifier head
        # Takes absolute difference of MO vectors → similarity score
        self.classifier = nn.Sequential(
            nn.Linear(mo_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()            # output between 0 and 1
        )

    def forward_one(self, x):
        """Process a single crime pattern through the twin subnetwork."""
        return self.twin(x)

    def forward(self, pattern_a, pattern_b):
        """
        pattern_a : (batch, seq_len, feature_dim) — Crime Pattern 1
        pattern_b : (batch, seq_len, feature_dim) — Crime Pattern 2

        Both go through the SAME twin (shared weights).
        This is the key Siamese principle.
        """
        # Both patterns through same twin → MO vectors
        mo_vec_a = self.forward_one(pattern_a)     # (batch, mo_dim)
        mo_vec_b = self.forward_one(pattern_b)     # (batch, mo_dim)

        # Absolute difference captures how different the two MOs are
        # Small difference → similar MO → high similarity score
        diff = torch.abs(mo_vec_a - mo_vec_b)      # (batch, mo_dim)

        # Classify: same MO or different MO?
        similarity = self.classifier(diff)          # (batch, 1)

        return similarity.squeeze(1)               # (batch,)

    def get_mo_vector(self, pattern):
        """
        Returns the raw MO vector for a crime pattern.
        Used during inference to compare crimes directly.
        """
        with torch.no_grad():
            return self.forward_one(pattern)

    def compute_similarity(self, pattern_a, pattern_b):
        """
        Returns similarity score between two crime patterns.
        Score closer to 1.0 = same MO (same attacker)
        Score closer to 0.0 = different MO (unrelated)
        """
        with torch.no_grad():
            return self.forward(pattern_a, pattern_b)


# ─── MODEL SUMMARY ────────────────────────────────────────────────────────────

def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, feature_dim=64, seq_len=5):
    """Prints a clean summary of the model architecture."""
    print("=" * 60)
    print("  HYBRID SIAMESE CNN BI-LSTM ARCHITECTURE")
    print("=" * 60)
    print(f"  Feature dim per artifact : {feature_dim}")
    print(f"  Sequence length          : {seq_len} artifacts")
    print(f"  CNN output dim           : 128")
    print(f"  Bi-LSTM hidden dim       : 128 × 2 = 256 (bidirectional)")
    print(f"  MO vector dim            : 256")
    print(f"  Total parameters         : {count_parameters(model):,}")
    print("=" * 60)
    print("\n  Flow:")
    print("  Crime Pattern (5, 64)")
    print("      │")
    print("      ▼")
    print("  CNN Extractor → fingerprints (5, 128)")
    print("      │")
    print("      ▼")
    print("  Bi-LSTM Modeler → MO Vector (256,)")
    print("      │")
    print("  [Twin A] ──abs diff── [Twin B]")
    print("      │")
    print("      ▼")
    print("  Similarity Score (0.0 - 1.0)")
    print("=" * 60)


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[model] Using device: {device}")

    # Build model
    model = SiameseCrimeMatcher(feature_dim=64, cnn_out_dim=128, mo_dim=256)
    model = model.to(device)

    # Print summary
    print_model_summary(model)

    # Test forward pass with dummy data
    batch_size = 4
    seq_len    = 5
    feat_dim   = 64

    dummy_a = torch.randn(batch_size, seq_len, feat_dim).to(device)
    dummy_b = torch.randn(batch_size, seq_len, feat_dim).to(device)

    scores = model(dummy_a, dummy_b)

    print(f"\n[model] Test forward pass:")
    print(f"  Input A shape  : {dummy_a.shape}")
    print(f"  Input B shape  : {dummy_b.shape}")
    print(f"  Output scores  : {scores}")
    print(f"  Output shape   : {scores.shape}")
    print(f"\n[model] Model built successfully!")