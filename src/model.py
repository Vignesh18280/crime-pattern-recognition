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
import torchvision.models as models


# ─── MODALITY EXTRACTORS ────────────────────────────────────────────────────────
# Each extractor is a specialized neural network responsible for converting
# one type of data (logs, images, binaries) into a fixed-size embedding vector.
# ────────────────────────────────────────────────────────────────────────────────

class LogExtractor(nn.Module):
    """
    Processes a sequence of network log events using a CNN to find spatial
    patterns within each event, followed by a Bi-LSTM to model the
    temporal sequence of the entire attack chain.
    """
    def __init__(self, feature_dim=31, seq_len=5, cnn_out_dim=128, mo_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(4)
        )
        self.cnn_fc = nn.Linear(128 * 4, cnn_out_dim)
        
        self.lstm = nn.LSTM(cnn_out_dim, cnn_out_dim // 2, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.lstm_fc = nn.Linear(cnn_out_dim, mo_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        batch_size, seq_len, feat_dim = x.shape
        
        # CNN processing
        x_flat = x.view(batch_size * seq_len, 1, feat_dim)
        cnn_out = self.cnn(x_flat)
        cnn_out = cnn_out.flatten(1)
        fingerprints = F.relu(self.cnn_fc(self.dropout(cnn_out)))
        fingerprints_seq = fingerprints.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(fingerprints_seq)
        last_out = lstm_out[:, -1, :]
        
        # Final embedding
        mo_vec = F.relu(self.lstm_fc(self.dropout(last_out)))
        return mo_vec

class HybridImageExtractor(nn.Module):
    """
    Processes visual evidence for an incident. It intelligently handles a
    mix of timestamped (sequential) and non-timestamped (static) images.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # --- Sub-Branch 1: Sequential Extractor (for timestamped images) ---
        # This branch uses a CNN+LSTM architecture to understand temporal stories.
        self.sequential_cnn = self._create_cnn_branch(embedding_dim=256)
        self.sequential_lstm = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, batch_first=True)

        # --- Sub-Branch 2: Static Extractor (for unordered images) ---
        # This branch uses a CNN+Pooling architecture to find key evidence in a set.
        self.static_cnn = self._create_cnn_branch(embedding_dim=256)

        # --- Final Fusion Layer ---
        # This layer combines the knowledge from both branches.
        # Input size is 512 (sequential LSTM's bidirectional output) + 256 (static CNN's output)
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, self.embedding_dim)
        )

    def _create_cnn_branch(self, embedding_dim):
        """Helper to create a ResNet-based feature extractor."""
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = resnet.fc.in_features
        features = nn.Sequential(*list(resnet.children())[:-1])
        fc = nn.Linear(num_ftrs, embedding_dim)
        return nn.Sequential(features, nn.Flatten(1), fc)

    def forward(self, timed_images, static_images):
        """
        timed_images: (batch, seq_len, 3, H, W) - A sequence of images with timestamps.
        static_images: (batch, set_size, 3, H, W) - A set of unordered images.
        """
        # --- Process Sequential Branch ---
        # Check if there is actual data to process to avoid errors on empty tensors
        if timed_images.nelement() > 0:
            batch_size, seq_len, c, h, w = timed_images.shape
            timed_flat = timed_images.view(batch_size * seq_len, c, h, w)
            seq_cnn_out = self.sequential_cnn(timed_flat)
            seq_cnn_out_reshaped = seq_cnn_out.view(batch_size, seq_len, -1)
            _, (hidden, _) = self.sequential_lstm(seq_cnn_out_reshaped)
            sequential_embedding = torch.cat((hidden[0], hidden[1]), dim=1)
        else:
            # If no timed images, use a zero-vector placeholder
            sequential_embedding = torch.zeros(timed_images.shape[0], 512, device=timed_images.device)

        # --- Process Static Branch ---
        if static_images.nelement() > 0:
            batch_size, set_size, c, h, w = static_images.shape
            static_flat = static_images.view(batch_size * set_size, c, h, w)
            static_cnn_out = self.static_cnn(static_flat)
            static_cnn_out_reshaped = static_cnn_out.view(batch_size, set_size, -1)
            # Max-pooling across the set of images
            static_embedding, _ = torch.max(static_cnn_out_reshaped, dim=1)
        else:
            # If no static images, use a zero-vector placeholder
            static_embedding = torch.zeros(static_images.shape[0], 256, device=static_images.device)

        # --- Final Fusion ---
        combined_embedding = torch.cat((sequential_embedding, static_embedding), dim=1)
        final_image_embedding = self.fusion_layer(combined_embedding)
        
        return final_image_embedding


class BinaryExtractor(nn.Module):
    """
    Processes an executable binary file, which has been converted to a
    grayscale image. It uses a pre-trained ResNet to extract features
    from the binary's visual texture and structure.
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        # Use a pretrained ResNet, but modify it for grayscale input
        # and a custom output embedding dimension.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Original first layer: Conv2d(3, 64, ...). We need Conv2d(1, 64, ...).
        # We can average the weights of the original first layer across the R,G,B channels.
        original_weights = resnet.conv1.weight.clone()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = original_weights.mean(dim=1, keepdim=True)

        # Replace the final classification layer with our own embedding layer
        num_ftrs = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        # x shape: (batch, 1, height, width) - grayscale image
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



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


# ─── SIAMESE NETWORK ──────────────────────────────────────────────────────────

class SiameseCrimeMatcher(nn.Module):
    """
    The main multi-modal Siamese Network. It takes two incidents, each with
    potentially multiple data types (logs, images, binaries), processes them
    through specialized extractors, fuses the results, and compares the final
    MO vectors to compute a similarity score.
    """
    def __init__(self, log_feature_dim=31, log_seq_len=5,
                 log_embedding_dim=256, image_embedding_dim=512, binary_embedding_dim=256,
                 fused_embedding_dim=512):
        super().__init__()

        # --- Instantiate Modular Extractors ---
        self.log_extractor = LogExtractor(feature_dim=log_feature_dim, mo_dim=log_embedding_dim)
        self.image_extractor = HybridImageExtractor(embedding_dim=image_embedding_dim)
        self.binary_extractor = BinaryExtractor(embedding_dim=binary_embedding_dim)

        # --- Top-Level Fusion Layer ---
        # This layer combines the outputs from all modality extractors.
        total_embedding_dim = log_embedding_dim + image_embedding_dim + binary_embedding_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_embedding_dim, total_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(total_embedding_dim // 2, fused_embedding_dim)
        )

        # --- Final Classifier Head ---
        # This takes the absolute difference between the two fused MO vectors
        # and outputs the final similarity score (0.0 to 1.0).
        self.classifier = nn.Sequential(
            nn.Linear(fused_embedding_dim, fused_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fused_embedding_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward_one(self, log_data, timed_images, static_images, binary_data):
        """
        Processes a single, complete incident through all modality extractors
        and fuses the results into a single MO vector.
        """
        # Get embedding from each extractor
        log_embedding = self.log_extractor(log_data)
        image_embedding = self.image_extractor(timed_images, static_images)
        binary_embedding = self.binary_extractor(binary_data)

        # Concatenate all embeddings for late fusion
        combined_embeddings = torch.cat([log_embedding, image_embedding, binary_embedding], dim=1)

        # Fuse into the final MO vector
        fused_mo_vector = self.fusion_layer(combined_embeddings)
        return fused_mo_vector

    def forward(self, incident_a, incident_b):
        """
        The main forward pass of the Siamese network.
        """
        # Unpack data for Incident A
        log_a, timed_img_a, static_img_a, bin_a = incident_a
        # Unpack data for Incident B
        log_b, timed_img_b, static_img_b, bin_b = incident_b

        # Process each incident to get its final MO vector
        mo_vector_a = self.forward_one(log_a, timed_img_a, static_img_a, bin_a)
        mo_vector_b = self.forward_one(log_b, timed_img_b, static_img_b, bin_b)

        # Compute the absolute difference between the two MO vectors
        diff = torch.abs(mo_vector_a - mo_vector_b)

        # Pass the difference through the classifier to get the similarity score
        similarity = self.classifier(diff)
        
        return similarity.squeeze(1)

    def compute_similarity(self, incident_a, incident_b):
        """
        High-level wrapper for inference.
        """
        with torch.no_grad():
            return self.forward(incident_a, incident_b)



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