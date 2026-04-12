# Architecture Deep Dive
### Complete Technical Explanation of CNN + Bi-LSTM + Siamese Network

---

This document provides a comprehensive technical explanation of every component in the crime pattern recognition system. By the end, you should understand how data flows through each component and why each choice was made.

---

## Table of Contents

1. [The Problem: Cybercrime MO Detection](#1-the-problem-cybercrime-mo-detection)
2. [Why Deep Learning?](#2-why-deep-learning)
3. [CNN: Spatial Feature Extraction](#3-cnn-spatial-feature-extraction)
4. [Bi-LSTM: Temporal Sequence Modeling](#4-bi-lstm-temporal-sequence-modeling)
5. [Siamese Network: Learning Similarity](#5-siamese-network-learning-similarity)
6. [The Complete Architecture](#6-the-complete-architecture)
7. [Training Process](#7-training-process)
8. [Inference: Making Predictions](#8-inference-making-predictions)
9. [SHAP: Explainable AI](#9-shap-explainable-ai)
10. [Key Parameters Summary](#10-key-parameters-summary)

---

## 1. The Problem: Cybercrime MO Detection

### 1.1 What is Modus Operandi?

In forensic science, **Modus Operandi (MO)** refers to the method or technique a criminal uses to commit a crime. Just as traditional criminals have distinctive methods (how they break in, what tools they use, how they escape), cybercriminals have distinctive patterns in their attacks.

### 1.2 The Challenge in Cyberspace

Two seemingly different cyberattacks might actually be related:

| Attack A | Attack B | Are They Related? |
|---|---|---|
| Target: Company X | Target: Company Y | Unknown |
| IP: 1.2.3.4 | IP: 5.6.7.8 | Unknown |
| Time: 9:00 AM | Time: 2:00 PM | Unknown |

**Traditional analysis** would say these are unrelated (different targets, IPs, times).

**MO-based analysis** looks deeper:
- Both attacks use the same TCP flag patterns
- Both have identical packet size distributions
- Both show the same timing between packets
- Same post-exploit behavior

This suggests the **same threat actor** — even though surface details differ.

### 1.3 Why This Is a Similarity Problem

We need to answer: *"How similar are these two crime patterns?"*

This is fundamentally different from classification (labeling an input as "DoS" or "Backdoor"). We need a system that:
1. Takes two patterns as input
2. Outputs a similarity score (0.0 to 1.0)
3. Learns from examples of same/different MO

This is exactly what a **Siamese Network** does.

---

## 2. Why Deep Learning?

### 2.1 Limitations of Traditional Machine Learning

Traditional ML approaches have significant drawbacks:

| Approach | Limitations |
|---|---|
| Rule-based systems | Cannot capture complex patterns; easily evaded |
| Manual feature engineering | Requires domain experts; may miss subtle patterns |
| Supervised classifiers | Need labeled data for each attack type |
| Anomaly detection | Cannot determine "same attacker"; only "anomalous" |

### 2.2 Advantages of Deep Learning

Deep learning offers fundamental advantages:

1. **Automatic Feature Learning**
   - Learns relevant features directly from raw data
   - No manual feature engineering needed
   - Discovers patterns humans might miss

2. **Non-linear Relationships**
   - Can learn complex, non-linear patterns
   - Captures interactions between features

3. **One-Shot Learning Capability**
   - Can identify new attack types from few examples
   - Doesn't need to retrain for every new attack

4. **Scalability**
   - Works with massive datasets
   - Improves with more data

### 2.3 Why the Hybrid Architecture?

No single neural network type handles all aspects well:

| Component | What It Captures | Why It Matters |
|---|---|---|
| **CNN** | Local spatial patterns | Attack signatures in individual events |
| **Bi-LSTM** | Sequential attack chains | Multi-stage attack progress |
| **Siamese** | Comparative similarity | Same vs different MO |

A system using only CNN would miss sequential patterns.
A system using only LSTM couldn't compare two incidents.
A Siamese network alone needs embeddings (provided by CNN+LSTM).

That's why we combine all three.

---

## 3. CNN: Spatial Feature Extraction

### 3.1 What is a CNN?

A **Convolutional Neural Network (CNN)** uses mathematical operations called **convolutions** to automatically learn spatial patterns from data.

While popularly used for images (2D grids), CNNs work on any structured data — including network traffic features treated as a 1D signal.

### 3.2 How Convolutions Work

Think of a convolution as a "sliding window" that looks at local patterns:

```
Input Features:     [f1, f2, f3, f4, f5, f6, f7, f8, ...]
                   └──────┘
Filter (kernel):    [w1, w2, w3]
                   └──────┘
                   = w1×f1 + w2×f2 + w3×f3
                   
Output Feature:    [g1, g2, g3, g4, g5, g6, ...]
                          └──────┘
                   Filter slides by 1 → next output
```

Each filter learns to detect a specific pattern:
- Filter 1: Detects high bandwidth
- Filter 2: Detects TCP anomalies
- Filter 3: Detects packet rate spikes

### 3.3 CNN Architecture in This Project

Here's the complete CNN implementation from `src/model.py`:

```python
class CNNExtractor(nn.Module):
    """
    Convolutional Neural Network that processes a single artifact
    and produces a compact feature vector (fingerprint).
    
    Input  : (batch, seq_len, feature_dim)    # e.g., (32, 5, 31)
    Output : (batch, seq_len, cnn_out_dim)    # e.g., (32, 5, 64)
    """
    
    def __init__(self, feature_dim=64, cnn_out_dim=128):
        super(CNNExtractor, self).__init__()
        
        # Conv1d: 1D convolution (treat features as 1D signal)
        # in_channels=1: Each feature is treated as a single channel
        # out_channels=32: Produce 32 different feature maps
        # kernel_size=3: Each filter looks at 3 consecutive features
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1  # Keep same length
        )
        
        # Second conv layer: More abstract features
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.relu = nn.ReLU()        # Activation function
        self.pool = nn.MaxPool1d(2)  # Downsample by 2
        self.dropout = nn.Dropout(0.3)  # Regularization
    
    def forward(self, x):
        """
        Forward pass through CNN
        
        Args:
            x: Input tensor, shape (batch, seq_len, feature_dim)
        
        Returns:
            Feature tensor, shape (batch, seq_len, cnn_out_dim)
        """
        # Reshape: (batch, seq_len, features) → (batch, 1, seq_len×features)
        # This treats the entire sequence as a single 1D signal
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        
        # Conv1 → ReLU → Conv2 → ReLU → Pool
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape back to (batch, seq_len, cnn_out_dim)
        x = x.view(batch_size, -1, self.cnn_out_dim)
        
        return x
```

### 3.4 Why CNN for Network Data?

Network traffic has inherent spatial structure:

1. **Feature Groupings**
   - Bytes: `sbytes`, `dbytes` often vary together
   - Packets: `spkts`, `dpkts` related
   - TCP: `tcprtt`, `synack`, `ackdat` form a group

2. **Attack Signatures**
   - DoS: High packet counts + low bytes
   - Data exfiltration: High dst_bytes + low src_bytes
   - Port scan: Many connections + low data transfer

3. **Local Patterns**
   - CNNs automatically learn these local relationships
   - No manual feature grouping needed

### 3.5 Input/Output Shapes

| Stage | Shape | Description |
|---|---|---|
| Input | `(batch, 5, 31)` | 5 events, 31 features each |
| After reshape | `(batch, 1, 155)` | Flattened to 1D signal |
| After conv1 | `(batch, 32, 155)` | 32 feature maps |
| After conv2 | `(batch, 64, 155)` | 64 feature maps |
| After pool | `(batch, 64, 77)` | Downsampled |
| After reshape | `(batch, 5, 64)` | Per-event features |

---

## 4. Bi-LSTM: Temporal Sequence Modeling

### 4.1 What is an LSTM?

**Long Short-Term Memory (LSTM)** is a type of recurrent neural network (RNN) designed to handle sequences and remember long-term dependencies.

Standard RNNs suffer from the **vanishing gradient problem** — they can't learn relationships between events far apart in time. LSTMs solve this with a special architecture that maintains long-term memory.

### 4.2 LSTM Architecture: The Three Gates

Each LSTM cell has three "gates" that control information flow:

```
┌─────────────────────────────────────────────────────────────────┐
│                      LSTM Cell                                   │
│                                                                  │
│  Input: New event                                                 │
│       │                                                          │
│       ▼                                                          │
│  ┌───────────────┐                                               │
│  │  Forget Gate  │ ◄── What old memory to discard?               │
│  └───────────────┘                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌───────────────┐                                               │
│  │  Input Gate  │ ◄── What new information to store?            │
│  └───────────────┘                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌───────────────────────────────────────┐                       │
│  │         Memory Cell (Long-term)       │                      │
│  │    (remembers patterns from all past)  │                      │
│  └───────────────────────────────────────┘                       │
│       │                                                          │
│       ▼                                                          │
│  ┌───────────────┐                                               │
│  │ Output Gate  │ ◄── What to output based on memory?            │
│  └───────────────┘                                               │
│       │                                                          │
│       ▼                                                          │
│  Output: Updated memory + current output                          │
└─────────────────────────────────────────────────────────────────┘
```

**Forget Gate**: Decides what from the past to discard
```python
forget = sigmoid(W_f × input + U_f × previous_hidden)
# If forget ≈ 0: discard old memory
# If forget ≈ 1: keep old memory
```

**Input Gate**: Decides what new information to add
```python
input_gate = sigmoid(W_i × input + U_i × previous_hidden)
new_content = tanh(W_c × input + U_c × previous_hidden)
# Store: input_gate × new_content
```

**Output Gate**: Decides what to output
```python
output_gate = sigmoid(W_o × input + U_o × previous_hidden)
hidden_state = output_gate × tanh(memory_cell)
```

### 4.3 What is Bidirectional LSTM?

A standard LSTM reads sequences in one direction only:
```
Event 1 → Event 2 → Event 3 → Event 4 → Event 5
```
This knows the **past** but not the **future**.

A **Bidirectional LSTM** reads sequences twice:
```
Forward:  Event 1 → Event 2 → Event 3 → Event 4 → Event 5
Backward: Event 5 ← Event 4 ← Event 3 ← Event 2 ← Event 1
```
Each event gets context from both directions.

### 4.4 Bi-LSTM Implementation

```python
class BiLSTMExtractor(nn.Module):
    """
    Bidirectional LSTM that processes the sequence of CNN fingerprints
    and produces a single MO (Modus Operandi) vector.
    
    The key insight: An attack isn't just five events.
    It's a STORY: What happened before affects what happens after.
    
    Input  : (batch, seq_len, cnn_out_dim)   # e.g., (32, 5, 64)
    Output : (batch, mo_dim * 2)           # e.g., (32, 256) - *2 for bidirectional
    """
    
    def __init__(self, input_dim=64, hidden_dim=128):
        super(BiLSTMExtractor, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM
        # input_dim: Features per event (64 from CNN)
        # hidden_dim: Size of hidden state (128)
        # bidirectional=True: Two passes, forward + backward
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True  # (batch, seq, features) format
        )
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass through Bi-LSTM
        
        Args:
            x: CNN features, shape (batch, seq_len, cnn_out_dim)
        
        Returns:
            MO vector, shape (batch, mo_dim * 2)
        """
        # LSTM processes the sequence
        # output: (batch, seq_len, hidden_dim * 2) - *2 for bidir
        # hidden: (num_layers * 2, batch, hidden_dim) - *2 for bidir
        output, (hidden, cell) = self.lstm(x)
        
        # Take only the final hidden states
        # Forward + backward are concatenated
        # Shape: (batch, hidden_dim * 2)
        mo_vector = torch.cat([hidden[0], hidden[1]], dim=1)
        
        mo_vector = self.dropout(mo_vector)
        
        return mo_vector
```

### 4.5 Why Bi-LSTM for Attack Sequences?

Cyberattacks are inherently sequential:

```
Stage 1: RECONNAISSANCE
  ↓ Port scan, vulnerability discovery
Stage 2: INITIAL ACCESS  
  ↓ Exploit, gain foothold
Stage 3: PRIVILEGE ESCALATION
  ↓ Elevate permissions
Stage 4: LATERAL MOVEMENT
  ↓ Spread to other systems
Stage 5: DATA EXFILTRATION / DAMAGE
  ↓ Steal data, cause damage
```

**Why Bi-LSTM matters:**

1. **Full Context**: Every event knows what happened before AND what happens after
2. **Retroactive Understanding**: Later events can reinterpret earlier ones
3. **Attack Chain Understanding**: Complete story of the attack

**Example**: A TCP connection might look normal in isolation. But if we know:
- It came AFTER a port scan (reconnaissance)
- It came BEFORE data transfer (exfiltration)

Then it's clearly part of an attack chain.

### 4.6 Input/Output Shapes

| Stage | Shape | Description |
|---|---|---|
| Input | `(batch, 5, 64)` | 5 events, 64 CNN features |
| LSTM output | `(batch, 5, 256)` | 5 timesteps, forward+backward |
| Hidden states | `(2, batch, 128)` | 2 directions |
| Final concat | `(batch, 256)` | MO vector |
| After dropout | `(batch, 256)` | Final MO vector |

---

## 5. Siamese Network: Learning Similarity

### 5.1 What is a Siamese Network?

A **Siamese Network** is a neural network architecture that compares two inputs and outputs a **similarity score**. The key innovation is **shared weights** — two "twin" networks that are identical copies.

```
              ┌─────────────────┐
   Input A ──▶│  Twin Network 1  │──▶ Embedding A
              └─────────────────┘
                    │
                Shared Weights
                (identical copy)
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

### 5.2 Why Shared Weights Matter

If we used two different networks:
```
Network A → Embedding A
Network B → Embedding B × (different weights)
```

Then differences in embeddings could be due to:
1. Actual differences in the inputs
2. Differences in the networks themselves

With shared weights:
```
Network → Embedding A
Network → Embedding B × (same weights)
```

Now any difference MUST be from the inputs — not the network.

### 5.3 Complete Siamese Implementation

```python
class SiameseCrimeMatcher(nn.Module):
    """
    Complete Siamese Network for crime pattern comparison.
    
    Architecture:
        1. CNN Extractor: Extract spatial features from each event
        2. Bi-LSTM: Model temporal attack sequence
        3. Twin Networks: Shared weights for fair comparison
        4. Distance + Dense: Compute similarity score
    
    Forward pass computes similarity between TWO patterns.
    """
    
    def __init__(self, feature_dim=31, seq_len=5, cnn_out_dim=64, mo_dim=128):
        super(SiameseCrimeMatcher, self).__init__()
        
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.cnn_out_dim = cnn_out_dim
        self.mo_dim = mo_dim
        
        # Shared twin networks (CNN + Bi-LSTM)
        # Note: Same instance = shared weights
        self.cnn = CNNExtractor(feature_dim, cnn_out_dim)
        self.lstm = BiLSTMExtractor(cnn_out_dim, mo_dim)
        
        # Similarity computation layers
        # Takes absolute difference between embeddings
        # Maps from (256,) to (1,) — the similarity score
        self.fc1 = nn.Linear(mo_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward_one(self, x):
        """
        Process a single pattern through twin network.
        
        Args:
            x: Pattern tensor, shape (batch, seq_len, feature_dim)
        
        Returns:
            MO vector, shape (batch, mo_dim * 2)
        """
        # CNN → Bi-LSTM → MO vector
        cnn_features = self.cnn(x)
        mo_vector = self.lstm(cnn_features)
        
        return mo_vector
    
    def compute_similarity(self, x1, x2):
        """
        Compute similarity between two patterns.
        
        Args:
            x1: Pattern A, shape (batch, seq_len, feature_dim)
            x2: Pattern B, shape (batch, seq_len, feature_dim)
        
        Returns:
            Similarity score, shape (batch, 1)
        """
        # Get embeddings from shared twin network
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        
        # Absolute difference between embeddings
        # This is a common similarity measure
        diff = torch.abs(embedding1 - embedding2)
        
        # Dense layers to compute similarity score
        out = self.relu(self.fc1(diff))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Sigmoid to get score between 0 and 1
        similarity = torch.sigmoid(out)
        
        return similarity
    
    def forward(self, x1, x2):
        """
        Main forward pass.
        
        Args:
            x1: Pattern A, shape (batch, seq_len, feature_dim)
            x2: Pattern B, shape (batch, seq_len, feature_dim)
        
        Returns:
            Similarity score, shape (batch, 1)
        """
        return self.compute_similarity(x1, x2)
```

### 5.4 How Similarity is Computed

The similarity computation follows these steps:

```
Step 1: Get embeddings
        embedding_A = twin_network(pattern_A)  → (128,)
        embedding_B = twin_network(pattern_B)  → (128,)

Step 2: Compute absolute difference
        diff = |embedding_A - embedding_B|      → (128,)

Step 3: Project to similarity
        hidden = ReLU(W × diff + b)             → (64,)
        score = sigmoid(W × hidden + b)         → (1,)
```

The **absolute difference** is key because:
- If patterns are similar → embeddings similar → difference small → score high
- If patterns are different → embeddings different → difference large → score low

### 5.5 Why This Works for MO Detection

The Siamese architecture is perfect for MO detection because:

1. **Same Ruler**: Both patterns are measured by the same network
2. **Learns Similarity**: Optimized to produce high scores for same-MO pairs
3. **Generalizes**: Can detect new attack types from learned patterns

---

## 6. The Complete Architecture

### 6.1 Full System Data Flow

Here's how all components fit together:

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Crime Pattern A (5 events)                                       │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│   │ Event 1 │ │ Event 2 │ │ Event 3 │ │ Event 4 │ │ Event 5 │    │
│   │[31 feat]│ │[31 feat]│ │[31 feat]│ │[31 feat]│ │[31 feat]│    │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │
│        │           │           │           │           │              │
│        └───────────┴───────────┴───────────┴───────────┘              │
│                          │                                        │
│                          ▼                                        │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │           CNN Extractor (Twin Network)                    │     │
│   │     (extracts spatial fingerprint from each event)       │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                          │                                        │
│                          ▼                                        │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                Bi-LSTM MO Modeler                          │     │
│   │     (models temporal attack sequence)                     │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                          │                                        │
│                          ▼                                        │
│                   MO Vector A (128-dim)                           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                                    │
                           ┌────────┴────────┐
                           │ Siamese Compare │
                           └────────┬────────┘
                                    │
┌───────────────────────────────────────────────────────────────────┐
│   Crime Pattern B (5 events)                                       │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│   │ Event 1 │ │ Event 2 │ │ Event 3 │ │ Event 4 │ │ Event 5 │    │
│   │[31 feat]│ │[31 feat]│ │[31 feat]│ │[31 feat]│ │[31 feat]│    │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │
│        │           │           │           │           │              │
│        └───────────┴───────────┴───────────┴───────────┘              │
│                          │                                        │
│                          ▼                                        │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │           CNN Extractor (Twin Network)                    │     │
│   │     (extracts spatial fingerprint from each event)       │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                          │                                        │
│                          ▼                                        │
│   ┌──────────────────��─��────────────────────────────────────────┐     │
│   │                Bi-LSTM MO Modeler                          │     │
│   │     (models temporal attack sequence)                     │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                          │                                        │
│                          ▼                                        │
│                   MO Vector B (128-dim)                           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │ |MO_A - MO_B|   │
                           │  (Absolute)   │
                           └───────┬────────┘
                                   │
                           ┌───────▼───────┐
                           │ Dense→Sigmoid│
                           └───────▼───────┘
                                   │
                                   ▼
                          Similarity Score
                             (0.0-1.0)
                             
                   ┌──────────┴──────────┐
                   ▼                     ▼
             SAME MO               DIFFERENT MO
          (score ≥ 0.5)          (score < 0.5)
```

### 6.2 Component-by-Component Summary

| Component | Input | Output | Purpose |
|---|---|---|---|
| Input | `(5, 31)` | Raw features | Network event data |
| CNN | `(5, 31)` | `(5, 64)` | Spatial fingerprints |
| Bi-LSTM | `(5, 64)` | `(128,)` | MO vector |
| Twin A | Pattern A | MO_A | Embed pattern A |
| Twin B | Pattern B | MO_B | Embed pattern B |
| Diff | MO_A, MO_B | \|MO_A-MO_B\| | Compare embeddings |
| Dense | Diff | Score | Final similarity |

---

## 7. Training Process

### 7.1 How Training Works

Training a Siamese network is different from training a typical classifier. We need **pairs** of patterns, not individual samples.

### 7.2 Training Data: Positive and Negative Pairs

| Pair Type | How Created | Label |
|---|---|---|
| **Positive (Same MO)** | Two patterns from same attack category | 1.0 |
| **Negative (Different MO)** | Two patterns from different categories | 0.0 |

**Example:**
```python
# Positive pair (DoS + DoS) → label = 1.0
pattern_dos_1 = [e1, e2, e3, e4, e5]  # DoS attack
pattern_dos_2 = [e1, e2, e3, e4, e5]  # DoS attack

# Negative pair (DoS + Backdoor) → label = 0.0
pattern_dos = [e1, e2, e3, e4, e5]    # DoS attack  
pattern_bd = [e1, e2, e3, e4, e5]    # Backdoor attack
```

### 7.3 Contrastive Loss Function

The loss function teaches the network to:
- **Maximize** similarity for positive pairs (same MO)
- **Minimize** similarity for negative pairs (different MO)

```python
def contrastive_loss(similarity, label, margin=0.5):
    """
    Contrastive loss for Siamese network.
    
    Args:
        similarity: Predicted similarity score (0-1)
        label: Ground truth (1 = same MO, 0 = different MO)
        margin: Threshold for negative pairs
    
    Returns:
        Loss value
    """
    # Positive pairs: Pull embeddings together (minimize similarity → 0)
    loss_positive = label * (similarity ** 2)
    
    # Negative pairs: Push embeddings apart (maximize difference → > margin)
    loss_negative = (1 - label) * max(0, margin - similarity) ** 2
    
    # Total loss
    total_loss = loss_positive + loss_negative
    
    return total_loss
```

### 7.4 Training Loop

```python
def train_model(model, train_loader, val_loader, epochs=30):
    """
    Main training loop for Siamese network.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = contrastive_loss
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for pattern_a, pattern_b, label in train_loader:
            # Forward pass
            similarity = model(pattern_a, pattern_b)
            
            # Compute loss
            loss = criterion(similarity, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            predicted = (similarity >= 0.5).float()
            train_correct += (predicted == label).sum().item()
            train_total += label.size(0)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for pattern_a, pattern_b, label in val_loader:
                similarity = model(pattern_a, pattern_b)
                predicted = (similarity >= 0.5).float()
                val_correct += (predicted == label).sum().item()
                val_total += label.size(0)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Acc: {100*train_correct/train_total:.2f}%")
        print(f"  Val Acc: {100*val_correct/val_total:.2f}%")
```

### 7.5 Training Results

```
Epoch 01/30  →  Train Acc: 59.24%  Val Acc: 72.37%
Epoch 10/30  →  Train Acc: 87.97%  Val Acc: 80.12%
Epoch 20/30  →  Train Acc: 89.45%  Val Acc: 83.56%
Epoch 30/30  →  Train Acc: 90.04%  Val Acc: 84.93%
```

---

## 8. Inference: Making Predictions

### 8.1 Prediction Flow

After training, making predictions is straightforward:

```python
def predict(model, incident1_log, incident2_log):
    """
    Compare two incident logs and determine if same MO.
    """
    # Step 1: Load and preprocess incidents
    patterns1 = process_incident_log(incident1_log)
    patterns2 = process_incident_log(incident2_log)
    
    # Step 2: Find best match
    max_similarity = -1.0
    best_pattern_pair = (None, None)
    
    for p1 in patterns1:
        for p2 in patterns2:
            similarity = model(p1, p2)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_pattern_pair = (p1, p2)
    
    # Step 3: Generate verdict
    if max_similarity >= 0.5:
        verdict = "SAME MO"
    else:
        verdict = "DIFFERENT MO"
    
    return {
        "verdict": verdict,
        "similarity_score": max_similarity,
        "best_pattern_pair": best_pattern_pair
    }
```

### 8.2 Why Check All Combinations?

We iterate through ALL pattern pairs because:
- Different parts of Incident 1 might match different parts of Incident 2
- We want the BEST possible match between the two incidents

---

## 9. SHAP: Explainable AI

### 9.1 Why Explainability Matters

In forensic science, AI decisions must be **legally defensible**. We cannot use a "black box" that says "same attacker" without explaining why.

### 9.2 What is SHAP?

**SHAP (SHapley Additive exPlanations)** provides feature-level explanations:

```python
import shap

# Create explainer
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values
shap_values = explainer.shap_values([pattern_a, pattern_b])

# Output shows exactly which features drove the decision
# Example:
#   Feature         Impact
#   ──────────────────────
#   src_bytes       +0.35  (increases similarity)
#   tcp_rtt        -0.22  (decreases similarity)
#   syn_ack        +0.15  (increases similarity)
#   ct_dst_dport   +0.08  (slight increase)
#   ...
```

### 9.3 Benefits for Forensics

SHAP provides:
- **Feature-level attribution**: Which features most influenced the decision
- **Directional impact**: Did the feature increase or decrease similarity?
- **Legal defensibility**: Can explain in court why AI made this decision
- **Investigator validation**: Investigators can verify if explanations make sense

---

## 10. Key Parameters Summary

| Parameter | Value | Description |
|---|---|---|
| `feature_dim` | 31 | Input features (UNSW-NB15) |
| `seq_len` | 5 | Events per pattern |
| `cnn_out_dim` | 64 | CNN output features |
| `mo_dim` | 128 | MO vector size |
| `learning_rate` | 0.001 | Adam optimizer |
| `batch_size` | 32 | Training batch size |
| `epochs` | 30 | Training iterations |
| `margin` | 0.5 | Contrastive loss margin |
| `similarity_threshold` | 0.5 | SAME vs DIFFERENT MO |

---

## Next Steps

- **[Dataset Reference](dataset.md)**: Learn about the UNSW-NB15 dataset
- **[Processing Pipeline](pipeline.md)**: Understand the end-to-end flow
- **[API Reference](api.md)**: Build on the Flask API

---

<div align="center">
  <sub>Part of the Intelligent Crime Pattern Recognition System</sub>
</div>