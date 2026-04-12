# Dataset Reference
### UNSW-NB15 Dataset, Features, and Attack Categories

---

This document provides a comprehensive reference for the dataset used in the crime pattern recognition system. By the end, you will understand the data source, every feature, and all attack categories.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Dataset Source](#2-dataset-source)
3. [Feature Reference](#3-feature-reference)
4. [Attack Categories](#4-attack-categories)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [Crime Pattern Generation](#6-crime-pattern-generation)
7. [Pair Generation for Training](#7-pair-generation-for-training)
8. [Sample Data Statistics](#8-sample-data-statistics)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Name** | UNSW-NB15 |
| **Full Name** | UNSW-NB15 Network Intrusion Detection Dataset |
| **Source** | Cyber Range Lab, University of New South Wales, Australia |
| **Total Records** | 175,341 (training) + 82,332 (testing) |
| **Features** | 31 numeric + categorical |
| **Attack Categories** | 10 |
| **Format** | Parquet, CSV |
| **Release Date** | 2015 |

### 1.1 Why UNSW-NB15?

UNSW-NB15 is chosen because:

1. **Realistic**: Created in a controlled lab environment with real network traffic
2. **Comprehensive**: Covers 10 different attack categories including modern threats
3. **Labeled**: Every record has ground truth labels for supervised learning
4. **Standard**: Widely used in network security research (benchmark dataset)
5. **Balanced**: Better class distribution than older datasets like KDD Cup 1999

---

## 2. Dataset Source

### 2.1 Origin and Creation

UNSW-NB15 was created by the **Cyber Range Lab** at the **University of New South Wales (UNSW)** in Sydney, Australia.

The dataset was created to address limitations of older intrusion detection datasets:
- **KDD Cup 1999** was too old and didn't reflect modern attacks
- **Labeled synthetic data** didn't capture real network behaviors
- **Insufficient diversity** in attack types

### 2.2 Data Collection Methodology

The data was collected using:

1. **Virtual Networks**: Created realistic corporate network topologies
2. **Normal Traffic**: Generated legitimate user behavior (browsing, email, file transfers)
3. **Attack Traffic**: Launched 10 different types of attacks
4. **Packet Capture**: Used tcpdump and Bro IDS to capture full packet headers
5. **Labeling**: Each connection labeled by the attack type used to generate it

### 2.3 Where to Download

```bash
# Official source: Kaggle
kaggle datasets download -d dhoogla/unswnb15

# Alternative: UNSW Cyber Range Lab
# https://www.unswaustralia.com/unsw-nb15-dataset.html
```

The dataset is also included in this project in the `data/` directory in Parquet format:
- `data/UNSW_NB15_training-set.parquet`
- `data/UNSW_NB15_testing-set.parquet`

---

## 3. Feature Reference

This section provides detailed descriptions of all 31 features in the UNSW-NB15 dataset.

### 3.1 Feature Categories

The features are organized into groups:

| Category | Features | Description |
|---|---|---|
| **Basic Flow** | Duration, Bytes, Packets | Fundamental traffic measurements |
| **TCP Metrics** | tcprtt, synack, ackdat | TCP handshake and timing |
| **Connection Counts** | ct_* | Various connection statistics |
| **Rate Metrics** | rate, srate, drate | Traffic rates |
| **Time Statistics** | t*, tt* | Time-based features |
| **Port Features** | dport, sport | Port numbers |

### 3.2 Complete Feature List

| # | Feature Name | Type | Description | Typical Values | Detection Use |
|---|---|---|---|---|---|
| 1 | **dur** | float | Duration of the flow (seconds) | 0-3600s | Long-running DoS; short probes |
| 2 | **sbytes** | int | Bytes sent from source to destination | 0-10^9 | Data exfiltration (high src) |
| 3 | **dbytes** | int | Bytes sent from destination to source | 0-10^9 | Data download; response size |
| 4 | **sttl** | int | Time-to-live value set by source | 1-255 | OS fingerprinting |
| 5 | **dttl** | int | Time-to-live value set by destination | 1-255 | OS fingerprinting |
| 6 | **sloss** | int | Packets lost from source | 0-1000 | Network quality issues |
| 7 | **dloss** | int | Packets lost from destination | 0-1000 | Network quality issues |
| 8 | | | | | | | | | 8 | **spkts** | int | Packets sent from source | 0-10^6 | DoS (high); ACK probes |
| 9 | **dpkts** | int | Packets sent from destination | 0-10^6 | Response intensity |
| 10 | **swin** | int | Source TCP window size | 0-65535 | TCP fingerprint |
| 11 | **dwin** | int | Destination TCP window size | 0-65535 | TCP fingerprint |
| 12 | **stcpb** | int | Source TCP base sequence number | 0-10^9 | TCP fingerprint |
| 13 | **dtcpb** | int | Destination TCP base sequence number | 0-10^9 | TCP fingerprint |
| 14 | **smeansz** | int | Mean packet size from source | 0-1500 | Attack signature |
| 15 | **dmeansz** | int | Mean packet size from destination | 0-1500 | Attack signature |
| 16 | **trans_depth** | int | HTTP transaction depth | 0-10 | Web attack indicator |
| 17 | **res_bdy_len** | int | HTTP response body length (bytes) | 0-10^6 | Data exfiltration |
| 18 | **sjit** | float | Source jitter (ms) | 0-1000 | Anomaly indicator |
| 19 | **djit** | float | Destination jitter (ms) | 0-1000 | Anomaly indicator |
| 20 | **sintpkt** | float | Source interpacket arrival time (ms) | 0-1000 | Timing attack |
| 21 | **dintpkt** | float | Destination interpacket arrival time (ms) | 0-1000 | Timing attack |
| 22 | **tcprtt** | float | TCP round-trip time (ms) | 0-5000 | Latency; SYN flood |
| 23 | **synack** | float | SYN-ACK delay (ms) | 0-5000 | TCP handshake anomaly |
| 24 | **ackdat** | float | ACK delay (ms) | 0-5000 | Data transfer pattern |
| 25 | **ct_state_ttl** | int | Connection count by state and TTL | 0-1000 | Connection scanning |
| 26 | **ct_flw_http_mthd** | int | Flow methods in HTTP | 0-10 | HTTP anomaly |
| 27 | **is_sm_ips_ports** | binary | Source/dest ports < 1024 | 0/1 | Port scan indicator |
| 28 | **ct_ Src_dport_ltm** | int | Source port counts | 0-100 | Port scanning |
| 29 | **ct_dst_sport_ltm** | int | Destination port counts | 0-100 | Port scanning |
| 30 | **ct_dst_srv_ltm** | int | Destination service counts | 0-100 | Service scanning |
| 31 | **ct_srv_dst** | int | Server connections count | 0-1000 | DDoS indicator |

### 3.3 Feature Descriptions in Detail

#### A. Basic Flow Features (Features 1-7)

**dur** (Duration)
```
Description: Length of the flow in seconds
Why it matters: 
  - DoS attacks often have very short durations (seconds)
  - Long-running scans might run for hours
  - Normal browsing is typically medium duration
Example:
  - Port scan: 0.01s (very short)
  - Normal browse: 5-60s (medium)
  - File download: 30-300s (long)
```

**sbytes / dbytes** (Source/Destination Bytes)
```
Description: Total bytes transferred in each direction
Why it matters:
  - Data exfiltration: High sbytes (source → attacker)
  - Downloads: High dbytes (server → client)
  - DoS: Often zero or very low (no actual data)
Example:
  - Normal HTTP: 5KB src → 50KB dst
  - Data theft: 1MB src → 0 dst
  - DoS: 1KB src → 0 dst
```

**sttl / dttl** (Time-to-Live)
```
Description: IP time-to-live value (1-255)
Why it matters:
  - Different operating systems set different default TTLs
  - Can fingerprint the OS
  - Anomalous TTLs might indicate manipulation
Example:
  - Linux: 64
  - Windows: 128
  - Network devices: 255
```

**sloss / dloss** (Lost Packets)
```
Description: Packets retransmitted or lost
Why it matters:
  - Network congestion
  - DoS attack impact
  - Unstable connections
```

#### B. TCP Features (Features 22-24)

**tcprtt** (TCP Round-Trip Time)
```
Description: Time from SYN to SYN-ACK in milliseconds
Why it matters:
  - Network latency indicator
  - SYN flood: Often very low (no real connection)
  - Normal TCP: 1-500ms typical
Example:
  - Local network: 1-5ms
  - Internet: 20-200ms
  - SYN flood: 0ms
```

**synack** (SYN-ACK Delay)
```
Description: Time from SYN-ACK to ACK in milliseconds
Why it matters:
  - Three-way handshake completion
  - Legitimate vs spoofed connections
  - Slowloris: Never completes (very high)
```

**ackdat** (ACK Data Time)
```
Description: Time from first ACK to first data packet
Why it matters:
  - Data transfer pattern
  - ACK scanning: sends only ACKs, no data
```

#### C. Rate Features (Features 18-21, 27)

**rate** (Packets per second)
```
Description: Overall packet rate for the flow
Why it matters:
  - DoS attacks have extremely high rates
  - Port scans have medium rates
  - Normal traffic has low rates
```

**srate / drate** (Source/Destination Rate)
```
Description: Packets per second in each direction
Why it matters:
  - DDoS: Often asymmetric
  - Normal: Generally symmetric
```

**sjit / djit** (Jitter)
```
Description: Variation in inter-packet time (ms)
Why it matters:
  - Stable connection: Low jitter
  - Attack traffic: Higher jitter
```

#### D. Count Features (Features 25-31)

**ct_state_ttl** (Connection State Count)
```
Description: Number of connections with same state and TTL
Why it matters:
  - Port scanning: Many connections with same TTL
  - Normal: Few connections per TTL
```

**ct_srv_src / ct_srv_dst** (Service Connection Counts)
```
Description: Connections to same service
Why it matters:
  - DDoS: Many connections to same service
  - Normal: Fewer connections
```

---

## 4. Attack Categories

### 4.1 Complete Attack List

UNSW-NB15 contains 10 attack categories + normal traffic:

| ID | Category | Description | MO Characteristics |
|---|---|---|---|
| 0 | **Normal** | Normal, legitimate traffic | No attack behavior |
| 1 | **Analysis** | Intrusion analysis | Port scans, sweeps |
| 2 | **Backdoor** | Covert channels, trojans | Stealthy C2, trojan traffic |
| 3 | **DoS** | Denial of Service | High volume, resource exhaustion |
| 4 | **Exploits** | Exploit kits, shellcode | Vulnerability exploitation |
| 5 | **Fuzzers** | Fuzzing, brute force | Random inputs, probing |
| 6 | **Generic** | Generic attacks | Crypto-agnostic, brute force |
| 7 | **Reconnaissance** | Reconnaissance | Information gathering |
| 8 | **Shellcode** | Shellcode attacks | Code injection |
| 9 | **Worms** | Self-replicating malware | Propagation |

### 4.2 Detailed Attack Descriptions

#### Category 1: Analysis (Reconnaissance)

```
Description: Attacks focused on information gathering
Attack Vectors:
  - Port scanning
  - Network sweeps
  - Vulnerability scanning
  
Network Signatures:
  - Many short connections
  - Low bytes per connection
  - Many different ports
  - High variety in destination IPs
  
Example: nmap -sS 192.168.1.0/24
```

#### Category 2: Backdoor

```
Description: Covert channels and trojan-style backdoors
Attack Vectors:
  - Trojans with backdoor capability
  - Covert channels
  - Rootkits
  
Network Signatures:
  - Unusual ports (not 80, 443)
  - Long-duration connections
  - Periodic "beaconing" pattern
  - Small, regular payloads
  
Example: Remote access trojan sending data every 60 seconds
```

#### Category 3: DoS (Denial of Service)

```
Description: Attacking availability by exhausting resources
Attack Vectors:
  - Volume-based DoS (flooding)
  - Protocol-based (SYN flood)
  - Application-layer
  
Network Signatures:
  - Extremely high packet rate
  - High packet counts, low bytes (overhead)
  - Same destination (targeted) or distributed
  - TCP SYN floods: No completion of handshake
  
Example: 
  - SYN flood: Many SYN packets, no SYN-ACK responses
  - UDP flood: High rate of small packets
  - HTTP flood: Many requests, no responses
```

#### Category 4: Exploits

```
Description: Active exploitation of vulnerabilities
Attack Vectors:
  - Exploit kits
  - Zero-day exploits
  - Shellcode execution
  
Network Signatures:
  - Often starts with reconnaissance
  - May include shellcode transmission
  - Abnormal payload sizes
  - Buffer overflow indicators
  
Example: Sending malformed packets to exploit buffer overflow
```

#### Category 5: Fuzzers

```
Description: Automated testing with malformed inputs
Attack Vectors:
  - Random/garbage inputs
  - Protocol fuzzing
  - Mutation-based fuzzing
  
Network Signatures:
  - Random/garbage data in payloads
  - Invalid protocol fields
  - Unexpected packet sizes
  - High variety in patterns
  
Example: Sending random data to crash a service
```

#### Category 6: Generic

```
Description: Generic brute-force or cryptographic attacks
Attack Vectors:
  - Password guessing
  - Key recovery
  - Crypto attacks
  
Network Signatures:
  - Many repeated attempts
  - Regular patterns
  - Often to login ports (22, 3389, etc.)
  
Example: Brute-forcing SSH passwords
```

#### Category 7: Reconnaissance

```
Description: Pre-attack information gathering
Attack Vectors:
  - Active reconnaissance
  - Social engineering
  - OSINT
  
Network Signatures:
  - Low volume (avoid detection)
  - Targeting multiple hosts
  - Different types of probes
  - Often combined with Analysis category
  
Example: Port scan + whois lookup + DNS enumeration
```

#### Category 8: Shellcode

```
Description: Code injection attacks
Attack Vectors:
  - Shellcode payloads
  - Code execution
  - Memory corruption
  
Network Signatures:
  - Raw binary in payloads
  - Non-printable characters
  - Specific byte sequences
  - Often follows Exploits category
  
Example: Buffer overflow leading to shellcode execution
```

#### Category 9: Worms

```
Description: Self-replicating malware
Attack Vectors:
  - Email worms
  - Network worms
  - USB worms
  
Network Signatures:
  - Self-propagation
  - Spreads to new targets
  - High replication rate
  - Similar patterns to source
  
Example: Conficker worm scanning and infecting new hosts
```

#### Category 0: Normal

```
Description: Legitimate, non-malicious traffic
Traffic Types:
  - Web browsing (HTTP/HTTPS)
  - Email (SMTP, POP3, IMAP)
  - File transfers (FTP, SFTP)
  - Remote access (SSH, RDP)
  - DNS queries
  
Network Signatures:
  - Predictable patterns
  - Reasonable durations
  - Appropriate byte counts
  - Standard ports
  - No anomalies
```

---

## 5. Data Processing Pipeline

### 5.1 Processing Steps Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Load Dataset                                       │
│         (parquet or CSV)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Identify and Extract Features                      │
│         - Drop non-numeric columns                         │
│         - Handle missing values                           │
│         - Encode categorical features                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Label Encoding                                     │
│         - Map attack categories to integers               │
│         - Normal = 0, DoS = 1, etc.                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Feature Scaling                                    │
│         - StandardScaler (z-score normalization)          │
│         - Fit on training data only                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Crime Pattern Creation                            │
│         - Group into 5-event sequences                     │
│         - Same attack category required                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Loading the Dataset

```python
# From src/dataset.py

def load_and_preprocess(data_path):
    """
    Loads parquet or CSV, cleans, encodes labels, scales features.
    """
    # Load based on file type
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    
    print(f"[dataset] Raw shape: {df.shape}")
    
    # Drop non-feature columns
    # These are identifiers, not features
    drop_cols = [
        "id", "srcip", "sport", "dstip", "dsport",
        "proto", "state", "service", "attack_cat", "is_sm_ips_ports"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    # Keep only numeric features
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Encode attack labels to integers
    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str).str.strip())
```

### 5.3 Scaling Features

```python
# Feature scaling using StandardScaler
# This normalizes each feature to mean=0, std=1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df.values)

# Example transformation:
# Before: src_bytes = [100, 500, 1000, 5000]
# After:  src_bytes = [-0.5, -0.2, 0.3, 1.4]
# (values represent standard deviations from mean)
```

---

## 6. Crime Pattern Generation

### 6.1 Why Pattern Groups?

Individual network events are not sufficient for MO detection. A single packet doesn't reveal the attacker's methodology.

**Attack sequences** (multiple events) reveal the MO:
- Port scan → Exploit → Shellcode → Backdoor install → Data exfil
- This sequence IS the Modus Operandi

### 6.2 Pattern Generation Process

```python
# From src/dataset.py

def create_crime_patterns(X, y, seq_len=5):
    """
    Groups individual network events into crime patterns.
    Each pattern = seq_len consecutive events of same attack type.
    
    Args:
        X: Scaled feature matrix
        y: Labels
        seq_len: Events per pattern (default: 5)
    
    Returns:
        patterns: Array of shape (num_patterns, seq_len, features)
        labels: Array of labels
    """
    patterns, labels = [], []
    
    # Group by attack category
    for cls in np.unique(y):
        # Get all events of this class
        idx = np.where(y == cls)[0]
        data = X[idx]
        
        # Create overlapping windows
        for i in range(0, len(data) - seq_len, 2):  # stride=2 for diversity
            pattern = data[i : i + seq_len]  # 5 consecutive events
            patterns.append(pattern)
            labels.append(cls)
    
    return np.array(patterns), np.array(labels)
```

### 6.3 Visual Example

```
Original events (DoS class):
  Event 1: [0.1, 0.8, 0.2, ...]  DoS
  Event 2: [0.2, 0.7, 0.1, ...]  DoS
  Event 3: [0.1, 0.9, 0.2, ...]  DoS
  Event 4: [0.2, 0.8, 0.1, ...]  DoS
  Event 5: [0.1, 0.7, 0.2, ...]  DoS
  Event 6: [0.2, 0.9, 0.1, ...]  DoS
  Event 7: [0.1, 0.8, 0.2, ...]  DoS
  Event 8: [0.2, 0.7, 0.1, ...]  DoS
  Event 9: [0.1, 0.9, 0.2, ...]  DoS
  Event 10: [0.2, 0.8, 0.1, ...] DoS

Generated patterns (seq_len=5, stride=2):
  Pattern 1: [E1, E2, E3, E4, E5]  ← Events 1-5
  Pattern 2: [E3, E4, E5, E6, E7]  ← Events 3-7
  Pattern 3: [E5, E6, E7, E8, E9]  ← Events 5-9
  
Why stride=2? Creates diverse patterns while maintaining same label
```

### 6.4 Pattern Attributes

| Attribute | Value | Description |
|---|---|---|
| Events per pattern | 5 | Captures multi-stage attacks |
| Stride | 2 | Creates overlapping patterns |
| Min events for pattern | 5 | Minimum threshold |
| Label | Same as source events | Attack category |

---

## 7. Pair Generation for Training

### 7.1 Why Pairs?

Siamese networks are trained on **pairs** of patterns:
- **Positive pair**: Two patterns of SAME MO (same attack category)
- **Negative pair**: Two patterns of DIFFERENT MO (different categories)

### 7.2 Pair Generation Process

```python
# From src/dataset.py

def create_pairs(patterns, labels, pairs_per_class=300):
    """
    Creates balanced positive and negative pairs for Siamese training.
    
    Args:
        patterns: Crime pattern array
        labels: Pattern labels
        pairs_per_class: Number of pairs to generate per class
    
    Returns:
        pairs: List of (pattern_a, pattern_b) tuples
        targets: List of labels (1 = same MO, 0 = different MO)
    """
    pairs, targets = [], []
    
    for cls in np.unique(labels):
        same_idx = np.where(labels == cls)[0]    # Same class indices
        diff_idx = np.where(labels != cls)[0]    # Different class indices
        n_pos = min(len(same_idx) - 1, pairs_per_class)
        
        # Positive pairs — SAME MO (same attack class)
        for i in range(n_pos):
            j = np.random.randint(0, len(same_idx))
            while j == i:  # Don't pair with self
                j = np.random.randint(0, len(same_idx))
            pairs.append((patterns[same_idx[i]], patterns[same_idx[j]]))
            targets.append(1)  # Same MO = 1
        
        # Negative pairs — DIFFERENT MO (different class)
        for i in range(n_pos):
            j = np.random.choice(diff_idx)
            pairs.append((patterns[same_idx[i % len(same_idx)]], patterns[j]))
            targets.append(0)  # Different MO = 0
    
    return pairs, targets
```

### 7.3 Visual Example

```
Positive Pair (SAME MO):
┌─────────────────────────────────────────────────┐
│ Pattern A (DoS)    Pattern B (DoS)              │
│ [e1,e2,e3,e4,e5] = [e1,e2,e3,e4,e5]             │
│                                                 │
│ Same attack category → Label = 1.0              │
└─────────────────────────────────────────────────┘

Negative Pair (DIFFERENT MO):
┌─────────────────────────────────────────────────┐
│ Pattern A (DoS)    Pattern B (Backdoor)         │
│ [e1,e2,e3,e4,e5] ≠ [e1,e2,e3,e4,e5]             │
│                                                 │
│ Different category → Label = 0.0               │
└─────────────────────────────────────────────────┘
```

### 7.4 Balanced Dataset

| Metric | Value |
|---|---|
| Total pairs | ~5,400 |
| Positive (same MO) | ~2,700 |
| Negative (different MO) | ~2,700 |
| Balance | 50/50 |

---

## 8. Sample Data Statistics

### 8.1 Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Normal | ~56,000 | 32.0% |
| Generic | ~40,000 | 22.8% |
| Exploits | ~33,000 | 18.8% |
| Fuzzers | ~18,000 | 10.3% |
| DoS | ~13,000 | 7.4% |
| Reconnaissance | ~10,000 | 5.7% |
| Backdoor | ~2,000 | 1.1% |
| Shellcode | ~1,300 | 0.7% |
| Worms | ~1,000 | 0.6% |
| Analysis | ~1,000 | 0.6% |

### 8.2 Feature Statistics

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| dur | 3.5 | 45.2 | 0 | 3600 |
| sbytes | 892 | 45,231 | 0 | 2.1M |
| dbytes | 1,204 | 89,432 | 0 | 4.4M |
| tcprtt | 0.06 | 0.89 | 0 | 15.2 |
| rate | 124 | 2,341 | 0 | 50,000 |

---

## Next Steps

- **[Processing Pipeline](pipeline.md)**: How data flows from input to prediction
- **[API Reference](api.md)**: Flask API details
- **[Frontend Guide](frontend.md)**: React frontend components

---

<div align="center">
  <sub>Part of the Intelligent Crime Pattern Recognition System</sub>
</div>