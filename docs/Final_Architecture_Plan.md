# Design Blueprint: An N-Modal Siamese Network for MO Comparison

**Document Status:** Final Approved Plan
**Version:** 1.0
**Date:** 2026-04-14

---

## 1. Vision & Goal

This document outlines the final architecture for the **Intelligent Crime Pattern Recognition System**. The system's primary goal is to determine the **Modus Operandi (MO) similarity** between two separate incidents.

The architecture is designed to be **multi-modal**, meaning it can ingest, process, and fuse evidence from multiple, distinct data sources to form a holistic understanding of an incident.

The initial implementation will support three core data modalities:
1.  **Network Logs** (Sequential Text/CSV Data)
2.  **Visual Evidence** (Hybrid of Timestamped Image Sequences and Unordered Image Sets)
3.  **Executable Binaries** (`.exe` files treated as images)

The system is explicitly designed to be **modular and extensible**, allowing for the straightforward addition of new data modalities in the future (e.g., audio, text reports).

---

## 2. High-Level Architecture

The system is built on a **Siamese Network** paradigm. Two identical "twin" networks—called **Multimodal MO Extractors**—process two separate incidents. The resulting output vectors are then compared to produce a similarity score.

The core innovation lies within the `Multimodal MO Extractor`. It is a modular network composed of specialized feature extractors for each data type, whose outputs are combined in a final fusion step.

**High-Level Diagram:**

```
==================================================================================================
                                    SYSTEM INPUT
==================================================================================================
        User provides all available evidence for two separate incidents.
        (Any data type can be missing for a given incident).

        INCIDENT "A"                                      INCIDENT "B"
        ┌───────────────────────────┐                       ┌───────────────────────────┐
        │   - Network Log File A    │                       │   - Network Log File B    │
        │   - [Image A1, A2...]     │                       │   - [Image B1, B2...]     │
        │   - Binary File A (.exe)  │                       │   - Binary File B (.exe)  │
        └───────────────────────────┘                       └───────────────────────────┘
                  │                                                 │
                  ▼                                                 ▼
==================================================================================================
                 TWIN 1: MO Extractor                TWIN 2: MO Extractor (Identical to Twin 1)
==================================================================================================
                  │                                                 │
┌─────────────────┴─────────────────┐               ┌─────────────────┴─────────────────┐
│   ┌───────────────────────────┐   │               │   ┌───────────────────────────┐   │
│   │   Log Extractor           │   │               │   │   Log Extractor           │   │
│   │   (CNN + Bi-LSTM)         ├───┼───────────────► │   │   (CNN + Bi-LSTM)         │   │
│   └───────────────────────────┘   │               │   └───────────────────────────┘   │
│   ┌───────────────────────────┐   │               │   ┌───────────────────────────┐   │
│   │ Hybrid Image Extractor    │   │               │   │ Hybrid Image Extractor    │   │
│   │ (Handles Timed & Static)  ├───┼───────────────► │   │ (Handles Timed & Static)  │   │
│   └───────────────────────────┘   │               │   └───────────────────────────┘   │
│   ┌───────────────────────────┐   │               │   ┌───────────────────────────┐   │
│   │   Binary Extractor        │   │               │   │   Binary Extractor        │   │
│   │   (CNN on Binary-Image)   ├───┼───────────────► │   │   (CNN on Binary-Image)   │   │
│   └───────────────────────────┘   │               │   └───────────────────────────┘   │
│                 │                 │               │                 │                 │
└─────────────────│─────────────────┘               └─────────────────│─────────────────┘
                  │                                                 │
                  ▼ (Concatenate all embeddings)                    ▼ (Concatenate all embeddings)
                  │                                                 │
          ┌───────┴────────┐                                  ┌───────┴────────┐
          │  Fusion Layer  │                                  │  Fusion Layer  │
          └───────┬────────┘                                  └───────┬────────┘
                  │                                                 │
                  ▼                                                 ▼
        ┌───────────────────┐                               ┌───────────────────┐
        │ Multimodal MO     │                               │ Multimodal MO     │
        │ Vector (MM-MO_A)  │                               │ Vector (MM-MO_B)  │
        └───────────────────┘                               └───────────────────┘
                  │                                                 │
                  └───────────────────────┬─────────────────────────┘
                                          │
==================================================================================================
                                    COMPARISON
==================================================================================================
                                          │
                                          ▼
                               ┌───────────────────┐
                               │  Compute Distance │
                               │  |MM-MO_A-MM-MO_B|  │
                               └────────┬──────────┘
                                        │
                                        ▼
                                ┌──────────────────┐
                                │ Final Similarity │
                                │   Score (0-1)    │
                                └──────────────────┘
```

---

## 3. Detailed Extractor Architectures

### 3.1 `LogExtractor` (For Network Logs)
*   **Purpose:** To model the temporal sequence of network events.
*   **Input:** A sequence of log entries (e.g., shape `[batch, 5, 31]`).
*   **Architecture:** `CNN -> Bi-LSTM`.
    1.  **1D CNN:** A 1D Convolutional Neural Network first processes each event in the sequence to extract a compact "fingerprint" or feature vector. This captures local patterns within an event's features.
    2.  **Bi-LSTM:** The sequence of these fingerprints is then fed into a Bidirectional LSTM, which models the forward and backward temporal dependencies between the events.
*   **Output:** A fixed-size `log_embedding` vector.

### 3.2 `HybridImageExtractor` (For Visual Evidence)
*   **Purpose:** To intelligently process a mix of timestamped (sequential) and non-timestamped (static) images from a single incident.
*   **Input:** Two sets of images for an incident: a sorted list of timestamped images and an unordered set of static images.
*   **Internal Architecture:** This module contains two sub-branches.
    1.  **Sequential Branch:**
        *   **Technology:** `CNN + Bi-LSTM`.
        *   **Process:** The timestamped images are processed by a pre-trained ResNet-18 (as the CNN) to get per-frame features, which are then fed into a Bi-LSTM to capture the temporal story.
        *   **Output:** `sequential_embedding`.
    2.  **Static Branch:**
        *   **Technology:** `CNN + Max-Pooling`.
        *   **Process:** The non-timestamped images are processed by the same pre-trained ResNet-18. The resulting feature vectors are then combined into a single vector via an element-wise Max-Pooling operation.
        *   **Output:** `static_embedding`.
*   **Fusion:** The `sequential_embedding` and `static_embedding` are concatenated and passed through a small fusion layer.
*   **Final Output:** A single, unified `image_embedding` vector.

### 3.3 `BinaryExtractor` (For `.exe` Files)
*   **Purpose:** To extract features from executable files.
*   **Input:** A grayscale image generated from the bytes of the `.exe` file.
*   **Architecture:** `CNN`.
    1.  **Preprocessing:** The raw binary file is read as a byte stream and reshaped into a 2D grid (an image).
    2.  **CNN:** This "binary image" is processed by a pre-trained ResNet-18 network. The visual textures created by the binary's structure (`.text`, `.data` sections) are learned by the CNN.
*   **Output:** A fixed-size `binary_embedding` vector.

---

## 4. Fusion and Comparison

### 4.1 Late Fusion
The system employs a **late fusion** strategy. Each modality is processed completely by its specialized extractor before the final embeddings are combined.

1.  **Concatenation:** The `log_embedding`, `image_embedding`, and `binary_embedding` are concatenated into one long vector.
2.  **Fusion Layer:** This vector is passed through a final `nn.Linear` layer which learns the optimal way to combine the features from all modalities, producing the final **Multimodal MO (MM-MO) Vector**.

### 4.2 Handling Missing Data
The fusion approach is robust to missing data. If an incident lacks a certain data type (e.g., no binary file), a zero-vector will be used as the input for that extractor's branch. The model will learn during training to effectively ignore these zero-embeddings and make a decision based on the evidence it does have.

### 4.3 Siamese Comparison
The final MM-MO vectors from the two incidents (`MM-MO_A` and `MM-MO_B`) are compared using a distance metric (e.g., absolute difference), which is then passed through a final dense layer with a sigmoid activation to produce the similarity score (0.0 to 1.0).

---

## 5. Implementation & Training Strategy

1.  **Dependencies:** Add `torchvision` and `Pillow` to `requirements.txt`.
2.  **Dataset Partitioning:** A preprocessing script will be created to scan the dataset and create a manifest file that tags incidents as having timestamped images, static images, or both.
3.  **Model Implementation:** The `src/model.py` file will be refactored to include the new `HybridImageExtractor` and `BinaryExtractor` modules and to update the main `SiameseCrimeMatcher` with the fusion logic.
4.  **Data Loader:** The `CrimePairDataset` in `src/dataset.py` will be updated to handle the hybrid image loading and padding for all modalities.
5.  **API & Frontend:** The `app.py` and frontend components will be updated to handle the multi-file upload workflow for logs, multiple images, and binaries.
6.  **Training:** A single model will be trained on the entire dataset. Its hybrid structure will allow it to learn from all types of data combinations simultaneously.
