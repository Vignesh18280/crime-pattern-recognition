# API Reference
### Flask REST API Documentation

---

This document provides complete reference for the Flask API that powers the crime pattern recognition system.

---

## Table of Contents

1. [API Overview](#1-api-overview)
2. [Server Setup](#2-server-setup)
3. [Endpoints](#3-endpoints)
4. [Request/Response Formats](#4-requestresponse-formats)
5. [Error Handling](#5-error-handling)
6. [Integration with Frontend](#6-integration-with-frontend)
7. [Example Usage](#7-example-usage)

---

## 1. API Overview

### 1.1 Architecture

```
┌─────────────────────────────────────────┐
│         Flask Server (Port 8080)           │
│                                         │
│  /api/upload   → Upload incident files     │
│  /api/predict → Compare incidents        │
│  /api/ready  → Health check           │
└─────────────────────────────────────────┘
            │
            ▼ (HTTP JSON)
┌─────────────────────────────────────────┐
│       React Frontend (Port 5173)           │
│              or                         │
│       Direct HTTP Client                │
└─────────────────────────────────────────┘
```

### 1.2 Base URL

```
http://localhost:8080/api
```

---

## 2. Server Setup

### 2.1 Starting the Server

```bash
python3 app.py
```

### 2.2 Expected Output

```
Loading model and data...
Checkpoint loaded successfully.
Dataset loaded.
Scaler loaded.
Server ready.
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:8080
```

### 2.3 Configuration

```python
# From app.py

# Server configuration
MODEL_PATH = "results/best_model.pth"
DATA_PATH = "data/UNSW_NB15_testing-set.parquet"
UPLOAD_FOLDER = 'data/temp_uploads'
SCALER_PATH = 'scaler.pkl'
DEVICE = torch.device("cpu")

ALLOWED_EXTENSIONS = {'csv', 'parquet'}
```

---

## 3. Endpoints

### 3.1 Health Check

#### `/api/ready`

Check if the server and model are ready.

**Method:** `GET`

**Request:**
```
GET http://localhost:8080/api/ready
```

**Response (Success):**
```json
{
  "ready": true,
  "message": "Server ready"
}
```

**Response (Not Ready):**
```json
{
  "ready": false,
  "message": "Model not loaded"
}
```

---

### 3.2 File Upload

#### `/api/upload`

Upload an incident log file.

**Method:** `POST`

**Content-Type:** `multipart/form-data`

**Request:**
```
POST http://localhost:8080/api/upload
Content-Type: multipart/form-data

Body:
  file: <file content>
```

**Response (Success):**
```json
{
  "filename": "abc123_incident_dos_a.csv"
}
```

**Response (Error):**
```json
{
  "error": "No file provided"
}
```

---

### 3.3 Predict

#### `/api/predict`

Compare two incident logs and determine if they share the same Modus Operandi.

**Method:** `POST`

**Content-Type:** `application/json`

**Request:**
```json
POST http://localhost:8080/api/predict
Content-Type: application/json

{
  "uploadedFile1": "abc123_incident_dos_a.csv",
  "uploadedFile2": "xyz789_incident_dos_b.csv"
}
```

**Response (Success):**
```json
{
  "verdict": "SAME MO",
  "similarityScore": 0.95,
  "incident1": "incident_dos_a.csv",
  "incident2": "incident_dos_b.csv",
  "best_patterns": {
    "pattern1_index": 4,
    "pattern2_index": 2,
    "pattern1_data": [0.12, 0.85, -0.33, 0.45, -0.12, 0.67, 0.23, -0.45, 0.89, 0.01],
    "pattern2_data": [0.10, 0.82, -0.30, 0.42, -0.10, 0.65, 0.21, -0.42, 0.87, 0.00]
  },
  "explanation": [
    {"feature": "sbytes", "importance": 0.35},
    {"feature": "tcprtt", "importance": -0.22},
    {"feature": "synack", "importance": 0.15}
  ]
}
```

---

## 4. Request/Response Formats

### 4.1 Complete Request Format

```json
{
  "uploadedFile1": "string",
  "uploadedFile2": "string"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `uploadedFile1` | string | Yes | Filename returned from `/api/upload` |
| `uploadedFile2` | string | Yes | Filename returned from `/api/upload` |

### 4.2 Complete Response Format

```json
{
  "verdict": "string",
  "similarityScore": "number",
  "incident1": "string",
  "incident2": "string",
  "best_patterns": {
    "pattern1_index": "number",
    "pattern2_index": "number",
    "pattern1_data": ["number"],
    "pattern2_data": ["number"]
  },
  "explanation": [
    {
      "feature": "string",
      "importance": "number"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `verdict` | string | "SAME MO" or "DIFFERENT MO" |
| `similarityScore` | number | 0.0-1.0, similarity percentage |
| `incident1` | string | First incident filename |
| `incident2` | string | Second incident filename |
| `best_patterns.pattern1_index` | number | Index of best matching pattern from incident 1 |
| `best_patterns.pattern2_index` | number | Index of best matching pattern from incident 2 |
| `best_patterns.pattern1_data` | array | Feature values from best pattern 1 |
| `best_patterns.pattern2_data` | array | Feature values from best pattern 2 |
| `explanation` | array | Features driving the similarity decision |

### 4.3 Field Descriptions

**Verdict Values:**
- `"SAME MO"`: Similarity score >= 0.5, likely same attacker
- `"DIFFERENT MO"`: Similarity score < 0.5, likely unrelated

**Similarity Score:**
- Range: 0.0 to 1.0
- 0.95 = 95% similar
- 0.50 = 50% similar (threshold)

---

## 5. Error Handling

### 5.1 Error Response Format

```json
{
  "error": "Error message describing the issue"
}
```

### 5.2 Common Errors

| Error | Status Code | Cause |
|---|---|---|
| "No file provided" | 400 | Missing file in upload |
| "Invalid file type" | 400 | File is not CSV or Parquet |
| "File not found: {filename}" | 400 | Uploaded file doesn't exist |
| "Please upload two incident logs" | 400 | Missing one or both files |
| "Could not process logs" | 400 | Not enough suspicious data |
| "Server not ready" | 500 | Model failed to load |

### 5.3 Handling Errors

```python
try:
    response = fetch('http://localhost:8080/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            uploadedFile1: file1,
            uploadedFile2: file2
        })
    })
    
    if (!response.ok) {
        const error = await response.json()
        alert(error.error)
        return
    }
    
    const result = await response.json()
    console.log(result.verdict, result.similarityScore)
    
} catch (err) {
    console.error(err)
}
```

---

## 6. Integration with Frontend

### 6.1 Frontend API Client

```typescript
// From frontend/src/lib/api.ts

const API_BASE_URL = "http://localhost:8080/api";

export async function uploadFile(file: File): Promise<{ filename: string }> {
    const formData = new FormData();
    formData.append("file", file);
    
    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "File upload failed");
    }
    
    return response.json();
}

export async function predict(file1: string, file2: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            uploadedFile1: file1,
            uploadedFile2: file2,
        }),
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
    }
    
    return response.json();
}
```

### 6.2 Using in React Component

```typescript
// From frontend/src/components/FileUploader.tsx

const handleAnalyze = async () => {
    if (!file1 || !file2) {
        alert('Please upload both incident log files.');
        return;
    }
    
    try {
        // Upload both files
        const [upload1, upload2] = await Promise.all([
            uploadFile(file1),
            uploadFile(file2),
        ]);
        
        // Get prediction
        const result = await predict(upload1.filename, upload2.filename);
        
        // Display result
        console.log(result.verdict)      // "SAME MO" or "DIFFERENT MO"
        console.log(result.similarityScore) // 0.95
        console.log(result.explanation)      // feature importance
        
    } catch (error) {
        console.error(error);
        alert(error.message);
    }
};
```

---

## 7. Example Usage

### 7.1 Using cURL

```bash
# Step 1: Upload first incident file
curl -X POST -F "file=@data/sample_incidents/incident_dos_a.csv" \
    http://localhost:8080/api/upload

# Response: {"filename": "uuid1_incident_dos_a.csv"}


# Step 2: Upload second incident file
curl -X POST -F "file=@data/sample_incidents/incident_dos_b.csv" \
    http://localhost:8080/api/upload

# Response: {"filename": "uuid2_incident_dos_b.csv"}


# Step 3: Get prediction
curl -X POST -H "Content-Type: application/json" \
    -d '{"uploadedFile1": "uuid1_incident_dos_a.csv", "uploadedFile2": "uuid2_incident_dos_b.csv"}' \
    http://localhost:8080/api/predict

# Response:
# {
#   "verdict": "SAME MO",
#   "similarityScore": 0.95,
#   ...
# }
```

### 7.2 Using Python

```python
import requests

BASE_URL = "http://localhost:8080/api"

# Upload file 1
with open("incident_dos_a.csv", "rb") as f:
    response1 = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
file1_name = response1.json()["filename"]

# Upload file 2
with open("incident_dos_b.csv", "rb") as f:
    response2 = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
file2_name = response2.json()["filename"]

# Get prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "uploadedFile1": file1_name,
        "uploadedFile2": file2_name
    }
)

result = response.json()
print(f"Verdict: {result['verdict']}")
print(f"Score: {result['similarityScore']}")
```

### 7.3 Using JavaScript

```javascript
const BASE_URL = "http://localhost:8080/api";

// Upload files
const formData1 = new FormData();
formData1.append("file", file1Input.files[0]);

const formData2 = new FormData();
formData2.append("file", file2Input.files[0]);

const [response1, response2] = await Promise.all([
    fetch(`${BASE_URL}/upload`, { method: "POST", body: formData1 }),
    fetch(`${BASE_URL}/upload`, { method: "POST", body: formData2 })
]);

const [result1, result2] = await Promise.all([
    response1.json(),
    response2.json()
]);

// Get prediction
const prediction = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        uploadedFile1: result1.filename,
        uploadedFile2: result2.filename
    })
});

const result = await prediction.json();
console.log(result.verdict, result.similarityScore);
```

---

## Next Steps

- **[Architecture Deep Dive](architecture.md)**: CNN, Bi-LSTM, Siamese details
- **[Dataset Reference](dataset.md)**: UNSW-NB15 features, attacks
- **[Frontend Guide](frontend.md)**: React components

---

<div align="center">
  <sub>Part of the Intelligent Crime Pattern Recognition System</sub>
</div>