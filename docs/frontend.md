# Frontend Guide
### React Frontend Components and Visualizations

---

This document provides a complete guide to the React frontend — its components, visualizations, and how it displays analysis results.

---

## Table of Contents

1. [Frontend Overview](#1-frontend-overview)
2. [Project Structure](#2-project-structure)
3. [Components](#3-components)
4. [Visualizations](#4-visualizations)
5. [Styling](#5-styling)
6. [API Integration](#6-api-integration)
7. [Running the Frontend](#7-running-the-frontend)

---

## 1. Frontend Overview

### 1.1 Technology Stack

| Technology | Purpose |
|---|---|
| React 18 | UI Framework |
| Vite | Build tool and dev server |
| Recharts | Charting library (bar, line charts) |
| React Dropzone | File upload component |
| Lucide React | Icons |
| CSS Modules | Component-level styling |

### 1.2 Features

- **File Upload**: Drag-and-drop incident log files
- **Analysis Display**: Verdict and similarity score
- **Feature Charts**: Visual comparison of features
- **Trend Analysis**: Line charts showing feature patterns
- **Data Tables**: Detailed feature-by-feature comparison

---

## 2. Project Structure

### 2.1 Directory Layout

```
frontend/
├── src/
│   ├── components/
│   │   ├── FileUploader.tsx      # File upload component
│   │   ├── FileUploader.module.css
│   │   ├── ResultDisplay.tsx     # Main result display
│   │   ├── ResultDisplay.module.css
│   │   ├── LandingPage.tsx       # Landing/welcome page
│   │   ├── LandingPage.module.css
│   │   └── common/
│   │       ├── Button.tsx
│   │       ├── Button.module.css
│   │       ├── Card.tsx
│   │       └── Card.module.css
│   ├── lib/
│   │   └── api.ts              # API client
│   ├── App.tsx                 # Main app
│   ├── App.module.css
│   ├── main.tsx                # Entry point
│   └── index.css               # Global styles
├── public/
│   └── favicon.svg
├── package.json
├── vite.config.ts
└── tsconfig.json
```

### 2.2 Key Files

| File | Purpose |
|---|---|
| `App.tsx` | Main app, routing, state management |
| `FileUploader.tsx` | Handles file upload and triggers analysis |
| `ResultDisplay.tsx` | Displays analysis results |
| `api.ts` | API client functions |
| `index.css` | Global CSS variables and styles |

---

## 3. Components

### 3.1 App (Root Component)

```typescript
// From frontend/src/App.tsx

function App() {
  const [view, setView] = useState<'landing' | 'app'>('landing');

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Crime Pattern Recognition</h1>
      </header>
      
      <main>
        {view === 'landing' ? (
          <LandingPage onStart={() => setView('app')} />
        ) : (
          <FileUploaderWithResults />
        )}
      </main>
    </div>
  );
}
```

### 3.2 FileUploader

The main upload component that handles file selection and triggers analysis.

```typescript
// From frontend/src/components/FileUploader.tsx

interface FileUploaderProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (result: any) => void;
}

// Single file uploader
const SingleUploader = ({ file, setFile, title }: { 
  file: File | null, 
  setFile: (file: File | null) => void, 
  title: string 
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, [setFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/parquet': ['.parquet'] },
    multiple: false,
  });

  return (
    <div {...getRootProps()} className={dropzoneClass}>
      <input {...getInputProps()} />
      {file ? <p>{file.name}</p> : <p>Drag & drop or click to select</p>}
    </div>
  );
};

// Main uploader with analyze button
export const FileUploader: React.FC<FileUploaderProps> = ({ 
  onAnalysisStart, 
  onAnalysisComplete 
}) => {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!file1 || !file2) {
      alert('Please upload both files.');
      return;
    }

    setIsLoading(true);
    onAnalysisStart();
    
    try {
      // Upload both files
      const [upload1, upload2] = await Promise.all([
        uploadFile(file1),
        uploadFile(file2),
      ]);

      // Get prediction
      const result = await predict(upload1.filename, upload2.filename);
      onAnalysisComplete(result);

    } catch (error) {
      console.error(error);
      alert('Error during analysis');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <div className={styles.grid}>
        <SingleUploader file={file1} setFile={setFile1} title="Incident Log 1" />
        <SingleUploader file={file2} setFile={setFile2} title="Incident Log 2" />
      </div>
      <button onClick={handleAnalyze} disabled={isLoading}>
        {isLoading ? 'Analyzing...' : 'Analyze Incidents'}
      </button>
    </div>
  );
};
```

### 3.3 ResultDisplay

Displays the complete analysis results including verdict, charts, and data.

```typescript
// From frontend/src/components/ResultDisplay.tsx

interface ResultDisplayProps {
  result: {
    verdict: string;
    similarityScore: number;
    incident1: string;
    incident2: string;
    best_patterns: {
      pattern1_index: number;
      pattern2_index: number;
      pattern1_data: number[];
      pattern2_data: number[];
    };
    explanation: { feature: string; importance: number }[];
  };
};

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  return (
    <div className={styles.card}>
      {/* Header */}
      <div className={styles.header}>
        <h2>Analysis Complete</h2>
        <p>Comparing {result.incident1} vs {result.incident2}</p>
      </div>
      
      {/* Verdict */}
      <Verdict verdict={result.verdict} score={result.similarityScore} />
      
      {/* Feature Chart */}
      <ExplanationChart data={result.explanation || []} />
      
      {/* Detailed Analysis */}
      <DetailedAnalysis result={result} />
    </div>
  );
};
```

---

## 4. Visualizations

### 4.1 Verdict Display

Shows the final verdict (SAME MO / DIFFERENT MO) with similarity score.

```typescript
const Verdict = ({ verdict, score }: { verdict: string, score: number }) => {
  const isSame = verdict === 'SAME MO';
  
  return (
    <div className={styles.verdictCard}>
      <div className={isSame ? styles.sameMO : styles.diffMO}>
        <h3>{verdict}</h3>
        <p>Similarity: {(score * 100).toFixed(1)}%</p>
      </div>
    </div>
  );
};
```

### 4.2 Explanation Bar Chart

Bar chart showing feature importance from SHAP values.

```typescript
const ExplanationChart = ({ data }: { data: { feature: string; importance: number }[] }) => {
  const chartData = data.slice(0, 10).map((item, index) => ({
    name: item.feature,
    value: Math.abs(item.importance),
    isPositive: item.importance > 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData}>
        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
        <YAxis />
        <Tooltip />
        <Bar dataKey="value">
          {chartData.map((entry, index) => (
            <Cell 
              key={index} 
              fill={entry.isPositive ? 'rgb(var(--primary-rgb))' : 'rgb(var(--muted-rgb))'} 
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};
```

### 4.3 Trend Line Chart

Line chart showing feature comparison between incidents.

```typescript
const TrendChart = ({ p1, p2 }: { p1: number[], p2: number[] }) => {
  const data = p1.map((val, idx) => ({
    feature: `Feature ${idx + 1}`,
    'Incident 1': val,
    'Incident 2': p2[idx] || 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <XAxis dataKey="feature" tick={{ fontSize: 11 }} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="Incident 1" 
          stroke="rgb(var(--primary-rgb))" 
          strokeWidth={2} 
        />
        <Line 
          type="monotone" 
          dataKey="Incident 2" 
          stroke="rgb(var(--muted-rgb))" 
          strokeWidth={2} 
        />
      </LineChart>
    </ResponsiveContainer>
  );
};
```

### 4.4 Data Table

Feature-by-feature comparison table.

```typescript
const DataTable = ({ p1, p2 }: { p1: number[], p2: number[] }) => {
  return (
    <table className={styles.dataTable}>
      <thead>
        <tr>
          <th>Feature</th>
          <th>Incident 1</th>
          <th>Incident 2</th>
          <th>Difference</th>
        </tr>
      </thead>
      <tbody>
        {p1.map((val, idx) => (
          <tr key={idx}>
            <td>Feature {idx + 1}</td>
            <td>{val.toFixed(2)}</td>
            <td>{(p2[idx] || 0).toFixed(2)}</td>
            <td>{(val - (p2[idx] || 0)).toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};
```

### 4.5 Stats Grid

Quick statistics summary.

```typescript
const StatsGrid = ({ result }) => {
  return (
    <div className={styles.statsGrid}>
      <div className={styles.statCard}>
        <div className={styles.statLabel}>Similarity</div>
        <div className={styles.statValue}>
          {(result.similarityScore * 100).toFixed(1)}%
        </div>
      </div>
      <div className={styles.statCard}>
        <div className={styles.statLabel}>Verdict</div>
        <div className={styles.statValue}>{result.verdict}</div>
      </div>
      <div className={styles.statCard}>
        <div className={styles.statLabel}>Features</div>
        <div className={styles.statValue}>
          {result.explanation?.length || 0}
        </div>
      </div>
      <div className={styles.statCard}>
        <div className={styles.statLabel}>Best Match</div>
        <div className={styles.statValue}>
          Pattern #{result.best_patterns?.pattern1_index || 0}
        </div>
      </div>
    </div>
  );
};
```

---

## 5. Styling

### 5.1 CSS Modules

We use CSS Modules for component-level styling to avoid conflicts.

```css
/* ResultDisplay.module.css */

.card {
  background: rgb(var(--background-rgb));
  border: 1px solid rgb(var(--border-rgb));
  border-radius: 8px;
  padding: 1.5rem;
  margin-top: 1rem;
}

.header {
  margin-bottom: 1rem;
  border-bottom: 1px solid rgb(var(--border-rgb));
  padding-bottom: 0.5rem;
}

.verdictCard {
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
  margin-bottom: 1rem;
}

.sameMO {
  background: rgba(var(--primary-rgb), 0.1);
  color: rgb(var(--primary-rgb));
  padding: 1rem;
  border-radius: 8px;
}

.diffMO {
  background: rgba(var(--muted-rgb), 0.1);
  color: rgb(var(--text-rgb));
  padding: 1rem;
  border-radius: 8px;
}

.statsGrid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin: 1rem 0;
}

.statCard {
  background: rgb(var(--surface-rgb));
  border: 1px solid rgb(var(--border-rgb));
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}

.chartCard {
  background: rgb(var(--surface-rgb));
  border: 1px solid rgb(var(--border-rgb));
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
}
```

### 5.2 Color Variables

```css
/* index.css */

:root {
  /* Primary colors (purple) */
  --primary-rgb: 139, 92, 246;
  
  /* Background */
  --background-rgb: 15, 23, 42;
  --surface-rgb: 30, 41, 59;
  
  /* Text */
  --text-rgb: 248, 250, 252;
  --muted-rgb: 148, 163, 184;
  
  /* Borders */
  --border-rgb: 51, 65, 85;
}
```

---

## 6. API Integration

### 6.1 API Client

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
    headers: { "Content-Type": "application/json" },
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

### 6.2 Using the API

```typescript
// Complete flow
const handleAnalysis = async () => {
  // Upload both files
  const [upload1, upload2] = await Promise.all([
    uploadFile(file1),
    uploadFile(file2),
  ]);

  // Get prediction from API
  const result = await predict(upload1.filename, upload2.filename);

  // Result contains:
  // - result.verdict: "SAME MO" or "DIFFERENT MO"
  // - result.similarityScore: 0.0-1.0
  // - result.best_patterns: pattern data
  // - result.explanation: feature importance
  console.log(result);
};
```

---

## 7. Running the Frontend

### 7.1 Prerequisites

```bash
# Install dependencies
cd frontend
npm install
```

### 7.2 Start Development Server

```bash
npm run dev
```

### 7.3 Access

Open **http://localhost:5173** in your browser.

### 7.4 Expected Workflow

```
1. Landing page (intro)
    │
    ▼ Click "Get Started"
2. File upload page
    │
    ▼ Drag & drop files + click "Analyze"
3. Loading state ("Analyzing...")
    │
    ▼ API response
4. Results page
    ├── Verdict (SAME MO / DIFFERENT MO)
    ├── Similarity score
    ├── Bar chart (feature importance)
    ├── Stats grid
    ├── Line chart (feature comparison)
    └── Data table (feature values)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| CORS errors | Backend must have CORS enabled |
| File not uploading | Check file format (CSV/Parquet) |
| No results | Need 5+ suspicious connections |
| Charts not showing | Check data structure |

### Development Tips

1. **Check Network tab** for API calls
2. **Console logs** for error messages
3. **Backend logs** for server-side issues

---

## Next Steps

- **[Architecture Deep Dive](architecture.md)**: CNN, Bi-LSTM, Siamese details
- **[Dataset Reference](dataset.md)**: UNSW-NB15 features, attacks
- **[Processing Pipeline](pipeline.md)**: End-to-end data flow

---

<div align="center">
  <sub>Part of the Intelligent Crime Pattern Recognition System</sub>
</div>