import React from 'react';
import { Card } from './common/Card';
import styles from './ResultDisplay.module.css';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend
} from 'recharts';

interface ResultDisplayProps {
  result: any;
}

const Verdict = ({ verdict, score }: { verdict: string; score: number }) => {
  const isSameMO = verdict === 'SAME MO';
  const percentage = (score * 100).toFixed(2);

  return (
    <div className={styles.verdictSection}>
      <p className={styles.verdictDescription}>Similarity Score</p>
      <p className={`${styles.score} ${isSameMO ? styles.sameMO : styles.differentMO}`}>
        {percentage}%
      </p>
      <p className={`${styles.verdict} ${isSameMO ? styles.sameMO : styles.differentMO}`}>
        {verdict}
      </p>
      <p className={styles.verdictDescription}>
        {isSameMO ? 'Indicates a high likelihood of the same threat actor.' : 'Indicates unrelated criminal activity.'}
      </p>
    </div>
  );
};

const ExplanationChart = ({ data }: { data: any[] }) => {
    if (!data || !data.length) return null;

    const chartData = data
        .map(item => ({ feature: item.feature?.slice(0, 12) || '?', value: item.importance || 0 }))
        .sort((a, b) => Math.abs(a.value) - Math.abs(b.value))
        .slice(-20);

    return (
        <div className={styles.chartSection}>
            <h3>Top Features Influencing Decision</h3>
            <ResponsiveContainer width="100%" height={400}>
                <BarChart data={chartData} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis dataKey="feature" type="category" width={120} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="value">
                        {chartData.map((entry, i) => (
                            <Cell key={i} fill={entry.value >= 0 ? 'rgb(var(--primary-rgb))' : 'rgb(var(--muted-rgb))'} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

// NEW: Stats Grid
const StatsGrid = ({ result }: { result: any }) => {
    return (
        <div className={styles.statsGrid}>
            <div className={styles.statCard}>
                <div className={styles.statLabel}>Similarity</div>
                <div className={styles.statValue}>{(result.similarityScore * 100).toFixed(1)}%</div>
            </div>
            <div className={styles.statCard}>
                <div className={styles.statLabel}>Verdict</div>
                <div className={styles.statValue} style={{ fontSize: '0.9rem' }}>{result.verdict}</div>
            </div>
            <div className={styles.statCard}>
                <div className={styles.statLabel}>Features</div>
                <div className={styles.statValue}>{result.explanation?.length || 0}</div>
            </div>
            <div className={styles.statCard}>
                <div className={styles.statLabel}>Best Match</div>
                <div className={styles.statValue}>Pattern #{result.best_patterns?.pattern1_index || 0}</div>
            </div>
        </div>
    );
};

// NEW: Modality Breakdown - Shows contribution from each modality
const ModalityBreakdown = ({ result }: { result: any }) => {
    const embeddings = result.embeddings || {};
    const modalities = result.modalities_a || {};
    const modalitiesB = result.modalities_b || {};
    
    const hasLog = modalities.has_log || modalitiesB.has_log;
    const hasImg = modalities.num_images > 0 || modalitiesB.num_images > 0;
    const hasBin = modalities.has_binary || modalitiesB.has_binary;
    
    const getSimilarity = (val: number) => (val * 100).toFixed(1) + '%';
    const getBarColor = (val: number) => val >= 0.7 ? '#22c55e' : val >= 0.4 ? '#eab308' : '#ef4444';
    
    if (!hasLog && !hasImg && !hasBin) return null;
    
    return (
        <div className={styles.modalitySection}>
            <h3>Modality Similarity Breakdown</h3>
            <div className={styles.modalityGrid}>
                {hasLog && (
                    <div className={styles.modalityCard}>
                        <div className={styles.modalityHeader}>
                            <span>📄 Network Logs</span>
                            <span style={{ color: getBarColor(embeddings.log_similarity), fontWeight: 'bold' }}>
                                {getSimilarity(embeddings.log_similarity)}
                            </span>
                        </div>
                        <div className={styles.progressBar}>
                            <div 
                                className={styles.progressFill} 
                                style={{ 
                                    width: `${(embeddings.log_similarity || 0) * 100}%`,
                                    backgroundColor: getBarColor(embeddings.log_similarity)
                                }}
                            />
                        </div>
                    </div>
                )}
                {hasImg && (
                    <div className={styles.modalityCard}>
                        <div className={styles.modalityHeader}>
                            <span>🖼️ Images</span>
                            <span style={{ color: getBarColor(embeddings.img_similarity), fontWeight: 'bold' }}>
                                {getSimilarity(embeddings.img_similarity)}
                            </span>
                        </div>
                        <div className={styles.progressBar}>
                            <div 
                                className={styles.progressFill}
                                style={{ 
                                    width: `${(embeddings.img_similarity || 0) * 100}%`,
                                    backgroundColor: getBarColor(embeddings.img_similarity)
                                }}
                            />
                        </div>
                    </div>
                )}
                {hasBin && (
                    <div className={styles.modalityCard}>
                        <div className={styles.modalityHeader}>
                            <span>⚙️ Binary</span>
                            <span style={{ color: getBarColor(embeddings.bin_similarity), fontWeight: 'bold' }}>
                                {getSimilarity(embeddings.bin_similarity)}
                            </span>
                        </div>
                        <div className={styles.progressBar}>
                            <div 
                                className={styles.progressFill}
                                style={{ 
                                    width: `${(embeddings.bin_similarity || 0) * 100}%`,
                                    backgroundColor: getBarColor(embeddings.bin_similarity)
                                }}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// NEW: Plain English Summary
const SummaryText = ({ result }: { result: any }) => {
    const isSame = result.verdict === 'SAME MO';
    const score = (result.similarityScore * 100).toFixed(1);
    const modalities = result.modalities_a || {};
    const modalitiesB = result.modalities_b || {};
    
    const modalitiesList = [];
    if (modalities.has_log || modalitiesB.has_log) modalitiesList.push("network logs");
    if (modalities.num_images > 0 || modalitiesB.num_images > 0) modalitiesList.push("images");
    if (modalities.has_binary || modalitiesB.has_binary) modalitiesList.push("binary files");
    
    const modStr = modalitiesList.length > 0 ? modalitiesList.join(", ") : "evidence";
    
    return (
        <div className={styles.summaryBox}>
            <p>
                <strong>Analysis Summary:</strong> The system analyzed multimodal evidence ({modStr}) from two incident cases.
                {isSame 
                    ? <><br />Found <strong>{score}% similarity</strong> — these incidents show characteristics typical of the <strong>same threat actor</strong>.</>
                    : <><br />Found only <strong>{score}% similarity</strong> — these incidents appear to be <strong>unrelated</strong> attacks.</>
                }
            </p>
        </div>
    );
};

// NEW: Data Comparison Table
const DataTable = ({ result }: { result: any }) => {
    const p1 = result.best_patterns?.pattern1_data || [];
    const p2 = result.best_patterns?.pattern2_data || [];
    
    if (!p1.length || !p2.length) return null;

    return (
        <div className={styles.tableWrapper}>
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
                    {p1.map((val: number, idx: number) => (
                        <tr key={idx}>
                            <td style={{ fontWeight: 600, color: 'rgb(var(--primary-rgb))' }}>Feature {idx + 1}</td>
                            <td>{val.toFixed(2)}</td>
                            <td>{(p2[idx] || 0).toFixed(2)}</td>
                            <td>{(val - (p2[idx] || 0)).toFixed(2)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

// NEW: Trend Line Chart
const TrendChart = ({ result }: { result: any }) => {
    const p1 = result.best_patterns?.pattern1_data || [];
    const p2 = result.best_patterns?.pattern2_data || [];
    
    if (!p1.length || !p2.length) return null;
    
    const data = p1.map((val: number, idx: number) => ({
        feature: `Feature ${idx + 1}`,
        'Incident 1': val,
        'Incident 2': p2[idx] || 0,
    }));

    return (
        <div className={styles.chartCard}>
            <h3>Feature Comparison Trend</h3>
            <ResponsiveContainer width="100%" height={200}>
                <LineChart data={data}>
                    <XAxis dataKey="feature" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="Incident 1" stroke="rgb(var(--primary-rgb))" strokeWidth={2} />
                    <Line type="monotone" dataKey="Incident 2" stroke="rgb(var(--muted-rgb))" strokeWidth={2} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  if (!result) return null;

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <h2>Analysis Complete</h2>
        <p>Comparing <span>{result.incident1}</span> vs <span>{result.incident2}</span></p>
      </div>
      
      <div className={styles.content}>
        <Verdict verdict={result.verdict} score={result.similarityScore} />
        <ExplanationChart data={result.explanation || []} />
      </div>

      {/* NEW: Detailed Section Below */}
      <div className={styles.detailSection}>
        <h2>Detailed Analysis</h2>
        
        <ModalityBreakdown result={result} />
        <StatsGrid result={result} />
        <SummaryText result={result} />
        <TrendChart result={result} />
        <DataTable result={result} />
      </div>
    </Card>
  );
};
