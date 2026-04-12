import React from 'react';
import { Card } from './common/Card';
import styles from './ResultDisplay.module.css';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

interface ResultDisplayProps {
  result: {
    verdict: string;
    similarityScore: number;
    incident1: string;
    incident2: string;
    explanation: { feature: string; importance: number }[];
  } | null;
}

const Verdict = ({ verdict, score }: { verdict: string; score: number }) => {
  const isSameMO = verdict === 'SAME MO';
  const scoreClass = isSameMO ? styles.sameMO : styles.differentMO;
  const percentage = (score * 100).toFixed(2);

  return (
    <div className={styles.verdictSection}>
      <p className={styles.verdictDescription}>Similarity Score</p>
      <p className={`${styles.score} ${scoreClass}`}>{percentage}%</p>
      <p className={`${styles.verdict} ${scoreClass}`}>{verdict}</p>
      <p className={styles.verdictDescription}>
        {isSameMO ? 'Indicates a high likelihood of the same threat actor.' : 'Indicates unrelated criminal activity.'}
      </p>
    </div>
  );
};

const ExplanationChart = ({ data }: { data: { feature: string; importance: number }[] }) => {
    const chartData = data.map(item => ({
        ...item,
        absImportance: Math.abs(item.importance)
    })).sort((a, b) => a.absImportance - b.absImportance);

    return (
        <div className={styles.chartSection}>
            <h3>Top Features Influencing Decision</h3>
            <ResponsiveContainer width="100%" height={400}>
                <BarChart data={chartData} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis 
                        dataKey="feature" 
                        type="category" 
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: 'rgb(var(--muted-rgb))', fontSize: 12 }}
                        width={150}
                    />
                    <Tooltip
                        cursor={{ fill: 'rgba(var(--primary-rgb), 0.2)' }}
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                return (
                                    <div style={{
                                        backgroundColor: 'rgb(var(--background-rgb))',
                                        border: '1px solid rgb(var(--border-rgb))',
                                        padding: '0.5rem',
                                        borderRadius: '0.5rem',
                                    }}>
                                        <p style={{fontWeight: 'bold'}}>{`${payload[0].payload.feature}`}</p>
                                        <p>{`Importance: ${payload[0].payload.importance.toFixed(4)}`}</p>
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Bar dataKey="absImportance">
                         {
                            chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.importance > 0 ? 'rgb(var(--primary-rgb))' : 'rgb(var(--muted-rgb))'} />
                            ))
                        }
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  if (!result) {
    return null;
  }

  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <h2>Analysis Complete</h2>
        <p>
          Comparison between <span>{result.incident1}</span> and <span>{result.incident2}</span>.
        </p>
      </div>
      <div className={styles.content}>
        <Verdict verdict={result.verdict} score={result.similarityScore} />
        <ExplanationChart data={result.explanation} />
      </div>
    </Card>
  );
};
