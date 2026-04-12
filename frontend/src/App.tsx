import { useState } from 'react';
import styles from './App.module.css';
import { LandingPage } from './components/LandingPage';
import { FileUploader } from './components/FileUploader';
import { ResultDisplay } from './components/ResultDisplay';

const AppUI = () => {
  const [result, setResult] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAnalysisComplete = (analysisResult: any) => {
    setResult(analysisResult);
    setIsLoading(false);
  };

  const handleAnalysisStart = () => {
    setIsLoading(true);
    setResult(null);
  }

  return (
    <div style={{width: '100%'}}>
      <FileUploader 
        onAnalysisStart={handleAnalysisStart}
        onAnalysisComplete={handleAnalysisComplete}
      />

      {isLoading && (
        <div style={{textAlign: 'center', marginTop: '2rem'}}>
          <p>Analyzing incidents... this may take a moment.</p>
        </div>
      )}
      
      {result && (
        <div style={{marginTop: '2rem'}}>
          <ResultDisplay result={result} />
        </div>
      )}
    </div>
  )
};



function App() {
  const [view, setView] = useState<'landing' | 'app'>('landing');

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerTitle}>
          <h1>Intelligent Crime Pattern Recognition</h1>
          <p>A Hybrid Siamese-CNN-BiLSTM Architecture for Cybercrime Modus Operandi Detection</p>
        </div>
      </header>

      <main className={styles.main}>
        {view === 'landing' ? (
          <LandingPage onLaunch={() => setView('app')} />
        ) : (
          <AppUI />
        )}
      </main>
    </div>
  );
}

export default App;


