import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styles from './FileUploader.module.css';
import { Button } from './common/Button';
import { FileUp } from 'lucide-react';
import { uploadFile, predict } from '../lib/api';


interface FileUploaderProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (result: any) => void;
}

const SingleUploader = ({ file, setFile, title }: { file: File | null, setFile: (file: File | null) => void, title: string }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, [setFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/vnd.ms-excel': ['.csv'], 'application/parquet': ['.parquet']},
    multiple: false,
  });

  const dropzoneClasses = `${styles.dropzone} ${isDragActive ? styles.active : ''}`;

  return (
    <div>
      <h3 style={{textAlign: 'center', marginBottom: '1rem'}}>{title}</h3>
      <div {...getRootProps()} className={dropzoneClasses}>
        <input {...getInputProps()} />
        {file ? (
          <p className={styles.fileName}>{file.name}</p>
        ) : (
          <>
            <FileUp size={48} />
            <p>{isDragActive ? 'Drop the file here ...' : 'Drag & drop a log file, or click to select'}</p>
            <p style={{fontSize: '0.8rem'}}>.csv or .parquet files</p>
          </>
        )}
      </div>
    </div>
  );
};

export const FileUploader: React.FC<FileUploaderProps> = ({ onAnalysisStart, onAnalysisComplete }) => {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!file1 || !file2) {
      alert('Please upload both incident log files.');
      return;
    }

    setIsLoading(true);
    onAnalysisStart();
    
    try {
      const [upload1, upload2] = await Promise.all([
        uploadFile(file1),
        uploadFile(file2),
      ]);

      const result = await predict(upload1.filename, upload2.filename);
      
      onAnalysisComplete(result);

    } catch (error: any) {
      console.error(error);
      alert(error.message || 'An unknown error occurred.');
      onAnalysisComplete(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.grid}>
        <SingleUploader file={file1} setFile={setFile1} title="Incident Log 1" />
        <SingleUploader file={file2} setFile={setFile2} title="Incident Log 2" />
      </div>
      <div className={styles.buttonContainer}>
        <Button onClick={handleAnalyze} disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Analyze Incidents'}
        </Button>
      </div>
    </div>
  );
};
