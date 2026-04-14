import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styles from './FileUploader.module.css';
import { Button } from './common/Button';
import { FileUp, Image, FileCode, Trash2 } from 'lucide-react';
import { predictMultimodal, type MultimodalFiles } from '../lib/api';

interface FileUploaderProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (result: any) => void;
}

interface IncidentUploaderProps {
  files: MultimodalFiles;
  setFiles: React.Dispatch<React.SetStateAction<MultimodalFiles>>;
  title: string;
}

const ModalityUploader: React.FC<{
  icon: React.ReactNode;
  label: string;
  accept: Record<string, string[]>;
  files: File[];
  onChange: (files: File[]) => void;
  multiple?: boolean;
}> = ({ icon, label, accept, files, onChange, multiple }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (multiple) {
      onChange([...files, ...acceptedFiles]);
    } else {
      onChange(acceptedFiles);
    }
  }, [files, onChange, multiple]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    multiple,
  });

  const removeFile = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();
    onChange(files.filter((_, i) => i !== index));
  };

  return (
    <div className={styles.modalitySection}>
      <div {...getRootProps()} className={`${styles.modalityDropzone} ${isDragActive ? styles.active : ''}`}>
        <input {...getInputProps()} />
        {icon}
        <p>{label}</p>
        <p style={{fontSize: '0.7rem'}}>{multiple ? 'Drop multiple or click' : 'Drop or click'}</p>
      </div>
      {files.length > 0 && (
        <div className={styles.fileList}>
          {files.map((file, idx) => (
            <div key={idx} className={styles.fileItem}>
              <span className={styles.fileName}>{file.name}</span>
              <button onClick={(e) => removeFile(e, idx)} className={styles.removeBtn}>
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const IncidentUploader: React.FC<IncidentUploaderProps> = ({ files, setFiles, title }) => {
  const updateFiles = (type: 'log' | 'images' | 'binary', newFiles: File[]) => {
    setFiles(prev => ({
      ...prev,
      [type]: type === 'images' ? newFiles : newFiles[0] || null
    }));
  };

  return (
    <div className={styles.incidentBox}>
      <h3 style={{textAlign: 'center', marginBottom: '1rem'}}>{title}</h3>
      
      <ModalityUploader
        icon={<FileUp size={24} />}
        label="Network Log"
        accept={{ 'text/csv': ['.csv'], 'application/vnd.ms-excel': ['.csv'], 'application/parquet': ['.parquet']}}
        files={files.log ? [files.log] : []}
        onChange={(f) => updateFiles('log', f)}
        multiple={false}
      />
      
      <ModalityUploader
        icon={<Image size={24} />}
        label="Images"
        accept={{ 'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp']}}
        files={files.images}
        onChange={(f) => updateFiles('images', f)}
        multiple={true}
      />
      
      <ModalityUploader
        icon={<FileCode size={24} />}
        label="Binary (.exe)"
        accept={{ 'application/x-executable': ['.exe'], 'application/octet-stream': ['.exe', '.bin']}}
        files={files.binary ? [files.binary] : []}
        onChange={(f) => updateFiles('binary', f)}
        multiple={false}
      />
    </div>
  );
};

export const FileUploader: React.FC<FileUploaderProps> = ({ onAnalysisStart, onAnalysisComplete }) => {
  const [filesA, setFilesA] = useState<MultimodalFiles>({ log: null, images: [], binary: null });
  const [filesB, setFilesB] = useState<MultimodalFiles>({ log: null, images: [], binary: null });
  const [isLoading, setIsLoading] = useState(false);

  const hasAnyFile = (files: MultimodalFiles) => 
    files.log !== null || files.images.length > 0 || files.binary !== null;

  const handleAnalyze = async () => {
    if (!hasAnyFile(filesA) || !hasAnyFile(filesB)) {
      alert('Please upload evidence for both incidents (at least one file per incident).');
      return;
    }

    setIsLoading(true);
    onAnalysisStart();
    
    try {
      const result = await predictMultimodal(filesA, filesB);
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
        <IncidentUploader files={filesA} setFiles={setFilesA} title="Incident A" />
        <IncidentUploader files={filesB} setFiles={setFilesB} title="Incident B" />
      </div>
      <div className={styles.buttonContainer}>
        <Button onClick={handleAnalyze} disabled={isLoading}>
          {isLoading ? 'Analyzing Multimodal Evidence...' : 'Compare MO Patterns'}
        </Button>
      </div>
    </div>
  );
};