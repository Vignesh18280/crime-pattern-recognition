import React from 'react';
import { Button } from './common/Button';
import styles from './LandingPage.module.css';
import { UploadCloud, BrainCircuit, ShieldCheck } from 'lucide-react';

interface LandingPageProps {
  onLaunch: () => void;
}

const FeatureCard = ({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) => (
  <div className={styles.featureCard}>
    {icon}
    <h3>{title}</h3>
    <p>{description}</p>
  </div>
);

export const LandingPage: React.FC<LandingPageProps> = ({ onLaunch }) => {
  return (
    <div className={styles.landingPage}>
      <div className={styles.backgroundAnimation}></div>
      
      <section className={styles.hero}>
        <h1>Unmasking Digital Ghosts</h1>
        <p>
          Our advanced AI analyzes cybercrime incidents to uncover the 'Modus Operandi' of threat actors, linking seemingly unrelated attacks.
        </p>
        <Button onClick={onLaunch}>
          Launch Investigation
        </Button>
      </section>

      <section className={styles.howItWorks}>
        <h2>Three Steps to Clarity</h2>
        <div className={styles.featuresGrid}>
          <FeatureCard
            icon={<UploadCloud size={48} />}
            title="1. Upload Incidents"
            description="Provide raw, un-processed log files (.csv or .parquet) from two separate security incidents."
          />
          <FeatureCard
            icon={<BrainCircuit size={48} />}
            title="2. AI Analysis"
            description="Our Siamese-CNN-BiLSTM model finds suspicious patterns and compares their core characteristics."
          />
          <FeatureCard
            icon={<ShieldCheck size={48} />}
            title="3. Receive Verdict"
            description="Get a definitive similarity score and a clear verdict on whether the incidents share the same origin."
          />
        </div>
      </section>
    </div>
  );
};
