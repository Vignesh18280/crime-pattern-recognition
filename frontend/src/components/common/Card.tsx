import React from 'react';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className }) => {
  const combinedClassName = `${styles.card} ${className || ''}`;
  return (
    <div className={combinedClassName}>
      {children}
    </div>
  );
};
