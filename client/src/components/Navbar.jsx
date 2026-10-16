// client/src/components/Navbar.jsx
import React from 'react';
import { Radio, Database, Cpu, Globe } from 'lucide-react';

export default function Navbar({ healthData }) {
  const isOnline = healthData && healthData.status === 'online';
  const isDbConnected = healthData && healthData.database && healthData.database.includes('MongoDB');

  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-radar-icon">
          <Radio size={22} />
        </div>
        <div className="brand-text">
          <h1>RadarSense AI</h1>
          <span>Micro-Doppler Target Classifier & Spectrogram Suite</span>
        </div>
      </div>

      <div className="nav-badges">
        <span className="badge badge-purple">
          <Cpu size={14} /> PyTorch 2D-CNN
        </span>

        <span className={`badge ${isDbConnected ? 'badge-green' : 'badge-cyan'}`}>
          <Database size={14} /> {isDbConnected ? 'MongoDB Atlas' : 'In-Memory DB'}
        </span>

        <span className="badge">
          <Globe size={14} /> MERN Stack
        </span>

        <span className="badge">
          <span className={`status-dot ${isOnline ? '' : 'offline'}`} />
          {isOnline ? 'System Live' : 'Backend Connecting...'}
        </span>
      </div>
    </header>
  );
}
