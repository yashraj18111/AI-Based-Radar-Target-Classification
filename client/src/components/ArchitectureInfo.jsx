// client/src/components/ArchitectureInfo.jsx
import React from 'react';
import { Layers, ArrowRight, ShieldCheck, CheckCircle } from 'lucide-react';

export default function ArchitectureInfo() {
  return (
    <section
      style={{
        marginTop: '2rem',
        background: 'var(--bg-card)',
        backdropFilter: 'blur(12px)',
        border: '1px solid var(--border-color)',
        borderRadius: '18px',
        padding: '1.5rem',
      }}
    >
      <div className="panel-header">
        <h2>
          <Layers size={18} color="#38bdf8" />
          <span>Deep Learning Architecture & Signal Pipeline</span>
        </h2>
        <p>End-to-end signal processing and convolutional classification flow.</p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1rem',
          marginTop: '1rem',
        }}
      >
        <div className="stats-box">
          <div style={{ color: '#38bdf8', fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            Phase 1: FMCW Signal Acquisition
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Raw 2D matrix representing reflected micro-Doppler signals sampled across frequency chirps and time intervals.
          </p>
        </div>

        <div className="stats-box">
          <div style={{ color: '#10b981', fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            Phase 2: Spectrogram Synthesis
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            1%–99% percentile clipping, log-power normalization, and colormap mapping into a 128×128 Viridis RGB heatmap.
          </p>
        </div>

        <div className="stats-box">
          <div style={{ color: '#a855f7', fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            Phase 3: PyTorch 2D-CNN
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            4 convolutional blocks (32 &rarr; 64 &rarr; 128 &rarr; 256 channels) with BatchNorm, MaxPool, and LeakyReLU activations.
          </p>
        </div>

        <div className="stats-box">
          <div style={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.35rem' }}>
            Phase 4: Target Prediction
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Adaptive average pooling + fully-connected linear layers computing softmax probabilities across Car, Drone, and Person.
          </p>
        </div>
      </div>
    </section>
  );
}
