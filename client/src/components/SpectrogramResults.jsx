// client/src/components/SpectrogramResults.jsx
import React from 'react';
import { Eye, BarChart3, Image as ImageIcon } from 'lucide-react';

export default function SpectrogramResults({ predictionData }) {
  const hasPrediction = !!predictionData;
  const predClass = predictionData?.predicted_class?.toLowerCase();
  const confidence = predictionData?.confidence ?? 0;
  const probabilities = predictionData?.probabilities || { car: 0, drone: 0, person: 0 };
  const spectrogramImg = predictionData?.spectrogram_image;

  const emojiMap = {
    car: '🚗',
    drone: '🛸',
    person: '🚶',
  };

  const colorMap = {
    car: '#ef4444',
    drone: '#38bdf8',
    person: '#10b981',
  };

  const activeColor = predClass ? colorMap[predClass] || '#38bdf8' : '#38bdf8';

  return (
    <section className="panel output-panel">
      <div className="panel-header">
        <h2>
          <Eye size={18} color="#10b981" />
          <span>2. Micro-Doppler Spectrogram & Inference</span>
        </h2>
        <p>Log-dB spectrogram RGB generation and PyTorch Softmax probability breakdown.</p>
      </div>

      {/* Rendered Spectrogram Display */}
      <div className="spectrogram-card">
        <div className="spectrogram-header">
          <span>Generated Range-Doppler Heatmap</span>
          <span style={{ fontFamily: 'var(--font-mono)' }}>128 × 128 (Viridis RGB)</span>
        </div>

        <div className="spectrogram-display">
          {spectrogramImg ? (
            <img src={spectrogramImg} alt="Micro-Doppler Spectrogram" />
          ) : (
            <div className="spectrogram-placeholder">
              <ImageIcon size={44} />
              <p>Upload a radar signal CSV or select a preset to generate the spectrogram</p>
            </div>
          )}
        </div>
      </div>

      {/* Prediction Banner */}
      <div
        className="prediction-banner"
        style={{
          borderColor: hasPrediction ? activeColor : 'rgba(56, 189, 248, 0.3)',
          boxShadow: hasPrediction ? `0 0 25px ${activeColor}33` : 'none',
        }}
      >
        <div className="banner-tag">Detected Radar Target</div>
        <div className="banner-title" style={{ color: hasPrediction ? activeColor : '#ffffff' }}>
          {hasPrediction
            ? `${emojiMap[predClass] || '🎯'} ${predClass.toUpperCase()}`
            : 'Awaiting Radar Input'}
        </div>
        <div className="banner-confidence">
          {hasPrediction
            ? `${confidence.toFixed(2)}% Confidence Score`
            : 'Select a preset or upload CSV to begin inference'}
        </div>
      </div>

      {/* Probability Bars */}
      <div className="prob-card">
        <h3>
          <BarChart3 size={15} style={{ display: 'inline', marginRight: '6px' }} />
          Target Classification Breakdown
        </h3>

        <div className="prob-row">
          <div className="prob-info">
            <span>🚗 Car</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {probabilities.car !== undefined ? `${probabilities.car.toFixed(1)}%` : '0.0%'}
            </span>
          </div>
          <div className="prob-track">
            <div
              className="prob-fill fill-car"
              style={{ width: `${probabilities.car || 0}%` }}
            />
          </div>
        </div>

        <div className="prob-row">
          <div className="prob-info">
            <span>🛸 Drone</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {probabilities.drone !== undefined ? `${probabilities.drone.toFixed(1)}%` : '0.0%'}
            </span>
          </div>
          <div className="prob-track">
            <div
              className="prob-fill fill-drone"
              style={{ width: `${probabilities.drone || 0}%` }}
            />
          </div>
        </div>

        <div className="prob-row">
          <div className="prob-info">
            <span>🚶 Person</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {probabilities.person !== undefined ? `${probabilities.person.toFixed(1)}%` : '0.0%'}
            </span>
          </div>
          <div className="prob-track">
            <div
              className="prob-fill fill-person"
              style={{ width: `${probabilities.person || 0}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
