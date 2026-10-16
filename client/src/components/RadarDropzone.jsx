// client/src/components/RadarDropzone.jsx
import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Zap, Loader2, Activity } from 'lucide-react';

export default function RadarDropzone({
  onFileUpload,
  onPresetSelect,
  isLoading,
  selectedFileName,
  signalStats,
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <section className="panel input-panel">
      <div className="panel-header">
        <h2>
          <Activity size={18} color="#38bdf8" />
          <span>1. Radar Signal Input & Presets</span>
        </h2>
        <p>Drop a raw micro-Doppler radar matrix (.csv) or trigger one of the presets.</p>
      </div>

      {/* Dropzone */}
      <div
        className={`dropzone ${isDragOver ? 'active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".csv,.txt"
          hidden
        />

        <div className="dropzone-icon">
          {isLoading ? (
            <Loader2 size={26} className="spinner" />
          ) : (
            <UploadCloud size={26} />
          )}
        </div>

        <div className="dropzone-title">
          {isLoading ? 'Processing Radar Signal...' : 'Drag & drop radar CSV matrix here'}
        </div>

        <div className="dropzone-sub">
          or <button type="button" className="btn-link">browse files</button>
        </div>

        {selectedFileName && (
          <div className="selected-file-pill">
            <FileText size={12} />
            <span>{selectedFileName}</span>
          </div>
        )}
      </div>

      {/* Quick Test Presets */}
      <div className="presets-group">
        <h3>
          <Zap size={14} color="#f59e0b" />
          <span>Quick Test Presets (Instant Inference):</span>
        </h3>

        <div className="preset-buttons">
          <button
            type="button"
            className="preset-btn"
            onClick={() => onPresetSelect('car')}
            disabled={isLoading}
          >
            <span className="preset-icon">🚗</span>
            <span className="preset-label">Car Signal</span>
          </button>

          <button
            type="button"
            className="preset-btn"
            onClick={() => onPresetSelect('drone')}
            disabled={isLoading}
          >
            <span className="preset-icon">🛸</span>
            <span className="preset-label">Drone Signal</span>
          </button>

          <button
            type="button"
            className="preset-btn"
            onClick={() => onPresetSelect('person')}
            disabled={isLoading}
          >
            <span className="preset-icon">🚶</span>
            <span className="preset-label">Person Signal</span>
          </button>
        </div>
      </div>

      {/* Signal Matrix Properties */}
      <div className="stats-box">
        <h3>Radar Signal Matrix Properties</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Dimensions</span>
            <span className="stat-value">
              {signalStats && signalStats.matrix_shape
                ? signalStats.matrix_shape.join(' × ')
                : '-- × --'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Min Power (dB)</span>
            <span className="stat-value">
              {signalStats && signalStats.min_power !== undefined
                ? `${signalStats.min_power} dB`
                : '--'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Max Power (dB)</span>
            <span className="stat-value">
              {signalStats && signalStats.max_power !== undefined
                ? `${signalStats.max_power} dB`
                : '--'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Mean Intensity</span>
            <span className="stat-value">
              {signalStats && signalStats.mean_power !== undefined
                ? `${signalStats.mean_power} dB`
                : '--'}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
