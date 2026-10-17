// client/src/components/GoogleDriveBox.jsx
// Dedicated Google Drive test data hub and direct link importer

import React, { useState } from 'react';
import {
  ExternalLink,
  Download,
  FolderOpen,
  Link as LinkIcon,
  Loader2,
  CheckCircle2,
  Info,
  Car,
  Plane,
  User,
} from 'lucide-react';

export default function GoogleDriveBox({
  onDriveClassify,
  isLoading,
  defaultFolderUrl = 'https://drive.google.com/drive/folders/1rVmUWYyHWgEyzrUYRLMaoy2uvZ0M1t_O?usp=sharing',
  apiBase = '',
}) {
  const [inputUrl, setInputUrl] = useState('');
  const [driveFolderUrl, setDriveFolderUrl] = useState(defaultFolderUrl);

  React.useEffect(() => {
    if (defaultFolderUrl) {
      setDriveFolderUrl(defaultFolderUrl);
    }
  }, [defaultFolderUrl]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputUrl.trim()) return;
    onDriveClassify(inputUrl.trim());
  };

  const handleDownloadSample = (className) => {
    window.open(`${apiBase}/api/download-sample/${className}`, '_blank');
  };

  return (
    <section className="gdrive-card">
      {/* Card Header */}
      <div className="gdrive-header">
        <div className="gdrive-title-group">
          <div className="gdrive-icon-badge">
            <FolderOpen size={24} />
          </div>
          <div className="gdrive-title-text">
            <h2>Google Drive Radar Test Data Repository</h2>
            <p>
              Access raw micro-Doppler radar test signals or paste a Google Drive file link below to test the AI model
            </p>
          </div>
        </div>

        {/* Primary Google Drive Link Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <a
            href={driveFolderUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="gdrive-main-btn"
            title="Open Google Drive Test Dataset in new tab"
          >
            <ExternalLink size={16} />
            <span>Open Google Drive Dataset</span>
          </a>
        </div>
      </div>

      {/* Direct Google Drive File Link Input */}
      <form onSubmit={handleSubmit}>
        <div className="gdrive-input-box">
          <LinkIcon size={18} color="#38bdf8" />
          <input
            type="text"
            placeholder="Paste Google Drive shareable file link here (e.g., https://drive.google.com/file/d/.../view)"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="gdrive-submit-btn"
            disabled={isLoading || !inputUrl.trim()}
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="spinner" />
                <span>Downloading & Inferring...</span>
              </>
            ) : (
              <>
                <CheckCircle2 size={16} />
                <span>Fetch from Drive & Classify</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Quick Download Sample Signals Bar */}
      <div className="sample-downloads-bar">
        <div className="download-bar-title">
          <Download size={14} color="#38bdf8" />
          <span>Quick Download Sample Radar Signals (CSV):</span>
        </div>

        <div className="download-chips">
          <button
            type="button"
            className="download-chip"
            onClick={() => handleDownloadSample('car')}
            title="Download real FMCW Car radar signal CSV"
          >
            <Car size={13} color="#f87171" />
            <span>Car_Sample.csv</span>
          </button>

          <button
            type="button"
            className="download-chip"
            onClick={() => handleDownloadSample('drone')}
            title="Download real FMCW Drone radar signal CSV"
          >
            <Plane size={13} color="#38bdf8" />
            <span>Drone_Sample.csv</span>
          </button>

          <button
            type="button"
            className="download-chip"
            onClick={() => handleDownloadSample('person')}
            title="Download real FMCW Person radar signal CSV"
          >
            <User size={13} color="#34d399" />
            <span>Person_Sample.csv</span>
          </button>
        </div>
      </div>

      {/* Helpful Instructions */}
      <div
        style={{
          marginTop: '1rem',
          padding: '0.75rem 1rem',
          background: 'rgba(56, 189, 248, 0.05)',
          borderRadius: '10px',
          border: '1px solid rgba(56, 189, 248, 0.15)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.6rem',
          fontSize: '0.78rem',
          color: '#94a3b8',
        }}
      >
        <Info size={16} color="#38bdf8" style={{ flexShrink: 0 }} />
        <span>
          <strong>Testing Guide:</strong> 1) Download any sample CSV above or from the Google Drive repository, 2) Drag and drop into the classifier dropzone below (or paste your own shared Drive link), 3) The PyTorch model converts the raw matrix to a 128×128 Viridis spectrogram and identifies the target.
        </span>
      </div>
    </section>
  );
}
