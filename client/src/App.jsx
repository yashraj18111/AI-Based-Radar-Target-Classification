// client/src/App.jsx
import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import GoogleDriveBox from './components/GoogleDriveBox';
import RadarDropzone from './components/RadarDropzone';
import SpectrogramResults from './components/SpectrogramResults';
import ClassificationHistory from './components/ClassificationHistory';
import ArchitectureInfo from './components/ArchitectureInfo';
import { Radio, AlertCircle, CheckCircle } from 'lucide-react';

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [signalStats, setSignalStats] = useState(null);
  const [healthData, setHealthData] = useState(null);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState(null);

  // Show temporary toast message
  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => {
      setToast(null);
    }, 4500);
  };

  // Check backend health & fetch MongoDB history on mount
  useEffect(() => {
    fetchHealth();
    fetchHistory();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealthData(data);
      }
    } catch (err) {
      console.warn('Backend not yet reachable:', err.message);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        if (data.success && data.history) {
          setHistory(data.history);
        }
      }
    } catch (err) {
      console.warn('History fetch error:', err.message);
    }
  };

  // Direct file upload
  const handleFileUpload = async (file) => {
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt')) {
      showToast('Please upload a valid radar CSV signal file (.csv or .txt)', 'error');
      return;
    }

    setIsLoading(true);
    setSelectedFileName(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Prediction request failed');
      }

      setPredictionData(data);
      setSignalStats(data.signal_stats);
      showToast(`Classified as ${data.predicted_class.toUpperCase()} (${data.confidence.toFixed(1)}%)`, 'success');
      fetchHistory();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Google Drive URL prediction
  const handleDriveClassify = async (driveUrl) => {
    setIsLoading(true);
    setSelectedFileName(`Google Drive File (${driveUrl.slice(0, 30)}...)`);

    try {
      const res = await fetch(`${API_BASE}/api/drive-predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ driveUrl }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Google Drive fetch & prediction failed');
      }

      setPredictionData(data);
      setSignalStats(data.signal_stats);
      setSelectedFileName(data.filename || 'Google Drive CSV');
      showToast(`Google Drive File Classified: ${data.predicted_class.toUpperCase()} (${data.confidence.toFixed(1)}%)`, 'success');
      fetchHistory();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Preset button selection
  const handlePresetSelect = async (className) => {
    setIsLoading(true);
    setSelectedFileName(`Sample ${className.toUpperCase()} Preset`);

    try {
      const res = await fetch(`${API_BASE}/api/sample/${className}`);
      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to load preset signal');
      }

      setPredictionData(data);
      setSignalStats(data.signal_stats);
      showToast(`Loaded ${className.toUpperCase()} preset: ${data.confidence.toFixed(1)}% match`, 'success');
      fetchHistory();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Clear history
  const handleClearHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`, { method: 'DELETE' });
      if (res.ok) {
        setHistory([]);
        showToast('Classification history cleared', 'success');
      }
    } catch (err) {
      showToast('Could not clear history: ' + err.message, 'error');
    }
  };

  return (
    <>
      <div className="bg-glow" />
      <div className="bg-grid" />

      <div className="app-container">
        {/* Navbar */}
        <Navbar healthData={healthData} />

        {/* Hero Section */}
        <div className="hero-section">
          <div className="hero-pill">
            <Radio size={14} />
            <span>AI Radar Signal Analysis & Micro-Doppler Classification</span>
          </div>
          <h1 className="hero-title">Classify Radar Targets with Deep Learning</h1>
          <p className="hero-subtitle">
            Transform raw FMCW micro-Doppler radar frequency chirps into 128×128 Range-Doppler spectrograms and classify People, Drones, and Cars using a PyTorch 2D Convolutional Neural Network.
          </p>
        </div>

        {/* PROMINENT GOOGLE DRIVE DEDICATED BOX */}
        <GoogleDriveBox
          onDriveClassify={handleDriveClassify}
          isLoading={isLoading}
          defaultFolderUrl={healthData?.defaultGoogleDriveFolder}
          apiBase={API_BASE}
        />

        {/* Main 2-Column Grid */}
        <main className="main-grid">
          {/* Left Column: Dropzone, Presets & Properties */}
          <RadarDropzone
            onFileUpload={handleFileUpload}
            onPresetSelect={handlePresetSelect}
            isLoading={isLoading}
            selectedFileName={selectedFileName}
            signalStats={signalStats}
          />

          {/* Right Column: Spectrogram & Inference */}
          <SpectrogramResults predictionData={predictionData} />
        </main>

        {/* Architecture & Pipeline Explanation */}
        <ArchitectureInfo />

        {/* Live MongoDB History Feed */}
        <ClassificationHistory
          history={history}
          onClearHistory={handleClearHistory}
        />
      </div>

      {/* Toast Notification Alert */}
      {toast && (
        <div className="toast-alert">
          {toast.type === 'error' ? (
            <AlertCircle size={18} color="#ef4444" />
          ) : (
            <CheckCircle size={18} color="#10b981" />
          )}
          <span>{toast.message}</span>
        </div>
      )}
    </>
  );
}
