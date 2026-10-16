// client/src/components/ClassificationHistory.jsx
import React from 'react';
import { History, Trash2, Database, ExternalLink } from 'lucide-react';

export default function ClassificationHistory({ history, onClearHistory }) {
  if (!history || history.length === 0) {
    return (
      <section className="history-section">
        <div className="history-header">
          <h2>
            <History size={18} color="#a855f7" />
            <span>Classification History (MongoDB Live Feed)</span>
          </h2>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          No classifications recorded yet. Run a test via Google Drive, Preset, or File Upload to see test logs here.
        </p>
      </section>
    );
  }

  const getSourceBadge = (source) => {
    switch (source) {
      case 'drive':
        return <span className="badge badge-cyan">Google Drive</span>;
      case 'preset':
        return <span className="badge badge-purple">Preset</span>;
      default:
        return <span className="badge">Direct Upload</span>;
    }
  };

  const getClassBadge = (cls) => {
    const c = (cls || '').toLowerCase();
    if (c === 'car') return <span style={{ color: '#ef4444', fontWeight: 600 }}>🚗 Car</span>;
    if (c === 'drone') return <span style={{ color: '#38bdf8', fontWeight: 600 }}>🛸 Drone</span>;
    if (c === 'person') return <span style={{ color: '#10b981', fontWeight: 600 }}>🚶 Person</span>;
    return <span>{cls}</span>;
  };

  return (
    <section className="history-section">
      <div className="history-header">
        <h2>
          <History size={18} color="#a855f7" />
          <span>Classification History (MongoDB Live Feed)</span>
        </h2>
        <button
          type="button"
          className="clear-btn"
          onClick={onClearHistory}
          title="Clear all stored classification history"
        >
          <Trash2 size={13} style={{ display: 'inline', marginRight: '4px' }} />
          Clear History
        </button>
      </div>

      <div className="history-table-wrapper">
        <table className="history-table">
          <thead>
            <tr>
              <th>Spectrogram</th>
              <th>Filename / ID</th>
              <th>Source</th>
              <th>Detected Target</th>
              <th>Confidence</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item, idx) => (
              <tr key={item._id || idx}>
                <td>
                  {item.spectrogramThumbnail ? (
                    <img
                      src={item.spectrogramThumbnail}
                      alt="Thumbnail"
                      style={{
                        width: '36px',
                        height: '36px',
                        borderRadius: '6px',
                        border: '1px solid rgba(56, 189, 248, 0.3)',
                        objectFit: 'cover',
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        width: '36px',
                        height: '36px',
                        borderRadius: '6px',
                        background: 'rgba(255, 255, 255, 0.05)',
                      }}
                    />
                  )}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                  {item.filename}
                </td>
                <td>{getSourceBadge(item.source)}</td>
                <td>{getClassBadge(item.predictedClass)}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {item.confidence ? `${item.confidence.toFixed(1)}%` : '--'}
                </td>
                <td style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  {item.createdAt ? new Date(item.createdAt).toLocaleTimeString() : 'Just now'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
