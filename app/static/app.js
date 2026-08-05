// Frontend Javascript for RadarSense AI Dashboard

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const fileNameDisplay = document.getElementById('fileName');

  // Check Backend Status
  checkBackendHealth();

  // Dropzone Events
  browseBtn.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });
});

async function checkBackendHealth() {
  const statusBadge = document.getElementById('systemStatus');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'online') {
      statusBadge.textContent = 'PyTorch Ready';
      statusBadge.classList.add('status-badge');
    }
  } catch (err) {
    statusBadge.textContent = 'Offline';
    statusBadge.style.color = '#ff4444';
  }
}

async function handleFileUpload(file) {
  if (!file.name.endsWith('.csv') && !file.name.endsWith('.txt')) {
    alert('Please upload a valid radar CSV signal file.');
    return;
  }

  document.getElementById('fileName').textContent = file.name;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || 'Prediction failed');
    }

    renderResults(data);
  } catch (err) {
    alert('Error processing radar signal: ' + err.message);
  }
}

async function loadSamplePreset(className) {
  document.getElementById('fileName').textContent = `Sample ${className.toUpperCase()} Preset`;

  try {
    const res = await fetch(`/api/sample-content/${className}`);
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.detail || 'Sample load failed');
    }

    renderResults(data);
  } catch (err) {
    alert(`Error loading sample preset for ${className}: ` + err.message);
  }
}

function renderResults(data) {
  // 1. Render Spectrogram Image
  const imgElement = document.getElementById('spectrogramImg');
  const placeholder = document.querySelector('.placeholder-text');

  imgElement.src = data.spectrogram_image;
  imgElement.classList.remove('hidden');
  placeholder.classList.add('hidden');

  // 2. Render Target Prediction Banner
  const predClassEl = document.getElementById('predClass');
  const predConfEl = document.getElementById('predConfidence');

  const predClass = data.predicted_class;
  const confidence = data.confidence;

  const emojiMap = { car: '🚗', drone: '🛸', person: '🚶' };
  const emoji = emojiMap[predClass] || '🎯';

  predClassEl.textContent = `${emoji} ${predClass.toUpperCase()}`;
  predConfEl.textContent = `${confidence.toFixed(1)}% Confidence`;

  // Color border of prediction banner according to target class
  const banner = document.getElementById('predictionBanner');
  const colorMap = { car: '#ff4444', drone: '#38bdf8', person: '#34d399' };
  banner.style.borderColor = colorMap[predClass] || '#38bdf8';

  // 3. Render Probability Bars
  const probs = data.probabilities;

  updateProbBar('valCar', 'barCar', probs.car || 0);
  updateProbBar('valDrone', 'barDrone', probs.drone || 0);
  updateProbBar('valPerson', 'barPerson', probs.person || 0);

  // 4. Render Signal Matrix Statistics
  if (data.signal_stats) {
    const stats = data.signal_stats;
    document.getElementById('statShape').textContent = stats.matrix_shape ? stats.matrix_shape.join(' × ') : '--';
    document.getElementById('statMin').textContent = stats.min_power ? stats.min_power.toFixed(2) : '--';
    document.getElementById('statMax').textContent = stats.max_power ? stats.max_power.toFixed(2) : '--';
    document.getElementById('statMean').textContent = stats.mean_power ? stats.mean_power.toFixed(2) : '--';
  }
}

function updateProbBar(valId, barId, percentage) {
  const valEl = document.getElementById(valId);
  const barEl = document.getElementById(barId);

  if (valEl) valEl.textContent = `${percentage.toFixed(1)}%`;
  if (barEl) barEl.style.width = `${percentage}%`;
}
