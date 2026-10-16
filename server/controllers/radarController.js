// server/controllers/radarController.js
// Handles radar file processing, Google Drive fetching, Python inference, and MongoDB storage

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const Classification = require('../models/Classification');
const { getDBStatus } = require('../config/db');

// In-memory fallback if MongoDB is not connected
const inMemoryHistory = [];

const PROJECT_ROOT = path.resolve(__dirname, '../..');
const PYTHON_CMD = process.env.PYTHON_CMD || 'python';
const PREDICT_SCRIPT = path.join(PROJECT_ROOT, 'ml_engine', 'predict.py');

/**
 * Helper to execute Python inference on a CSV file
 */
const runPythonInference = (csvFilePath) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn(PYTHON_CMD, [PREDICT_SCRIPT, csvFilePath], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(
          new Error(
            `Python process exited with code ${code}. Error: ${stderrData || stdoutData}`
          )
        );
      }

      try {
        // Find the last JSON line if any debug logs were printed
        const lines = stdoutData.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        const parsed = JSON.parse(lastLine);

        if (!parsed.success) {
          return reject(new Error(parsed.error || 'Model inference failed'));
        }

        resolve(parsed);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse inference result: ${err.message}. Raw output: ${stdoutData.slice(0, 200)}`
          )
        );
      }
    });

    pythonProcess.on('error', (err) => {
      reject(new Error(`Failed to start Python process: ${err.message}`));
    });
  });
};

/**
 * Save record to MongoDB or in-memory array
 */
const saveRecord = async (data) => {
  const recordData = {
    filename: data.filename || 'radar_signal.csv',
    source: data.source || 'upload',
    driveUrl: data.driveUrl || null,
    predictedClass: data.predicted_class,
    confidence: data.confidence,
    probabilities: data.probabilities,
    signalStats: data.signal_stats,
    spectrogramThumbnail: data.spectrogram_image,
    createdAt: new Date(),
  };

  if (getDBStatus()) {
    try {
      const record = new Classification(recordData);
      await record.save();
      return record;
    } catch (err) {
      console.error('Error saving to MongoDB, using memory fallback:', err.message);
    }
  }

  // Fallback to in-memory history (limit 50 items)
  recordData._id = 'mem_' + Date.now() + Math.random().toString(36).substring(2, 7);
  inMemoryHistory.unshift(recordData);
  if (inMemoryHistory.length > 50) inMemoryHistory.pop();
  return recordData;
};

/**
 * Extract Google Drive file ID from standard Google Drive share links
 */
const extractDriveFileId = (inputUrl) => {
  if (!inputUrl) return null;
  const trimmed = inputUrl.trim();

  // Pattern 1: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
  const matchFileD = trimmed.match(/\/file\/d\/([a-zA-Z0-9_-]+)/);
  if (matchFileD && matchFileD[1]) return matchFileD[1];

  // Pattern 2: https://drive.google.com/open?id=FILE_ID or uc?id=FILE_ID
  const matchIdParam = trimmed.match(/[?&]id=([a-zA-Z0-9_-]+)/);
  if (matchIdParam && matchIdParam[1]) return matchIdParam[1];

  // Pattern 3: direct raw file ID string
  if (/^[a-zA-Z0-9_-]{25,}$/.test(trimmed)) {
    return trimmed;
  }

  return null;
};

/**
 * POST /api/predict
 * Upload a radar CSV file directly
 */
exports.predictUploadedFile = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No file uploaded' });
  }

  const tempFilePath = req.file.path;

  try {
    const inferenceResult = await runPythonInference(tempFilePath);
    inferenceResult.filename = req.file.originalname;

    // Save to database
    await saveRecord({
      ...inferenceResult,
      source: 'upload',
    });

    res.json({
      success: true,
      ...inferenceResult,
    });
  } catch (err) {
    console.error('Prediction error:', err);
    res.status(500).json({ success: false, error: err.message });
  } finally {
    // Clean up temporary uploaded file
    fs.unlink(tempFilePath, () => {});
  }
};

/**
 * POST /api/drive-predict
 * Fetch a CSV file from a Google Drive share link and run inference
 */
exports.predictFromGoogleDrive = async (req, res) => {
  const { driveUrl } = req.body;

  if (!driveUrl) {
    return res.status(400).json({
      success: false,
      error: 'Google Drive URL or File ID is required.',
    });
  }

  const fileId = extractDriveFileId(driveUrl);
  if (!fileId) {
    return res.status(400).json({
      success: false,
      error:
        'Could not extract Google Drive File ID. Please make sure the link is in the format: https://drive.google.com/file/d/<FILE_ID>/view or provide a public sharing link.',
    });
  }

  // Google Drive direct download URL
  const downloadUrl = `https://drive.google.com/uc?export=download&id=${fileId}`;
  const tempDir = path.join(PROJECT_ROOT, 'uploads');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  const tempFilePath = path.join(tempDir, `gdrive_${fileId}_${Date.now()}.csv`);

  try {
    console.log(`Downloading file from Google Drive (ID: ${fileId})...`);

    const response = await axios({
      method: 'GET',
      url: downloadUrl,
      responseType: 'stream',
      timeout: 15000,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
      },
    });

    // Write stream to temp file
    const writer = fs.createWriteStream(tempFilePath);
    response.data.pipe(writer);

    await new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });

    // Verify downloaded file is not an HTML virus warning page
    const fileHeader = fs.readFileSync(tempFilePath, { encoding: 'utf-8', flag: 'r' }).slice(0, 300);
    if (fileHeader.includes('<!DOCTYPE html>') || fileHeader.includes('<html')) {
      throw new Error(
        'Google Drive returned an HTML page instead of a CSV file. Please make sure the file permissions are set to "Anyone with the link can view".'
      );
    }

    // Run inference
    const inferenceResult = await runPythonInference(tempFilePath);
    inferenceResult.filename = `gdrive_${fileId.substring(0, 8)}.csv`;

    // Save record
    await saveRecord({
      ...inferenceResult,
      source: 'drive',
      driveUrl: driveUrl,
    });

    res.json({
      success: true,
      fileId,
      ...inferenceResult,
    });
  } catch (err) {
    console.error('Google Drive prediction error:', err.message);
    res.status(500).json({
      success: false,
      error: `Google Drive Fetch Failed: ${err.message}`,
    });
  } finally {
    fs.unlink(tempFilePath, () => {});
  }
};

/**
 * GET /api/sample-content/:className
 * Run inference on a preset class sample (car, drone, person)
 */
exports.getSamplePreset = async (req, res) => {
  const { className } = req.params;
  const validClasses = ['car', 'drone', 'person'];

  if (!validClasses.includes(className.toLowerCase())) {
    return res.status(404).json({ success: false, error: 'Target class not found' });
  }

  const folderMap = {
    car: 'Cars',
    drone: 'Drones',
    person: 'People',
  };

  const folderName = folderMap[className.toLowerCase()];
  const rawDir = path.join(PROJECT_ROOT, 'dataset', 'raw', folderName);

  try {
    // Search for any CSV file recursively
    const findCsv = (dir) => {
      if (!fs.existsSync(dir)) return null;
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          const res = findCsv(fullPath);
          if (res) return res;
        } else if (entry.name.endsWith('.csv')) {
          return fullPath;
        }
      }
      return null;
    };

    let samplePath = findCsv(rawDir);
    if (!samplePath) {
      samplePath = findCsv(path.join(PROJECT_ROOT, 'dataset', 'raw', 'data', folderName));
    }

    if (!samplePath) {
      return res.status(404).json({
        success: false,
        error: `No sample CSV found for ${className}`,
      });
    }

    const inferenceResult = await runPythonInference(samplePath);
    inferenceResult.filename = path.basename(samplePath);
    inferenceResult.actual_class = className.toLowerCase();

    // Save to history
    await saveRecord({
      ...inferenceResult,
      source: 'preset',
    });

    res.json({
      success: true,
      ...inferenceResult,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/**
 * GET /api/download-sample/:className
 * Direct download for sample CSV files so users can test local upload
 */
exports.downloadSampleCsv = (req, res) => {
  const { className } = req.params;
  const folderMap = {
    car: 'Cars',
    drone: 'Drones',
    person: 'People',
  };

  const folderName = folderMap[className.toLowerCase()];
  if (!folderName) {
    return res.status(404).send('Class not found');
  }

  const rawDir = path.join(PROJECT_ROOT, 'dataset', 'raw', folderName);

  const findCsv = (dir) => {
    if (!fs.existsSync(dir)) return null;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const res = findCsv(fullPath);
        if (res) return res;
      } else if (entry.name.endsWith('.csv')) {
        return fullPath;
      }
    }
    return null;
  };

  const samplePath = findCsv(rawDir) || findCsv(path.join(PROJECT_ROOT, 'dataset', 'raw', 'data', folderName));

  if (!samplePath || !fs.existsSync(samplePath)) {
    return res.status(404).send('Sample file not found');
  }

  res.setHeader('Content-Disposition', `attachment; filename="radar_${className}_sample.csv"`);
  res.setHeader('Content-Type', 'text/csv');
  fs.createReadStream(samplePath).pipe(res);
};

/**
 * GET /api/history
 * Fetch past classification tests
 */
exports.getHistory = async (req, res) => {
  try {
    if (getDBStatus()) {
      const history = await Classification.find()
        .sort({ createdAt: -1 })
        .limit(20)
        .lean();
      return res.json({ success: true, history, source: 'mongodb' });
    }

    res.json({ success: true, history: inMemoryHistory.slice(0, 20), source: 'memory' });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/**
 * DELETE /api/history
 * Clear classification history
 */
exports.clearHistory = async (req, res) => {
  try {
    if (getDBStatus()) {
      await Classification.deleteMany({});
    }
    inMemoryHistory.length = 0;
    res.json({ success: true, message: 'Classification history cleared' });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/**
 * GET /api/health
 * System diagnostic status
 */
exports.getHealth = (req, res) => {
  res.json({
    status: 'online',
    server: 'Express.js Node.js MERN Backend',
    database: getDBStatus() ? 'MongoDB Connected' : 'In-Memory Mode',
    targetClasses: ['car', 'drone', 'person'],
    modelPath: 'output/models/best_radar_model.pkl',
    defaultGoogleDriveFolder:
      process.env.GOOGLE_DRIVE_FOLDER_URL ||
      'https://drive.google.com/drive/folders/1yashraj18111-radar-samples?usp=sharing',
  });
};
