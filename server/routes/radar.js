// server/routes/radar.js
// Express API routing for radar classification

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const os = require('os');
const radarController = require('../controllers/radarController');

// Configure Multer storage
const upload = multer({
  dest: path.join(os.tmpdir(), 'radar_uploads'),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext === '.csv' || ext === '.txt') {
      return cb(null, true);
    }
    cb(new Error('Only .csv and .txt radar signal files are supported'));
  },
});

// Routes
router.get('/health', radarController.getHealth);
router.post('/predict', upload.single('file'), radarController.predictUploadedFile);
router.post('/drive-predict', radarController.predictFromGoogleDrive);
router.get('/sample/:className', radarController.getSamplePreset);
router.get('/download-sample/:className', radarController.downloadSampleCsv);
router.get('/history', radarController.getHistory);
router.delete('/history', radarController.clearHistory);

module.exports = router;
