// server/server.js
// Main entry point for MERN Express backend

const express = require('express');
const cors = require('cors');
const path = require('path');
const dotenv = require('dotenv');
const { connectDB } = require('./config/db');
const radarRoutes = require('./routes/radar');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Connect to MongoDB (graceful - continues in-memory if offline)
connectDB();

// Middlewares
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

// API Routes
app.use('/api', radarRoutes);

// Root Health Endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    project: 'AI-Based Radar Target Classification MERN API',
    endpoints: {
      health: 'GET /api/health',
      predictUpload: 'POST /api/predict (multipart/form-data: file)',
      predictDrive: 'POST /api/drive-predict (json: { driveUrl })',
      getSample: 'GET /api/sample/:className (car, drone, person)',
      downloadSample: 'GET /api/download-sample/:className',
      getHistory: 'GET /api/history',
      clearHistory: 'DELETE /api/history',
    },
  });
});

// Serve frontend build if located in client/dist (for unified production deployment)
const clientBuildPath = path.join(__dirname, '../client/dist');
if (require('fs').existsSync(clientBuildPath)) {
  app.use(express.static(clientBuildPath));
  app.get('*', (req, res) => {
    res.sendFile(path.join(clientBuildPath, 'index.html'));
  });
}

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('Unhandled server error:', err);
  res.status(err.status || 500).json({
    success: false,
    error: err.message || 'Internal Server Error',
  });
});

app.listen(PORT, () => {
  console.log('\x1b[36m%s\x1b[0m', `🚀 Radar MERN Backend running on http://localhost:${PORT}`);
  console.log('\x1b[35m%s\x1b[0m', `📡 Radar Classification Endpoints active on http://localhost:${PORT}/api/`);
});
