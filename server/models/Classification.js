// server/models/Classification.js
// Mongoose schema for radar classification records

const mongoose = require('mongoose');

const ClassificationSchema = new mongoose.Schema(
  {
    filename: {
      type: String,
      required: true,
      trim: true,
    },
    source: {
      type: String,
      enum: ['upload', 'drive', 'preset'],
      default: 'upload',
    },
    driveUrl: {
      type: String,
      default: null,
    },
    predictedClass: {
      type: String,
      required: true,
      enum: ['car', 'drone', 'person'],
    },
    confidence: {
      type: Number,
      required: true,
      min: 0,
      max: 100,
    },
    probabilities: {
      car: { type: Number, default: 0 },
      drone: { type: Number, default: 0 },
      person: { type: Number, default: 0 },
    },
    signalStats: {
      matrix_shape: [Number],
      min_power: Number,
      max_power: Number,
      mean_power: Number,
    },
    spectrogramThumbnail: {
      type: String, // base64 data URL
      default: null,
    },
    createdAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    timestamps: true,
  }
);

// Index for fast query of recent records
ClassificationSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Classification', ClassificationSchema);
