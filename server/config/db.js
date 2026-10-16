// server/config/db.js
// MongoDB connection configuration with graceful fallback

const mongoose = require('mongoose');

let isConnected = false;

const connectDB = async () => {
  const mongoURI = process.env.MONGODB_URI;

  if (!mongoURI) {
    console.warn('\x1b[33m%s\x1b[0m', '⚠️  MONGODB_URI not found in environment variables.');
    console.warn('\x1b[33m%s\x1b[0m', 'ℹ️  Operating in memory-history mode. Set MONGODB_URI to persist classifications.');
    return false;
  }

  try {
    const conn = await mongoose.connect(mongoURI, {
      serverSelectionTimeoutMS: 5000,
    });
    isConnected = true;
    console.log('\x1b[32m%s\x1b[0m', `✅ MongoDB Connected: ${conn.connection.host}`);
    return true;
  } catch (error) {
    console.error('\x1b[31m%s\x1b[0m', `❌ MongoDB Connection Failed: ${error.message}`);
    console.warn('\x1b[33m%s\x1b[0m', 'ℹ️  Falling back to in-memory history. App will continue running.');
    return false;
  }
};

const getDBStatus = () => isConnected;

module.exports = { connectDB, getDBStatus };
