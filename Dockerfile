# Production Dockerfile for Render Web Service (MERN + PyTorch)
FROM python:3.11-slim

# Install system dependencies & Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python ML dependencies (CPU-optimized for cloud)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Install Node.js backend dependencies
COPY server/package*.json ./server/
WORKDIR /app/server
RUN npm install --production

# Copy application source files
WORKDIR /app
COPY config.py .
COPY model ./model
COPY preprocessing ./preprocessing
COPY ml_engine ./ml_engine
COPY dataset/raw ./dataset/raw
COPY server ./server
COPY client/dist ./client/dist

# Expose Render port
ENV PORT=5000
ENV NODE_ENV=production
ENV PYTHON_CMD=python3
EXPOSE 5000

# Start Express server
WORKDIR /app/server
CMD ["node", "server.js"]
