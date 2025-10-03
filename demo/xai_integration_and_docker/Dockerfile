# Multi-stage build for Angular + Flask integration
FROM node:18-alpine AS angular-build

# Working directory for Angular build
WORKDIR /app

# Copy Angular package files from local frontend
COPY frontend/package*.json ./

# Install Angular dependencies using clean CI install
RUN npm ci

# Copy Angular source and build for production
COPY frontend/ ./

# Build Angular for production
RUN npm run build

# ---------- Stage 2: Python Flask runtime ----------
FROM python:3.11-slim
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Set working directory
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy the Flask API and backend files
COPY backend/ ./backend/

# Copy built Angular files from previous stage
# angular.json outputs to /app/static; Flask expects /app/static (see backend/api_xAI.py)
RUN mkdir -p /app/static
COPY --from=angular-build /app/dist/browser /app/static/

# Create runtime data directories (mounted via volumes in compose)
RUN mkdir -p /app/backend/models /app/backend/datasets

# Environment configuration
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Healthcheck - uses swagger.json which is always available when API is up
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/swagger.json || exit 1

# Run the application
CMD ["python", "backend/api_xAI.py"]
