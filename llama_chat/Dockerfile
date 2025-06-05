FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads static/outputs cache/analysis cache/chat cache/overlay logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV MODEL_CACHE_DIR=/app/model_cache
ENV LOG_LEVEL=INFO

# Volume for persistent data
VOLUME ["/app/static/outputs", "/app/cache", "/app/logs"]

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 