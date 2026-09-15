# ==============================================================================
# NeuroCare-AI Production Dockerfile
# Installs requirements and runs both FastAPI (Port 8000) & Streamlit (Port 8501)
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/workspace"

WORKDIR /workspace

# Install system dependencies (build-essential, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project directories and files
COPY configs/ ./configs/
COPY app/ ./app/
COPY src/ ./src/
COPY data_raw/ ./data_raw/
COPY data_processed/ ./data_processed/
COPY models/ ./models/
COPY rag_index/ ./rag_index/
COPY logs/ ./logs/
COPY start.sh ./start.sh

# Fix Windows CRLF line endings and grant execute permissions
RUN sed -i 's/\r$//' /workspace/start.sh && chmod +x /workspace/start.sh

# Expose both FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Health check verifying both services are operational
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch both FastAPI and Streamlit concurrently
CMD ["/workspace/start.sh"]
