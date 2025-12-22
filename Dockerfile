# -------------------------
# Base image
# -------------------------
FROM python:3.10-slim

# -------------------------
# Install system dependencies (REQUIRED for LightGBM)
# -------------------------
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Environment settings
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy application code
# -------------------------
COPY api ./api
COPY src ./src
COPY data ./data

# -------------------------
# Expose API port
# -------------------------
EXPOSE 8000

# -------------------------
# Run FastAPI
# -------------------------
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
