# =========================================================================
# ResumeAI — Production Dockerfile
# Practical 2 (2CEAI702 - MLOps): Dockerizing the ML System
# =========================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

# Set container working directory
WORKDIR /app

# Install curl for container health check and system utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies only (from pinned requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create dedicated non-root application user and persistent instance directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/instance /app/uploads /app/model && \
    chown -R appuser:appuser /app

# Copy application source code and artifacts
COPY --chown=appuser:appuser app.py config.py ./
COPY --chown=appuser:appuser database/ ./database/
COPY --chown=appuser:appuser routes/ ./routes/
COPY --chown=appuser:appuser services/ ./services/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser model/resume_classifier.joblib ./model/

# Switch to unprivileged non-root user
USER appuser

# Expose production port
EXPOSE 5000

# Container healthcheck querying existing /health route
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run Flask via Gunicorn WSGI production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
