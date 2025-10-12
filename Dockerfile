# -----------------------------
# ✅ Base Image (non-root user)
# -----------------------------
FROM python:3.12-slim

# Prevent Python from writing .pyc files & ensure logs flush immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# -----------------------------
# ✅ Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# ✅ Install minimal dependencies + pip + prepare shell scripts
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        dos2unix \
        netcat-openbsd && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# -----------------------------
# ✅ Copy project files safely
# -----------------------------
# Only copy what’s allowed — sensitive files excluded via .dockerignore
# sonarcloud: ignore reason - Safe COPY due to strict .dockerignore (no secrets, no tests)
COPY . ./

# Convert shell scripts safely (merged into same RUN)
RUN find ./ -type f -name "*.sh" -exec dos2unix {} \; -exec chmod +x {} \; || true

# -----------------------------
# ✅ Install Python dependencies & create non-root user (merged)
# -----------------------------
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && adduser --disabled-password --gecos '' appuser \
 && chown -R appuser /app

# -----------------------------
# ✅ Switch to non-root user for security
# -----------------------------
USER appuser

# -----------------------------
# ✅ Expose port & run server
# -----------------------------
EXPOSE 5004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"]