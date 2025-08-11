# --------- Minimal runtime image ----------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV (headless) needs glib; nothing else
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY flask4/requirements.txt ./requirements.txt

# Make sure requirements.txt uses opencv-python-headless and no duplicate mlflow_skinny
# Example:
# Flask==3.1.1
# mlflow==2.9.2
# numpy==1.23.5
# opencv-python-headless==4.11.0.86
# pandas==1.5.3
# tensorflow==2.10.1
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY flask4/ .

EXPOSE 5000

# --------- Default (dev) entrypoint ----------
# Runs Flask directly (fine for testing / small loads)
CMD ["python", "app.py"]

# =========================
# Production (optional)
# =========================
# Uncomment to use Gunicorn in production.
# 1) Add gunicorn to requirements.txt or install here:
# RUN pip install --no-cache-dir gunicorn
#
# 2) Switch CMD to Gunicorn (uses your factory create_app()):
# CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", \
#      "--timeout", "120", "--bind", "0.0.0.0:5000", "app:create_app()"]
#
# Notes:
# - Increase workers/threads for more concurrency if CPU allows.
# - If behind a proxy/load balancer, consider:
#   ENV FORWARDED_ALLOW_IPS="*"
#   and adding --forwarded-allow-ips="*"
#
# Optional healthcheck:
# HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
#   CMD wget -qO- http://127.0.0.1:5000/ || exit 1

# =========================
# End Production section
# =========================
