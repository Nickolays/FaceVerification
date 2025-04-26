# Stage 1 - Build
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt .
RUN pip install --user --no-cache-dir --upgrade pip
# Install light CPU-only torch
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --user --no-cache-dir -r requirements-service.txt
# Clean cache after installing
RUN pip cache purge

# Copy only needed files (filtered by .dockerignore)
COPY . .

# Stage 2 - Production
FROM python:3.11-slim

WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only the minimal app code
COPY --from=builder /app .

EXPOSE 8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]