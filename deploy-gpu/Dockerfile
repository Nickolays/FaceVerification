# Dockerfile for GPU-enabled FastAPI app
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python 3 and pip
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch 2.6.0 with CUDA 12.4 support (per PyTorch docs&#8203;:contentReference[oaicite:1]{index=1})
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy dependencies file and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
# COPY ..src/ ./src/
# COPY ..models/ ./models/
# COPY main.py .
COPY ../main.py .
COPY ../src/ ./src/
COPY ../models/ ./models/

# Expose the port (use ${PORT} from environment in docker-compose)
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
