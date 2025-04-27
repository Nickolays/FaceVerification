# Stage 1: Base Image
FROM arm64v8/python:3.10 AS base

WORKDIR /app

# Copy only required files
COPY RaspberryPI/requirements.txt .
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .
COPY .env .

# Install dependencies (CPU optimized Torch)
# RUN pip install --no-cache-dir \
#       --extra-index-url https://www.piwheels.org/simple \
#       torch torchvision
RUN pip install --no-cache-dir \
      https://github.com/Qengineering/PyTorch-Raspberry-Pi-64-OS/releases/download/v2.0.0/torch-2.0.0-cp39-cp39-linux_aarch64.whl \
      https://github.com/Qengineering/PyTorch-Raspberry-Pi-64-OS/releases/download/v2.0.0/torchvision-0.15.2-cp39-cp39-linux_aarch64.whl
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image (smaller)    # balenalib/raspberrypi4-64-python:3.10-slim 
FROM arm64v8/python:3.10 

WORKDIR /app

# Copy installed libraries from builder
COPY --from=base /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=base /app /app

# Expose port from .env (set at runtime)
ENV PORT=${PORT}

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]