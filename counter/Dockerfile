FROM python:3.11-slim-bookworm

# Install system UI, OpenGL, and window-manager dependencies for OpenCV/Pygame
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libx11-6 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install the CPU-only version of PyTorch first (crucial for ARM64 Pi)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vehicle_counter_pi.py .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "vehicle_counter_pi.py"]