# LibroBot VLA Docker Configuration
# Build: docker build -t librobot-vla .
# Run: docker run --gpus all -p 8000:8000 librobot-vla
# Optimized for RTX 5090 + AMD 9950X or multi-GPU (4x L20) setups

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.9.0 with CUDA 13.0 support
RUN pip3 install --break-system-packages --no-cache-dir \
    torch==2.9.0+cu130 \
    torchvision==0.24.0+cu130 \
    torchaudio==2.9.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Copy requirements first for better caching
COPY requirements/base.txt requirements/base.txt
COPY requirements/train.txt requirements/train.txt
COPY requirements/inference.txt requirements/inference.txt

# Install Python dependencies
RUN pip3 install --break-system-packages --no-cache-dir -r requirements/base.txt
RUN pip3 install --break-system-packages --no-cache-dir -r requirements/inference.txt

# Copy the library
COPY . .

# Install the package
RUN pip3 install --break-system-packages -e .

# Expose port for inference server
EXPOSE 8000

# Default command
CMD ["python", "-m", "librobot.cli.serve", "--host", "0.0.0.0", "--port", "8000"]
