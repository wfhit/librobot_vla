# LibroBot VLA Docker Configuration
# Build: docker build -t librobot-vla .
# Run: docker run --gpus all -p 8000:8000 librobot-vla

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements/base.txt requirements/base.txt
COPY requirements/train.txt requirements/train.txt
COPY requirements/inference.txt requirements/inference.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/base.txt
RUN pip install --no-cache-dir -r requirements/inference.txt

# Copy the library
COPY . .

# Install the package
RUN pip install -e .

# Expose port for inference server
EXPOSE 8000

# Default command
CMD ["python", "-m", "librobot.cli.serve", "--host", "0.0.0.0", "--port", "8000"]
