# -----------------------------
# Quantum-Classical-Hybrid Dockerfile
# -----------------------------

# Base image with CUDA 12.2
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install build tools, Python, and OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    libomp-dev \
    curl \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libicu-dev \
    libssl-dev \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11 globally
RUN pip3 install --no-cache-dir pybind11

# Set working directory
WORKDIR /workspace

# Copy all project files, excluding build/ and .venv
COPY . /workspace

# Optional: clean old build
RUN rm -rf /workspace/build

# Create fresh build directory
RUN mkdir -p /workspace/build
WORKDIR /workspace/build

# Configure the project
RUN cmake /workspace

# Build the project
RUN make -j$(nproc)

# Default command: interactive bash for testing
CMD ["/bin/bash"]
