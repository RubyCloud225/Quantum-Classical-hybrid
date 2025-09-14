# How to Run Quantum-Classical-Hybrid on a VM with GPU Support

Since M4 Mac cannot run CUDA natively, this project requires a virtual machine or cloud instance with NVIDIA GPU support.

## Prerequisites
- Access to a VM/cloud instance with NVIDIA GPU (e.g., AWS EC2 p3, Google Cloud with GPU)
- Ubuntu 22.04 or compatible Linux distribution
- NVIDIA drivers compatible with CUDA 12.2

## Setup Instructions

### 1. Install NVIDIA Drivers and Docker
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (version 525 or compatible with CUDA 12.2)
sudo apt install nvidia-driver-525 -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Clone Repository and Build
```bash
git clone https://github.com/yourusername/Quantum-Classical-hybrid.git
cd Quantum-Classical-hybrid
sudo docker build -t quantum-classical-hybrid .
```

### 3. Run Container with GPU
```bash
sudo docker run --gpus all -it quantum-classical-hybrid
```

### 4. Verify Setup and Test
Inside the container:
```bash
# Check GPU status
nvidia-smi

# Navigate to build directory
cd /workspace/build

# Build the project
make -j$(nproc)

# Test CUDA functionality (run the test_echo executable)
./test_echo

# Additional verification commands:
# Check CUDA version
nvcc --version

# List CUDA devices
nvidia-smi -L

# Test CUDA compilation
echo "#include <cuda_runtime.h>
#include <iostream>
int main() { int deviceCount; cudaGetDeviceCount(&deviceCount); std::cout << 'CUDA devices: ' << deviceCount << std::endl; return 0; }" > test_cuda.cpp && nvcc test_cuda.cpp -o test_cuda && ./test_cuda
```

## Cloud Options
- **AWS EC2:** Use p3.2xlarge or similar GPU instance
- **Google Cloud:** Use instances with NVIDIA Tesla T4/V100
- **Azure:** Use NC series VMs

## Troubleshooting
- Ensure NVIDIA drivers match CUDA version
- Restart after driver installation
- Use `nvidia-docker` for GPU passthrough
