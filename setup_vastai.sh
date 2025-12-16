#!/bin/bash
# =============================================================================
# VAST.AI GPU Cloud Setup Script for RAG Chatbot System
# =============================================================================
# Script này tự động setup hệ thống RAG trên Vast.ai GPU instance
# Yêu cầu: Instance với GPU NVIDIA có ít nhất 24GB VRAM (RTX 3090/4090/A5000+)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_DIR="${HOME}/rag_prod_final"
DATA_DIR="${PROJECT_DIR}/data"
VLLM_PORT=8000
SERVER_PORT=8081
REDIS_PORT=6379

# Model configurations
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL="BAAI/bge-m3"

# =============================================================================
# CHECK SYSTEM REQUIREMENTS
# =============================================================================
print_step "Kiểm tra yêu cầu hệ thống..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Đang chạy với quyền root"
fi

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "Không tìm thấy NVIDIA GPU driver!"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
print_success "Phát hiện GPU: ${GPU_NAME} - ${GPU_MEMORY}"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    print_warning "CUDA toolkit không có trong PATH, nhưng có thể vẫn hoạt động"
fi

# =============================================================================
# SYSTEM UPDATE & BASIC PACKAGES
# =============================================================================
print_step "Cập nhật hệ thống và cài đặt packages cơ bản..."

apt-get update -qq
apt-get install -y -qq \
    git \
    wget \
    curl \
    htop \
    tmux \
    redis-server \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    > /dev/null 2>&1

print_success "Đã cài đặt packages cơ bản"

# =============================================================================
# PYTHON ENVIRONMENT SETUP
# =============================================================================
print_step "Thiết lập Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: ${PYTHON_VERSION}"

# Upgrade pip
pip install --upgrade pip -q

# =============================================================================
# CLONE REPOSITORY (nếu chưa có)
# =============================================================================
print_step "Kiểm tra và clone repository..."

if [ -d "$PROJECT_DIR" ]; then
    print_warning "Thư mục project đã tồn tại tại ${PROJECT_DIR}"
    cd "$PROJECT_DIR"
    print_step "Đang pull latest changes..."
    git pull origin main || true
else
    print_step "Cloning repository..."
    git clone https://github.com/someone-in-somewhere/rag_prod_final.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# =============================================================================
# CREATE DATA DIRECTORIES
# =============================================================================
print_step "Tạo thư mục data..."

mkdir -p "${DATA_DIR}/uploads"
mkdir -p "${DATA_DIR}/chroma_db"
mkdir -p "${DATA_DIR}/logs"
mkdir -p "${DATA_DIR}/models"

print_success "Đã tạo thư mục data"

# =============================================================================
# INSTALL PYTORCH WITH CUDA
# =============================================================================
print_step "Cài đặt PyTorch với CUDA support..."

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
print_success "Phát hiện CUDA version: ${CUDA_VERSION}"

# Install PyTorch based on CUDA version
if [[ "${CUDA_VERSION}" == "12"* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
elif [[ "${CUDA_VERSION}" == "11"* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
else
    print_warning "CUDA version không xác định, cài đặt PyTorch mặc định..."
    pip install torch torchvision torchaudio -q
fi

print_success "Đã cài đặt PyTorch"

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# =============================================================================
# INSTALL PADDLEPADDLE FOR OCR
# =============================================================================
print_step "Cài đặt PaddlePaddle cho OCR..."

if [[ "${CUDA_VERSION}" == "12"* ]]; then
    pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ -q || \
    pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ -q || \
    pip install paddlepaddle -q
else
    pip install paddlepaddle-gpu -q || pip install paddlepaddle -q
fi

print_success "Đã cài đặt PaddlePaddle"

# =============================================================================
# INSTALL VLLM
# =============================================================================
print_step "Cài đặt vLLM..."

pip install vllm -q
print_success "Đã cài đặt vLLM"

# =============================================================================
# INSTALL PROJECT REQUIREMENTS
# =============================================================================
print_step "Cài đặt project dependencies..."

cd "$PROJECT_DIR"
pip install -r requirements.txt -q

print_success "Đã cài đặt tất cả dependencies"

# =============================================================================
# CONFIGURE REDIS
# =============================================================================
print_step "Cấu hình Redis..."

# Stop existing Redis if running
systemctl stop redis-server 2>/dev/null || true
pkill redis-server 2>/dev/null || true

# Start Redis with persistence
redis-server --daemonize yes \
    --port ${REDIS_PORT} \
    --appendonly yes \
    --dir "${DATA_DIR}" \
    --logfile "${DATA_DIR}/logs/redis.log"

sleep 2

# Verify Redis
if redis-cli ping | grep -q "PONG"; then
    print_success "Redis đang chạy trên port ${REDIS_PORT}"
else
    print_error "Không thể khởi động Redis!"
fi

# =============================================================================
# PRE-DOWNLOAD MODELS
# =============================================================================
print_step "Pre-download models (có thể mất 15-30 phút lần đầu)..."

# Download embedding model
python3 -c "
from FlagEmbedding import BGEM3FlagModel
print('Downloading BGE-M3 embedding model...')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
print('BGE-M3 model loaded successfully!')
del model
import torch
torch.cuda.empty_cache()
" || print_warning "Không thể pre-load embedding model, sẽ download khi chạy server"

print_success "Hoàn thành pre-download models"

# =============================================================================
# CREATE STARTUP SCRIPTS
# =============================================================================
print_step "Tạo startup scripts..."

# Create vLLM startup script
cat > "${PROJECT_DIR}/start_vllm.sh" << 'EOF'
#!/bin/bash
# Start vLLM server for LLM inference

MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${GPU_MEMORY_UTIL:-0.8}"

echo "Starting vLLM server..."
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "GPU Memory Utilization: ${GPU_UTIL}"

vllm serve "${MODEL}" \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization ${GPU_UTIL} \
    --port ${PORT} \
    --trust-remote-code \
    --max-model-len 4096
EOF
chmod +x "${PROJECT_DIR}/start_vllm.sh"

# Create RAG server startup script
cat > "${PROJECT_DIR}/start_server.sh" << 'EOF'
#!/bin/bash
# Start RAG FastAPI server

cd "$(dirname "$0")"
PORT="${SERVER_PORT:-8081}"

echo "Starting RAG server on port ${PORT}..."
python3 server.py
EOF
chmod +x "${PROJECT_DIR}/start_server.sh"

# Create full system startup script
cat > "${PROJECT_DIR}/start_all.sh" << 'EOF'
#!/bin/bash
# Start all services for RAG system

PROJECT_DIR="$(dirname "$0")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "RAG System Startup Script"
echo "=========================================="

# Start Redis if not running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "[1/3] Starting Redis..."
    redis-server --daemonize yes --appendonly yes --dir "${PROJECT_DIR}/data"
    sleep 2
else
    echo "[1/3] Redis already running"
fi

# Start vLLM in background using tmux
echo "[2/3] Starting vLLM server in tmux session 'vllm'..."
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "${PROJECT_DIR}/start_vllm.sh"
echo "    Waiting for vLLM to load model (60-120 seconds)..."
sleep 60

# Check if vLLM is ready
for i in {1..12}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "    vLLM server is ready!"
        break
    fi
    echo "    Still waiting... (${i}/12)"
    sleep 10
done

# Start RAG server in background using tmux
echo "[3/3] Starting RAG server in tmux session 'rag'..."
tmux kill-session -t rag 2>/dev/null || true
tmux new-session -d -s rag "${PROJECT_DIR}/start_server.sh"
sleep 5

echo ""
echo "=========================================="
echo "SYSTEM STARTED!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - Redis: localhost:6379"
echo "  - vLLM:  localhost:8000"
echo "  - RAG:   localhost:8081"
echo ""
echo "Access UI: http://<your-vast-ai-ip>:8081"
echo ""
echo "Manage sessions:"
echo "  - View vLLM logs: tmux attach -t vllm"
echo "  - View RAG logs:  tmux attach -t rag"
echo "  - Detach:         Ctrl+B, then D"
echo ""
echo "Health check:"
echo "  curl http://localhost:8081/health"
echo "=========================================="
EOF
chmod +x "${PROJECT_DIR}/start_all.sh"

# Create stop script
cat > "${PROJECT_DIR}/stop_all.sh" << 'EOF'
#!/bin/bash
# Stop all RAG services

echo "Stopping RAG services..."

tmux kill-session -t rag 2>/dev/null && echo "Stopped RAG server"
tmux kill-session -t vllm 2>/dev/null && echo "Stopped vLLM server"
redis-cli shutdown 2>/dev/null && echo "Stopped Redis"

echo "All services stopped."
EOF
chmod +x "${PROJECT_DIR}/stop_all.sh"

print_success "Đã tạo startup scripts"

# =============================================================================
# CREATE ENVIRONMENT FILE
# =============================================================================
print_step "Tạo file environment..."

cat > "${PROJECT_DIR}/.env" << EOF
# RAG System Configuration for Vast.ai

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8081
LOG_LEVEL=INFO

# vLLM
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_TIMEOUT=120

# Models
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Paths
DATA_DIR=${DATA_DIR}

# Retrieval settings
TOP_K=5
RELEVANCE_THRESHOLD=0.4
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3

# Generation
MAX_TOKENS=1024
TEMPERATURE=0.7
EOF

print_success "Đã tạo file .env"

# =============================================================================
# PRINT FINAL INSTRUCTIONS
# =============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}SETUP HOÀN TẤT!${NC}"
echo "=============================================="
echo ""
echo "Các bước tiếp theo:"
echo ""
echo "1. Khởi động toàn bộ hệ thống:"
echo "   cd ${PROJECT_DIR}"
echo "   ./start_all.sh"
echo ""
echo "2. Hoặc khởi động từng service:"
echo "   # Terminal 1 - vLLM"
echo "   ./start_vllm.sh"
echo ""
echo "   # Terminal 2 - RAG Server"
echo "   ./start_server.sh"
echo ""
echo "3. Truy cập Web UI:"
echo "   http://<your-vast-ai-ip>:8081"
echo ""
echo "4. Port forwarding (nếu cần):"
echo "   - Port 8081: RAG Web UI"
echo "   - Port 8000: vLLM API (optional)"
echo ""
echo "5. Monitor logs:"
echo "   tmux attach -t vllm  # xem vLLM logs"
echo "   tmux attach -t rag   # xem RAG logs"
echo ""
echo "6. Health check:"
echo "   curl http://localhost:8081/health"
echo ""
echo "=============================================="
echo -e "${YELLOW}LƯU Ý QUAN TRỌNG:${NC}"
echo "=============================================="
echo "- Lần đầu chạy vLLM sẽ download model (~15GB)"
echo "- Cần ít nhất 24GB VRAM cho tất cả models"
echo "- Đảm bảo port 8081 được open trong Vast.ai"
echo "=============================================="
