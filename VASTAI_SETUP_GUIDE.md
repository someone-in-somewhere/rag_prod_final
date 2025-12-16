# Hướng Dẫn Setup RAG Chatbot trên Vast.ai GPU Cloud

## Mục Lục
1. [Yêu Cầu Hệ Thống](#1-yêu-cầu-hệ-thống)
2. [Tạo Instance trên Vast.ai](#2-tạo-instance-trên-vastai)
3. [Kết Nối SSH](#3-kết-nối-ssh)
4. [Cài Đặt Tự Động](#4-cài-đặt-tự-động)
5. [Cài Đặt Thủ Công](#5-cài-đặt-thủ-công)
6. [Khởi Động Hệ Thống](#6-khởi-động-hệ-thống)
7. [Truy Cập Web UI](#7-truy-cập-web-ui)
8. [Xử Lý Lỗi](#8-xử-lý-lỗi)
9. [Tối Ưu Chi Phí](#9-tối-ưu-chi-phí)

---

## 1. Yêu Cầu Hệ Thống

### GPU Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| VRAM | 24GB | 48GB+ |
| GPU | RTX 3090 / A5000 | RTX 4090 / A6000 |
| CUDA | 11.8+ | 12.x |

### Disk & RAM
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Disk | 50GB | 100GB |
| RAM | 16GB | 32GB+ |

### Models sử dụng (~35GB total)
- `BAAI/bge-m3`: ~2GB (Embedding)
- `Qwen/Qwen2.5-7B-Instruct`: ~15GB (LLM)
- `Qwen/Qwen2-VL-7B-Instruct`: ~16GB (Vision - optional)

---

## 2. Tạo Instance trên Vast.ai

### Bước 1: Đăng nhập Vast.ai
1. Truy cập [https://vast.ai](https://vast.ai)
2. Đăng nhập hoặc tạo tài khoản
3. Nạp credit (minimum $10)

### Bước 2: Tìm GPU phù hợp
1. Vào **Search** hoặc **Templates**
2. Sử dụng filter:
   - **GPU RAM**: >= 24GB
   - **Disk Space**: >= 50GB
   - **CUDA Version**: >= 11.8
   - **Reliability**: >= 95%

### Bước 3: Chọn Docker Image
Chọn một trong các image sau:

**Option 1: PyTorch + CUDA (Recommended)**
```
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

**Option 2: vLLM ready**
```
vllm/vllm-openai:latest
```

**Option 3: Ubuntu base**
```
nvidia/cuda:12.4.0-devel-ubuntu22.04
```

### Bước 4: Cấu hình Instance
```yaml
# Recommended settings
Disk Space: 80GB
On-start Script: (để trống, setup sau)
Jupyter: Disabled
SSH: Enabled
Direct Port Access: Enabled

# Open ports (quan trọng!)
Ports: 8081, 8000, 6379
```

### Bước 5: Rent Instance
1. Click **RENT**
2. Đợi instance khởi động (2-5 phút)
3. Copy SSH connection string

---

## 3. Kết Nối SSH

### Sử dụng Terminal/PowerShell
```bash
# Format từ Vast.ai
ssh -p <PORT> root@<HOST> -L 8081:localhost:8081

# Ví dụ:
ssh -p 22345 root@ssh5.vast.ai -L 8081:localhost:8081
```

### Sử dụng SSH Key (recommended)
```bash
# Tạo SSH key (chỉ lần đầu)
ssh-keygen -t ed25519

# Copy public key lên Vast.ai Account Settings
cat ~/.ssh/id_ed25519.pub

# Connect
ssh -p <PORT> root@<HOST> -L 8081:localhost:8081
```

### Port Forwarding
Khi SSH với flag `-L 8081:localhost:8081`, bạn có thể truy cập:
- Web UI: `http://localhost:8081` (trên máy local)

---

## 4. Cài Đặt Tự Động

### One-liner Setup
```bash
# Clone và chạy setup script
git clone https://github.com/someone-in-somewhere/rag_prod_final.git ~/rag_prod_final
cd ~/rag_prod_final
chmod +x setup_vastai.sh
./setup_vastai.sh
```

Script sẽ tự động:
1. Kiểm tra GPU và CUDA
2. Cài đặt system packages
3. Cài đặt PyTorch với CUDA
4. Cài đặt PaddlePaddle cho OCR
5. Cài đặt vLLM
6. Cài đặt project dependencies
7. Cấu hình Redis
8. Tạo startup scripts

**Thời gian:** 10-20 phút (tùy internet speed)

---

## 5. Cài Đặt Thủ Công

Nếu script tự động gặp lỗi, làm theo các bước sau:

### 5.1 Cập nhật hệ thống
```bash
apt-get update
apt-get install -y git wget curl htop tmux redis-server \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### 5.2 Clone repository
```bash
cd ~
git clone https://github.com/someone-in-somewhere/rag_prod_final.git
cd rag_prod_final
```

### 5.3 Cài đặt PyTorch
```bash
# Cho CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

### 5.4 Cài đặt PaddlePaddle
```bash
# CUDA 12.x
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Hoặc CPU fallback
pip install paddlepaddle
```

### 5.5 Cài đặt vLLM
```bash
pip install vllm
```

### 5.6 Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 5.7 Tạo thư mục data
```bash
mkdir -p data/{uploads,chroma_db,logs}
```

### 5.8 Cấu hình environment
```bash
cat > .env << EOF
SERVER_HOST=0.0.0.0
SERVER_PORT=8081
VLLM_BASE_URL=http://localhost:8000/v1
REDIS_HOST=localhost
REDIS_PORT=6379
EOF
```

---

## 6. Khởi Động Hệ Thống

### Option A: Khởi động tất cả (Recommended)
```bash
cd ~/rag_prod_final
./start_all.sh
```

### Option B: Khởi động từng service

**Terminal 1 - Redis:**
```bash
redis-server --appendonly yes --dir ~/rag_prod_final/data
```

**Terminal 2 - vLLM:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.8 \
    --port 8000 \
    --trust-remote-code
```

**Terminal 3 - RAG Server:**
```bash
cd ~/rag_prod_final
python server.py
```

### Sử dụng tmux (recommended cho production)
```bash
# Tạo session cho vLLM
tmux new-session -d -s vllm 'vllm serve Qwen/Qwen2.5-7B-Instruct --dtype float16 --port 8000'

# Tạo session cho RAG
tmux new-session -d -s rag 'cd ~/rag_prod_final && python server.py'

# Xem logs
tmux attach -t vllm  # Ctrl+B, D để detach
tmux attach -t rag
```

### Kiểm tra services
```bash
# Health check
curl http://localhost:8081/health

# Stats
curl http://localhost:8081/stats

# vLLM status
curl http://localhost:8000/health
```

---

## 7. Truy Cập Web UI

### Method 1: SSH Port Forwarding (Secure)
```bash
# Từ máy local, SSH với port forwarding
ssh -p <PORT> root@<HOST> -L 8081:localhost:8081

# Mở browser
http://localhost:8081
```

### Method 2: Direct Access (Vast.ai)
1. Vào Vast.ai dashboard
2. Click instance của bạn
3. Trong **Open Ports**, click port 8081
4. Hoặc sử dụng URL: `http://<instance-ip>:8081`

### Method 3: Ngrok (Public Access)
```bash
# Cài ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xf ngrok-v3-stable-linux-amd64.tgz

# Đăng ký tại ngrok.com và lấy authtoken
./ngrok authtoken <YOUR_TOKEN>

# Tạo tunnel
./ngrok http 8081
```

---

## 8. Xử Lý Lỗi

### Lỗi: CUDA out of memory
```bash
# Giảm GPU memory utilization
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.6 \  # Giảm từ 0.8 xuống 0.6
    --max-model-len 2048            # Giảm context length
```

### Lỗi: vLLM không start
```bash
# Kiểm tra CUDA
nvidia-smi

# Clear cache và thử lại
pip cache purge
pip uninstall vllm -y
pip install vllm --no-cache-dir
```

### Lỗi: Redis connection refused
```bash
# Khởi động Redis
redis-server --daemonize yes --appendonly yes

# Kiểm tra
redis-cli ping  # Should return: PONG
```

### Lỗi: Port already in use
```bash
# Tìm process sử dụng port
lsof -i :8081
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Lỗi: Model download chậm/fail
```bash
# Set HuggingFace mirror (nếu ở China)
export HF_ENDPOINT=https://hf-mirror.com

# Hoặc download trước
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### Lỗi: PaddleOCR không cài được
```bash
# Thử phiên bản CPU
pip install paddlepaddle paddleocr

# Hoặc skip OCR (sẽ không process được images)
```

---

## 9. Tối Ưu Chi Phí

### Chọn GPU phù hợp
| GPU | VRAM | Price/hr | Recommendation |
|-----|------|----------|----------------|
| RTX 3090 | 24GB | $0.15-0.25 | Budget option |
| RTX 4090 | 24GB | $0.35-0.50 | Best value |
| A5000 | 24GB | $0.20-0.30 | Stable choice |
| A6000 | 48GB | $0.40-0.60 | For large batches |

### Tips tiết kiệm
1. **Interruptible instances**: Rẻ hơn 50-70% nhưng có thể bị ngắt
2. **Off-peak hours**: Giá rẻ hơn vào ban đêm (UTC)
3. **Snapshot**: Tạo snapshot sau khi setup xong để không phải install lại
4. **Stop khi không dùng**: Pause instance thay vì destroy

### Tạo Snapshot (quan trọng!)
```bash
# Trước khi stop instance, tạo snapshot trên Vast.ai dashboard
# Lần sau chỉ cần restore từ snapshot
```

---

## Quick Reference

### Startup Commands
```bash
cd ~/rag_prod_final
./start_all.sh          # Start everything
./stop_all.sh           # Stop everything
```

### Useful Commands
```bash
# View logs
tmux attach -t vllm     # vLLM logs
tmux attach -t rag      # RAG logs
tail -f data/logs/*.log # File logs

# Health check
curl localhost:8081/health
curl localhost:8000/health

# GPU status
nvidia-smi
watch -n 1 nvidia-smi   # Real-time monitoring
```

### Ports
| Port | Service | Description |
|------|---------|-------------|
| 8081 | RAG Server | Web UI & API |
| 8000 | vLLM | LLM inference |
| 6379 | Redis | Cache & metadata |

---

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề, vui lòng tạo issue tại:
https://github.com/someone-in-somewhere/rag_prod_final/issues
