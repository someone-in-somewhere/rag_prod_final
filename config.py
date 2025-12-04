"""Cấu hình hệ thống RAG cho Embedded Programming"""
"""config.py"""
import os
from pathlib import Path

# ============================================
# Paths
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOG_DIR = DATA_DIR / "logs"

# Tạo thư mục
for dir_path in [UPLOAD_DIR, CHROMA_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# Redis
# ============================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", 10))

# ============================================
# ChromaDB
# ============================================
CHROMA_COLLECTION = "embedded_docs"

# ============================================
# Models
# ============================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
VISION_MODEL = os.getenv("VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# ============================================
# vLLM Server
# ============================================
VLLM_BASE_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", 60))

# ============================================
# Chunking
# ============================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() == "true"

# ============================================
# Retrieval
# ============================================
TOP_K = int(os.getenv("TOP_K", 5))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.4))

# Hybrid search weights (dense + sparse = 1.0)
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.7))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.3))

# ============================================
# Caching
# ============================================
QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", 1000))
ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"

# ============================================
# File Limits
# ============================================
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 100))      # Giới hạn cho cả PDF và DOCX
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 20))

# ============================================
# Generation
# ============================================
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))

# ============================================
# Server
# ============================================
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8081))

# ============================================
# Logging
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")