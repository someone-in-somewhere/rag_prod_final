# BÁO CÁO DỰ ÁN

# HỆ THỐNG RAG CHATBOT CHO TÀI LIỆU LẬP TRÌNH NHÚNG

**Phiên bản:** 2.0.0
**Ngày báo cáo:** 05/12/2024

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Công nghệ sử dụng](#2-công-nghệ-sử-dụng)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Các thành phần chính](#4-các-thành-phần-chính)
5. [Luồng xử lý dữ liệu](#5-luồng-xử-lý-dữ-liệu)
6. [API Endpoints](#6-api-endpoints)
7. [Cơ sở dữ liệu](#7-cơ-sở-dữ-liệu)
8. [Tính năng nổi bật](#8-tính-năng-nổi-bật)
9. [Cấu hình hệ thống](#9-cấu-hình-hệ-thống)
10. [Thống kê mã nguồn](#10-thống-kê-mã-nguồn)
11. [Hướng dẫn triển khai](#11-hướng-dẫn-triển-khai)
12. [Kết luận](#12-kết-luận)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Giới thiệu

Dự án **Embedded RAG Chatbot** là một hệ thống trả lời câu hỏi thông minh sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)**. Hệ thống được thiết kế đặc biệt cho lĩnh vực **lập trình nhúng và vi điều khiển**, cho phép người dùng tải lên tài liệu kỹ thuật và đặt câu hỏi để nhận được câu trả lời chính xác với nguồn trích dẫn.

### 1.2. Mục tiêu

- Xây dựng hệ thống chatbot có khả năng hiểu và trả lời câu hỏi về tài liệu kỹ thuật
- Hỗ trợ đa định dạng tài liệu: PDF, DOCX, TXT, hình ảnh
- Tích hợp OCR cho nhận dạng văn bản trong hình ảnh
- Hỗ trợ song ngữ Tiếng Việt và Tiếng Anh
- Đảm bảo độ chính xác cao với cơ chế tìm kiếm hybrid (kết hợp dense và sparse)

### 1.3. Phạm vi ứng dụng

| Lĩnh vực | Mô tả |
|----------|-------|
| Lập trình nhúng | Tài liệu vi điều khiển, MCU |
| Giao thức phần cứng | I2C, SPI, UART, GPIO |
| Cấu hình thanh ghi | Register configurations |
| Sơ đồ mạch điện | Circuit diagrams |
| Datasheet | Tài liệu kỹ thuật từ nhà sản xuất |

### 1.4. Cấu trúc thư mục dự án

```
rag_prod_final/
├── Backend (Python)
│   ├── server.py              # FastAPI server & REST endpoints
│   ├── rag_pipeline.py        # Pipeline xử lý RAG
│   ├── document_ingest.py     # Xử lý và phân đoạn tài liệu
│   ├── vectorstore_chroma.py  # Quản lý vector store
│   ├── embedder.py            # Model embedding BGE-M3
│   ├── redis_store.py         # Lưu trữ metadata Redis
│   ├── ocr_utils.py           # OCR và Vision processing
│   └── config.py              # Cấu hình hệ thống
├── Frontend
│   └── index.html             # Giao diện web SPA
├── Infrastructure
│   ├── docker-compose.yml     # Docker cho Redis
│   └── requirements.txt       # Python dependencies
└── Data
    ├── uploads/               # File upload tạm
    ├── chroma_db/             # Vector database
    └── logs/                  # Log hệ thống
```

---

## 2. CÔNG NGHỆ SỬ DỤNG

### 2.1. Backend Technologies

| Thành phần | Công nghệ | Phiên bản | Mục đích |
|------------|-----------|-----------|----------|
| Web Framework | FastAPI | 0.115.0+ | REST API server hiệu năng cao |
| Server | Uvicorn | 0.30.0+ | ASGI server |
| Validation | Pydantic | 2.9.0+ | Kiểm tra dữ liệu request/response |
| LLM Inference | vLLM | - | Tối ưu hóa sinh văn bản |
| LLM Model | Qwen2.5-7B-Instruct | - | Model ngôn ngữ lớn |

### 2.2. Document Processing

| Thành phần | Công nghệ | Mục đích |
|------------|-----------|----------|
| PDF Processing | PyMuPDF (fitz) | Trích xuất text và bảng từ PDF |
| DOCX Processing | python-docx | Xử lý file Word |
| OCR | PaddleOCR | Nhận dạng văn bản từ hình ảnh |
| Vision Model | Qwen2-VL-7B-Instruct | Mô tả hình ảnh kỹ thuật |

### 2.3. Vector & Retrieval

| Thành phần | Công nghệ | Mục đích |
|------------|-----------|----------|
| Embeddings | BGE-M3 (FlagEmbedding) | Tạo vector embedding dense và sparse |
| Vector Database | ChromaDB | Lưu trữ và tìm kiếm vector |
| Hybrid Search | Custom Implementation | Kết hợp dense + sparse search |

### 2.4. Caching & Storage

| Thành phần | Công nghệ | Mục đích |
|------------|-----------|----------|
| Cache | Redis | Lưu metadata và cache query |
| In-Memory Cache | Python LRU Cache | Cache embedding query (1000 entries) |
| ML Framework | PyTorch | Xử lý neural network |

### 2.5. Frontend

| Thành phần | Công nghệ | Mục đích |
|------------|-----------|----------|
| UI | HTML/CSS/JavaScript | Single Page Application |
| API Communication | Fetch API | Giao tiếp REST với backend |
| Styling | CSS3 | Thiết kế responsive |

---

## 3. KIẾN TRÚC HỆ THỐNG

### 3.1. Sơ đồ kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                           │
│  index.html - Chat UI + Document Manager                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP/REST
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FastAPI Server (server.py)                     │
│  ┌─ /upload ──┐  ┌─ /ingest ──┐  ┌─ /chat ──┐  ┌─ /documents ─┐│
│  │ Validation │  │ Chunking   │  │ Query    │  │ Management   ││
│  └────────────┘  └────────────┘  └──────────┘  └──────────────┘│
└──────────────────┬──────────────┬──────────────┬────────────────┘
                   │              │              │
        ┌──────────▼─┐    ┌───────▼──────┐    ┌──▼──────────────┐
        │ Document   │    │ RAG Pipeline │    │ Redis Store     │
        │ Ingest     │    │              │    │ (Metadata)      │
        │            │    │ - Retrieve   │    │                 │
        │ Parse:     │    │ - Prompt     │    │ doc:{doc_id}    │
        │ PDF/DOCX   │    │ - Generate   │    │ doc:meta:{id}   │
        │ TXT/Image  │    │ - Retry      │    │ doc:index       │
        └──────┬─────┘    └───────┬──────┘    └─────────────────┘
               │                  │
        ┌──────▼──────────────────▼───────────────────┐
        │  ChromaDB Vector Store (Hybrid Search)      │
        │  ┌──────────────────────────────────────┐   │
        │  │ Dense Embeddings (BGE-M3, 1024-dim)  │   │
        │  │ + Sparse Index (Token-based)         │   │
        │  │ Hybrid: 70% dense + 30% sparse       │   │
        │  └──────────────────────────────────────┘   │
        └─────────────────────────────────────────────┘
               │
        ┌──────▼─────────────┐
        │ vLLM Server        │
        │ Qwen2.5-7B-Instruct│
        │ (OpenAI compatible)│
        └────────────────────┘
```

### 3.2. Mô hình RAG (Retrieval-Augmented Generation)

```
┌────────────┐    ┌─────────────┐    ┌────────────┐    ┌──────────┐
│ User Query │───▶│  Retriever  │───▶│  Context   │───▶│Generator │
└────────────┘    │  (Hybrid)   │    │  Builder   │    │  (LLM)   │
                  └─────────────┘    └────────────┘    └────┬─────┘
                         │                                   │
                         ▼                                   ▼
                  ┌─────────────┐                   ┌──────────────┐
                  │ ChromaDB    │                   │   Response   │
                  │ + Sparse    │                   │  + Sources   │
                  └─────────────┘                   └──────────────┘
```

---

## 4. CÁC THÀNH PHẦN CHÍNH

### 4.1. server.py - FastAPI REST Server

**Vai trò:** Server HTTP chính và API gateway

**Chức năng:**
- Xử lý các request HTTP/REST từ frontend
- Routing đến các service xử lý tương ứng
- Middleware CORS cho cross-origin requests
- Validation file upload (định dạng, kích thước)
- Hỗ trợ streaming response

**Endpoints chính:**
| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | /upload | Upload file tài liệu |
| POST | /ingest | Xử lý và index tài liệu |
| POST | /chat | Truy vấn chatbot |
| GET | /documents | Danh sách tài liệu |
| DELETE | /documents/{id} | Xóa tài liệu |
| GET | /health | Kiểm tra sức khỏe hệ thống |

### 4.2. rag_pipeline.py - RAG Processing Pipeline

**Vai trò:** Điều phối workflow retrieve-then-generate

**Chức năng:**
- `retrieve_with_cache()`: Tìm kiếm hybrid với caching
- `chat()`: Endpoint chat chính (non-streaming)
- `chat_stream()`: Streaming chat responses
- `detect_language()`: Phát hiện ngôn ngữ (Việt/Anh)
- `build_prompt()`: Xây dựng prompt theo ngữ cảnh
- `generate_with_retry()`: Sinh văn bản với retry logic

**Đặc điểm:**
- Cache kết quả query (tối đa 1000 queries)
- Lọc theo ngưỡng relevance (mặc định 0.4)
- Tự động phát hiện ngôn ngữ
- Retry logic với exponential backoff

### 4.3. document_ingest.py - Document Processing

**Vai trò:** Phân tích và chia nhỏ tài liệu

**Định dạng hỗ trợ:**
- PDF: Trích xuất text và bảng
- DOCX: Paragraphs, tables, images
- TXT: Plain text
- Images: JPG, PNG (OCR + Vision)

**Phương pháp chunking:**
1. **Semantic Chunking:** Tôn trọng cấu trúc tài liệu
   - Nhận dạng headings, code blocks, tables
   - Giữ nguyên các đoạn register description
   - Không cắt giữa code blocks
2. **Simple Chunking:** Chia theo số từ (fallback)

**Quy trình xử lý:**
```
Document → Parse → Detect Boundaries → Create Segments →
Merge Chunks → Add Overlap → Generate Metadata
```

### 4.4. vectorstore_chroma.py - Hybrid Vector Store

**Vai trò:** Quản lý embeddings và tìm kiếm

**Kiến trúc:**
- Primary: ChromaDB (persistent)
- Supplementary: In-memory sparse index

**Hybrid Search:**
- **Dense Search:** Cosine similarity trên BGE-M3 embeddings (1024-dim)
- **Sparse Search:** BM25-like token scoring
- **Trọng số:** 70% dense + 30% sparse (có thể cấu hình)

### 4.5. embedder.py - BGE-M3 Embedding Model

**Vai trò:** Tạo dense và sparse embeddings

**Model:** BAAI/bge-m3

**Khả năng:**
- Dense embeddings: Vector 1024 chiều
- Sparse embeddings: Lexical weights
- Batch processing
- GPU acceleration (CUDA)

### 4.6. redis_store.py - Document Metadata Storage

**Vai trò:** Lưu trữ metadata và document gốc

**Cấu trúc dữ liệu:**
| Key Pattern | Kiểu | Nội dung |
|-------------|------|----------|
| `doc:{doc_id}` | String | JSON đầy đủ của document |
| `doc:meta:{doc_id}` | Hash | Metadata nhẹ |
| `doc:index` | Set | Tập hợp tất cả doc_id |

### 4.7. ocr_utils.py - Image Processing

**OCR Engine:** PaddleOCR
- Hỗ trợ tiếng Việt
- Xử lý văn bản xoay
- Lọc theo confidence (threshold 0.5)

**Vision Captioning:** Qwen2-VL-7B-Instruct
- Hiểu hình ảnh kỹ thuật
- Prompt song ngữ
- Model persist trong VRAM

### 4.8. config.py - Configuration Management

**Quản lý cấu hình qua environment variables:**
- Paths: Upload, ChromaDB, Logs
- Redis: Host, port, connection pool
- Models: Embedding, Vision, LLM
- Retrieval: Top-K, threshold, weights
- Chunking: Size, overlap, semantic toggle

---

## 5. LUỒNG XỬ LÝ DỮ LIỆU

### 5.1. Luồng Upload và Index Tài liệu

```
┌──────────────┐
│ User Upload  │
│ (PDF/DOCX/   │
│  TXT/Image)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────────┐
│ [POST /upload]│────▶│ Validate      │
│              │     │ - Format      │
│              │     │ - Size (50MB) │
└──────────────┘     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │ Save to       │
                     │ /data/uploads/│
                     └───────┬───────┘
                             │
                             ▼
┌──────────────┐     ┌───────────────┐
│[POST /ingest]│────▶│ Parse Document│
└──────────────┘     │ - PDF: PyMuPDF│
                     │ - DOCX: python│
                     │ - Image: OCR  │
                     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │ Semantic      │
                     │ Chunking      │
                     │ (512 words)   │
                     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │ Generate      │
                     │ Embeddings    │
                     │ (BGE-M3)      │
                     └───────┬───────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
      ┌───────────────┐              ┌───────────────┐
      │ Store in      │              │ Store in      │
      │ ChromaDB      │              │ Redis         │
      │ (vectors)     │              │ (metadata)    │
      └───────────────┘              └───────────────┘
```

### 5.2. Luồng Truy vấn Chat

```
┌──────────────┐
│ User Query   │
│ "Cách cấu    │
│ hình I2C?"   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────────┐
│ [POST /chat] │────▶│ Language      │
│              │     │ Detection     │
│              │     │ (VI/EN)       │
└──────────────┘     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │ Check Cache   │
                     │ (MD5 hash)    │
                     └───────┬───────┘
                             │
              ┌──────────────┴──────────────┐
              │                              │
      ┌───────▼───────┐              ┌───────▼───────┐
      │ Cache HIT     │              │ Cache MISS    │
      │ Return cached │              │               │
      └───────────────┘              └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Embed Query   │
                                     │ (dense+sparse)│
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Hybrid Search │
                                     │ 70% dense     │
                                     │ 30% sparse    │
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Filter by     │
                                     │ Relevance     │
                                     │ (threshold:0.4)│
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Build Prompt  │
                                     │ + Context     │
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Call vLLM     │
                                     │ Qwen2.5-7B    │
                                     │ (retry: 3x)   │
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Cache Result  │
                                     └───────┬───────┘
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │ Response +    │
                                     │ Sources +     │
                                     │ Timing        │
                                     └───────────────┘
```

---

## 6. API ENDPOINTS

### 6.1. Document Management APIs

#### POST /upload
**Mô tả:** Upload file tài liệu

**Request:**
```
Content-Type: multipart/form-data
file: <binary>
```

**Response:**
```json
{
  "filename": "stm32_datasheet.pdf",
  "path": "/data/uploads/stm32_datasheet.pdf",
  "size_mb": 2.5,
  "status": "uploaded"
}
```

#### POST /ingest
**Mô tả:** Xử lý và index tài liệu vào vector store

**Request:**
```json
{
  "filename": "stm32_datasheet.pdf",
  "use_semantic_chunking": true
}
```

**Response:**
```json
{
  "doc_id": "abc123",
  "chunk_count": 45,
  "status": "indexed",
  "chunking_method": "semantic"
}
```

#### GET /documents
**Mô tả:** Lấy danh sách tất cả tài liệu

**Response:**
```json
{
  "documents": [
    {
      "id": "abc123",
      "filename": "stm32_datasheet.pdf",
      "doc_type": "pdf",
      "chunk_count": 45
    }
  ],
  "count": 1
}
```

#### DELETE /documents/{doc_id}
**Mô tả:** Xóa tài liệu khỏi hệ thống

**Response:**
```json
{
  "status": "deleted",
  "chunks_deleted": 45
}
```

### 6.2. Chat APIs

#### POST /chat
**Mô tả:** Gửi câu hỏi và nhận câu trả lời

**Request:**
```json
{
  "query": "Cách cấu hình giao tiếp I2C trên STM32?",
  "top_k": 5,
  "max_tokens": 1024,
  "stream": false,
  "use_hybrid": true
}
```

**Response:**
```json
{
  "query": "Cách cấu hình giao tiếp I2C trên STM32?",
  "response": "Để cấu hình I2C trên STM32, bạn cần...",
  "language": "vi",
  "sources": [
    {
      "source": "stm32_datasheet.pdf",
      "chunk_index": 12,
      "relevance": 0.85,
      "text_preview": "I2C Configuration..."
    }
  ],
  "context_used": true,
  "retrieval_info": {
    "docs_found": 5,
    "docs_relevant": 3,
    "timing_ms": 245,
    "hybrid_search": true
  }
}
```

### 6.3. System APIs

#### GET /health
**Mô tả:** Kiểm tra sức khỏe hệ thống

**Response:**
```json
{
  "status": "healthy",
  "vectorstore": {
    "total_chunks": 150,
    "documents": 3
  },
  "redis": "connected",
  "document_count": 3
}
```

#### GET /stats
**Mô tả:** Thống kê hệ thống

#### GET /limits
**Mô tả:** Lấy cấu hình giới hạn

**Response:**
```json
{
  "max_pdf_pages": 100,
  "max_file_size_mb": 50,
  "supported_formats": ["pdf", "docx", "txt", "jpg", "png"]
}
```

#### POST /cache/clear
**Mô tả:** Xóa cache query

---

## 7. CƠ SỞ DỮ LIỆU

### 7.1. ChromaDB (Vector Database)

**Mục đích:** Lưu trữ và tìm kiếm vector embeddings

**Đặc điểm:**
- Persistent storage tại `/data/chroma_db/`
- Collection: `embedded_docs`
- Index: HNSW với cosine similarity
- Dense vectors: 1024 chiều
- Sparse vectors: JSON trong metadata

**Cấu trúc document:**
```python
{
    "id": "chunk_unique_id",
    "embedding": [0.1, 0.2, ...],  # 1024 dimensions
    "metadata": {
        "source": "filename.pdf",
        "chunk_index": 0,
        "doc_type": "pdf",
        "sparse_vector": {"token": weight, ...}
    },
    "document": "Nội dung chunk..."
}
```

### 7.2. Redis (Metadata Store)

**Mục đích:** Lưu trữ metadata và document gốc

**Đặc điểm:**
- Docker container với persistence AOF
- Connection pooling (10 connections)
- Auto-reconnect

**Data Structures:**

| Key | Type | Nội dung |
|-----|------|----------|
| `doc:{doc_id}` | String | JSON document đầy đủ |
| `doc:meta:{doc_id}` | Hash | Metadata nhẹ (filename, type, chunks) |
| `doc:index` | Set | Danh sách tất cả doc_id |

### 7.3. Local Filesystem

**Mục đích:** Lưu file upload tạm thời

**Cấu trúc:**
```
/data/
├── uploads/      # File upload tạm
├── chroma_db/    # Vector database
└── logs/         # Application logs
```

---

## 8. TÍNH NĂNG NỔI BẬT

### 8.1. Hybrid Search (Tìm kiếm Hybrid)

Kết hợp hai phương pháp tìm kiếm:

| Phương pháp | Trọng số | Ưu điểm |
|-------------|----------|---------|
| Dense Search | 70% | Hiểu ngữ nghĩa, synonym |
| Sparse Search | 30% | Chính xác với keyword, technical terms |

**Công thức:**
```
final_score = 0.7 * dense_score + 0.3 * sparse_score
```

### 8.2. Semantic Chunking

Chia tài liệu thông minh theo cấu trúc:

- Nhận dạng headings và sections
- Giữ nguyên code blocks
- Tôn trọng bảng và register descriptions
- Overlap 50 từ giữa các chunks

### 8.3. Multilingual Support

**Tiếng Việt:**
- OCR với PaddleOCR (lang='vi')
- System prompt tiếng Việt
- Auto-detect qua ký tự Unicode

**Tiếng Anh:**
- Hỗ trợ đầy đủ
- System prompt tiếng Anh

### 8.4. Vision & OCR

**OCR (PaddleOCR):**
- Nhận dạng văn bản trong hình ảnh
- Hỗ trợ tiếng Việt với dấu
- Xử lý văn bản xoay

**Vision (Qwen2-VL):**
- Mô tả hình ảnh kỹ thuật
- Hiểu sơ đồ mạch điện
- Nhận dạng components

### 8.5. Caching System

**Query Cache:**
- In-memory LRU cache
- Capacity: 1000 queries
- Key: MD5 hash của query

**Embedding Cache:**
- Cache query embeddings
- Giảm latency cho queries lặp lại

### 8.6. Reliability Features

**Retry Logic:**
- Max 3 lần retry cho LLM calls
- Exponential backoff

**Connection Pool:**
- Redis connection pooling
- Auto-reconnect

**Health Monitoring:**
- `/health` endpoint
- Stats collection

---

## 9. CẤU HÌNH HỆ THỐNG

### 9.1. Các tham số chính

| Parameter | Giá trị mặc định | Mô tả |
|-----------|-----------------|-------|
| CHUNK_SIZE | 512 | Số từ mỗi chunk |
| CHUNK_OVERLAP | 50 | Số từ overlap giữa chunks |
| TOP_K | 5 | Số documents retrieve |
| RELEVANCE_THRESHOLD | 0.4 | Ngưỡng relevance tối thiểu |
| DENSE_WEIGHT | 0.7 | Trọng số dense search |
| SPARSE_WEIGHT | 0.3 | Trọng số sparse search |
| MAX_TOKENS | 1024 | Độ dài output LLM |
| TEMPERATURE | 0.7 | Độ sáng tạo của LLM |
| MAX_RETRIES | 3 | Số lần retry LLM |
| MAX_FILE_SIZE_MB | 50 | Giới hạn kích thước file |
| MAX_PDF_PAGES | 100 | Giới hạn số trang PDF |
| QUERY_CACHE_SIZE | 1000 | Kích thước cache query |
| SERVER_PORT | 8081 | Port server |

### 9.2. Environment Variables

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Models
EMBEDDING_MODEL=BAAI/bge-m3
VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# vLLM Server
VLLM_URL=http://localhost:8000/v1

# Paths
UPLOAD_DIR=/data/uploads
CHROMA_DIR=/data/chroma_db
LOG_DIR=/data/logs
```

---

## 10. THỐNG KÊ MÃ NGUỒN

### 10.1. Lines of Code

| File | Số dòng | Chức năng |
|------|---------|-----------|
| server.py | 361 | FastAPI endpoints |
| rag_pipeline.py | 332 | RAG orchestration |
| document_ingest.py | 522 | Document processing |
| vectorstore_chroma.py | 250 | Vector store |
| redis_store.py | 190 | Redis storage |
| ocr_utils.py | 173 | OCR & Vision |
| embedder.py | 71 | BGE-M3 embedding |
| config.py | 91 | Configuration |
| **Backend Total** | **1,990** | - |
| index.html | 420 | Frontend SPA |
| **Grand Total** | **~2,410** | - |

### 10.2. Dependencies

**Core Dependencies:**
- fastapi >= 0.115.0
- uvicorn >= 0.30.0
- pydantic >= 2.9.0
- chromadb >= 0.5.0
- redis >= 5.0.0
- torch >= 2.4.0
- FlagEmbedding >= 1.2.0
- PyMuPDF >= 1.25.0
- python-docx >= 1.1.0
- paddleocr >= 3.0.0
- transformers >= 4.40.0

---

## 11. HƯỚNG DẪN TRIỂN KHAI

### 11.1. Yêu cầu hệ thống

**Hardware:**
- CPU: 4+ cores
- RAM: 16GB+ (32GB recommended)
- GPU: NVIDIA với 16GB+ VRAM (cho LLM và Vision)
- Storage: 50GB+ SSD

**Software:**
- Python 3.10+
- Docker & Docker Compose
- CUDA 11.8+ (cho GPU)
- vLLM server riêng biệt

### 11.2. Các bước triển khai

**Bước 1: Clone repository**
```bash
git clone <repository-url>
cd rag_prod_final
```

**Bước 2: Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

**Bước 3: Khởi động Redis**
```bash
docker-compose up -d
```

**Bước 4: Khởi động vLLM server** (terminal riêng)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

**Bước 5: Khởi động RAG server**
```bash
python server.py
```

**Bước 6: Truy cập giao diện**
```
http://localhost:8081
```

### 11.3. Docker Deployment (Production)

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  rag-server:
    build: .
    ports:
      - "8081:8081"
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
    volumes:
      - ./data:/data

volumes:
  redis_data:
```

---

## 12. KẾT LUẬN

### 12.1. Tổng kết

Hệ thống **Embedded RAG Chatbot** đã được xây dựng thành công với các chức năng chính:

- **Chatbot hỏi đáp thông minh:** Người dùng có thể đặt câu hỏi và nhận câu trả lời dựa trên nội dung tài liệu đã tải lên
- **Hỗ trợ nhiều loại file:** PDF, Word, văn bản thuần và hình ảnh
- **Nhận dạng chữ trong ảnh:** Tự động đọc văn bản từ hình ảnh kỹ thuật
- **Hỗ trợ tiếng Việt:** Cả giao diện và nội dung đều xử lý tốt tiếng Việt

### 12.2. Những gì đã đạt được

| Mục tiêu | Kết quả |
|----------|---------|
| Tải và xử lý tài liệu | Hoàn thành |
| Tìm kiếm nội dung liên quan | Hoàn thành |
| Sinh câu trả lời tự động | Hoàn thành |
| Giao diện web dễ sử dụng | Hoàn thành |
| Hỗ trợ tiếng Việt | Hoàn thành |

### 12.3. Hướng phát triển tiếp theo

**Ngắn hạn:**
- Thêm đăng nhập/đăng ký người dùng
- Cải thiện giao diện đẹp và dễ dùng hơn
- Hỗ trợ thêm file Excel, PowerPoint

**Dài hạn:**
- Cho phép nhiều người dùng cùng lúc
- Thêm tính năng lưu lịch sử hội thoại
- Triển khai lên cloud để sử dụng từ xa
- Tích hợp với các ứng dụng khác qua API

---

## PHỤ LỤC

### A. Glossary

| Thuật ngữ | Định nghĩa |
|-----------|------------|
| RAG | Retrieval-Augmented Generation - Kỹ thuật kết hợp retrieval và generation |
| Embedding | Vector biểu diễn semantic của văn bản |
| Chunking | Chia tài liệu thành các đoạn nhỏ |
| Dense Search | Tìm kiếm dựa trên vector embedding |
| Sparse Search | Tìm kiếm dựa trên keyword/token |
| LLM | Large Language Model - Model ngôn ngữ lớn |
| OCR | Optical Character Recognition - Nhận dạng ký tự quang học |

### B. References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- ChromaDB Documentation: https://docs.trychroma.com/
- BGE-M3 Model: https://huggingface.co/BAAI/bge-m3
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- vLLM: https://github.com/vllm-project/vllm

---

*Báo cáo được tạo tự động từ phân tích mã nguồn dự án.*
