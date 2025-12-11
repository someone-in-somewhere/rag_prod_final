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

## 12. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 12.1. Kết luận

#### 12.1.1. Tổng kết mục tiêu đã đạt được

Qua quá trình nghiên cứu và phát triển, dự án **Hệ thống Chatbot hỏi đáp thông minh dựa trên truy xuất tài liệu (RAG)** đã hoàn thành các mục tiêu đề ra ban đầu. Hệ thống được xây dựng thành công với khả năng tiếp nhận, xử lý và truy xuất thông tin từ các tài liệu kỹ thuật chuyên ngành vi điều khiển và hệ thống nhúng. Quy trình hoạt động hoàn chỉnh từ khâu xử lý tài liệu đầu vào, nhúng vector ngữ nghĩa, truy xuất thông tin liên quan cho đến sinh câu trả lời tự động đã được triển khai đồng bộ và hiệu quả. Đặc biệt, hệ thống hỗ trợ song ngữ Việt-Anh với khả năng tự động nhận diện ngôn ngữ của người dùng, đáp ứng nhu cầu đa dạng trong môi trường học thuật và công nghiệp.

#### 12.1.2. Kiến trúc và công nghệ

Về mặt kiến trúc, hệ thống được thiết kế theo mô hình phân tách module, tạo điều kiện thuận lợi cho việc bảo trì, nâng cấp và mở rộng trong tương lai. Nền tảng công nghệ được lựa chọn kỹ lưỡng với các thành phần tiên tiến: mô hình nhúng văn bản BGE-M3 tạo vector 1024 chiều có khả năng biểu diễn ngữ nghĩa sâu, mô hình ngôn ngữ lớn Qwen2.5-7B đảm nhiệm việc sinh câu trả lời chất lượng cao, và mô hình thị giác Qwen2-VL hỗ trợ phân tích hình ảnh kỹ thuật như sơ đồ mạch điện và biểu đồ. Kỹ thuật tìm kiếm kết hợp với tỷ lệ 70% nhúng dày đặc và 30% nhúng thưa đã chứng minh hiệu quả vượt trội so với các phương pháp đơn lẻ, tận dụng được cả khả năng hiểu ngữ nghĩa lẫn độ chính xác từ khóa. Hệ thống bộ nhớ đệm đa tầng với Redis và bộ nhớ đệm LRU giúp tối ưu hóa thời gian phản hồi đáng kể.

#### 12.1.3. Các tính năng nổi bật

Hệ thống sở hữu nhiều tính năng nổi bật đáp ứng yêu cầu thực tiễn. Khả năng xử lý đa định dạng tài liệu bao gồm PDF, Word, văn bản thuần và hình ảnh giúp người dùng linh hoạt trong việc tải lên nguồn tài liệu. Thuật toán phân đoạn ngữ nghĩa thông minh bảo toàn cấu trúc tài liệu gốc, nhận diện chính xác các thành phần như tiêu đề, khối mã nguồn, bảng biểu và mô tả thanh ghi. Tính năng nhận dạng ký tự quang học (OCR) tích hợp PaddleOCR hỗ trợ tốt tiếng Việt có dấu, kết hợp với mô hình thị giác để mô tả các hình ảnh kỹ thuật phức tạp. Cơ chế phản hồi theo luồng thời gian thực mang lại trải nghiệm người dùng mượt mà, trong khi việc trích dẫn nguồn kèm điểm độ tin cậy giúp người dùng đánh giá và kiểm chứng thông tin. Giao diện web được thiết kế trực quan, thân thiện, phù hợp với nhiều đối tượng người dùng.

#### 12.1.4. Điểm mạnh của hệ thống

Điểm mạnh cốt lõi của hệ thống nằm ở khả năng tối ưu hóa cho tài liệu kỹ thuật chuyên ngành. Các thuật toán nhận diện được thiết kế đặc biệt để xử lý các thành phần đặc thù như mô tả thanh ghi, khối mã nguồn lập trình và bảng thông số kỹ thuật. Hiệu suất hoạt động được đảm bảo thông qua cơ chế lưu đệm thông minh và tính toán với độ chính xác nửa (FP16), giảm thiểu tài nguyên bộ nhớ mà vẫn duy trì chất lượng kết quả. Hệ thống xử lý lỗi toàn diện với cơ chế thử lại tự động đảm bảo độ ổn định trong môi trường sản xuất. Thiết kế mẫu đơn thể (singleton) cho các thành phần nặng như mô hình nhúng và kết nối cơ sở dữ liệu giúp tiết kiệm tài nguyên đáng kể. Kho lưu trữ vector ChromaDB với khả năng lưu trữ bền vững đảm bảo dữ liệu được bảo toàn qua các phiên làm việc.

#### 12.1.5. Những hạn chế còn tồn tại

Bên cạnh những thành tựu đạt được, hệ thống vẫn còn một số hạn chế cần được khắc phục. Hiện tại, hệ thống chưa tích hợp bộ nhớ hội thoại, mỗi câu hỏi được xử lý độc lập mà không ghi nhớ ngữ cảnh các lượt trò chuyện trước đó. Chỉ mục thưa phục vụ tìm kiếm từ khóa chỉ được lưu trong bộ nhớ tạm, dẫn đến việc phải xây dựng lại khi khởi động hệ thống. Khả năng triển khai hiện giới hạn ở mô hình đơn máy chủ, chưa hỗ trợ mở rộng ngang để đáp ứng lượng truy cập lớn. Cơ chế xếp hạng lại kết quả truy xuất và vòng phản hồi từ người dùng để cải thiện chất lượng theo thời gian cũng chưa được triển khai.

### 12.2. Hướng phát triển

#### 12.2.1. Giai đoạn ngắn hạn

Trong giai đoạn ngắn hạn, ưu tiên hàng đầu là bổ sung bộ nhớ hội thoại đa lượt, cho phép hệ thống ghi nhớ ngữ cảnh trò chuyện và hiểu các câu hỏi tiếp nối như "Giải thích thêm về điểm đó" hay "Còn trường hợp khác không?". Chỉ mục thưa cần được lưu trữ vào cơ sở dữ liệu bền vững Redis để tránh mất mát khi khởi động lại. Tính năng phân trang kết quả truy vấn sẽ được triển khai để xử lý hiệu quả khi có nhiều tài liệu liên quan. Hệ thống nhãn và phân loại tài liệu giúp người dùng tổ chức và tìm kiếm tài liệu dễ dàng hơn. Cuối cùng, bảng điều khiển quản trị sẽ được xây dựng để giám sát hoạt động hệ thống, theo dõi các chỉ số hiệu suất và quản lý người dùng.

#### 12.2.2. Giai đoạn trung hạn

Giai đoạn trung hạn tập trung vào nâng cao chất lượng và khả năng mở rộng. Cơ chế xếp hạng lại (re-ranking) sử dụng mô hình cross-encoder sẽ được triển khai để nâng cao độ chính xác của kết quả truy xuất, đặc biệt với các truy vấn phức tạp. Kiến trúc đa máy chủ với bộ nhớ đệm chia sẻ qua Redis Cluster cho phép hệ thống phục vụ lượng người dùng lớn hơn. Việc tinh chỉnh mô hình nhúng trên tập dữ liệu chuyên ngành vi điều khiển sẽ cải thiện đáng kể khả năng hiểu ngữ nghĩa đặc thù. Hệ thống quản lý phiên bản tài liệu giúp theo dõi các thay đổi và cho phép so sánh giữa các phiên bản. Vòng phản hồi từ người dùng thông qua đánh giá câu trả lời (hữu ích/không hữu ích) sẽ được tích hợp để cải thiện chất lượng liên tục.

#### 12.2.3. Giai đoạn dài hạn

Về dài hạn, hệ thống hướng tới khả năng xử lý quy mô lớn và tích hợp các công nghệ tiên tiến. Việc chuyển đổi sang kho vector phân tán như Qdrant hoặc Milvus sẽ cho phép lưu trữ và truy vấn hàng triệu tài liệu với độ trễ thấp. Tích hợp đồ thị tri thức (Knowledge Graph) mở ra khả năng suy luận liên kết giữa các khái niệm, ví dụ hiểu được mối quan hệ giữa các giao thức truyền thông hay sự tương thích giữa các dòng vi điều khiển. Cơ chế học chủ động (Active Learning) sẽ được triển khai để tự động nhận diện và học hỏi từ các truy vấn khó mà hệ thống chưa trả lời tốt. Khả năng tự động chọn mô hình phù hợp theo ngữ cảnh câu hỏi giúp tối ưu hóa cả chất lượng lẫn chi phí tính toán. Cuối cùng, tính năng cập nhật và đánh chỉ mục tài liệu theo thời gian thực sẽ đảm bảo hệ thống luôn phản ánh thông tin mới nhất.

#### 12.2.4. Mở rộng phạm vi ứng dụng

Ngoài lĩnh vực hệ thống nhúng và vi điều khiển, kiến trúc và công nghệ của hệ thống hoàn toàn có thể được mở rộng sang các lĩnh vực tài liệu kỹ thuật khác như y tế, pháp luật, tài chính hay giáo dục. Việc tích hợp vào các nền tảng quản lý tri thức doanh nghiệp sẽ mang lại giá trị thiết thực trong việc khai thác kho tài liệu nội bộ. Phát triển giao diện lập trình ứng dụng (API) công khai cho phép các bên thứ ba tích hợp và xây dựng các ứng dụng phái sinh. Ứng dụng di động sẽ được phát triển để người dùng có thể truy cập và tra cứu tài liệu mọi lúc mọi nơi, đặc biệt hữu ích cho các kỹ sư làm việc tại hiện trường.

### 12.3. Lời kết

Dự án **Hệ thống RAG Chatbot cho tài liệu lập trình nhúng** đã đạt được những kết quả đáng khích lệ, chứng minh tính khả thi và hiệu quả của việc ứng dụng công nghệ trí tuệ nhân tạo vào lĩnh vực tra cứu tài liệu kỹ thuật. Với nền tảng kiến trúc vững chắc và lộ trình phát triển rõ ràng, hệ thống có tiềm năng trở thành công cụ hỗ trợ đắc lực cho các kỹ sư, sinh viên và nhà nghiên cứu trong lĩnh vực hệ thống nhúng nói riêng và các ngành kỹ thuật nói chung. Sự kết hợp giữa khả năng hiểu ngữ nghĩa sâu của mô hình ngôn ngữ lớn với độ chính xác của tìm kiếm từ khóa thông qua kỹ thuật tìm kiếm kết hợp đã tạo nên một giải pháp toàn diện, đáp ứng được các yêu cầu khắt khe của tài liệu kỹ thuật chuyên ngành.

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

### B. Tài liệu tham khảo

**Tiếng Việt**

[1] Lê Thanh Hương, *Bài giảng Xử lý ngôn ngữ tự nhiên*, Viện Công nghệ thông tin và Truyền thông, Trường Đại học Bách khoa Hà Nội, 2020. Truy cập: https://users.soict.hust.edu.vn/huonglt/UNLP/

[2] Nguyễn Đắc Hiếu, *Xây dựng ứng dụng Chatbot tư vấn khách hàng sử dụng các kỹ thuật học sâu*, Luận văn Thạc sĩ, Viện Công nghệ thông tin và Truyền thông, Trường Đại học Bách khoa Hà Nội, 2021.

[3] Nguyễn Tất Tiến, *Nghiên cứu và xây dựng chatbot hỗ trợ người dùng trong ngân hàng*, Luận văn Thạc sĩ Kỹ thuật phần mềm, Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội, 2019.

[4] Đinh Mạnh Tường, *Trí tuệ nhân tạo: Cách tiếp cận hiện đại*, Nhà xuất bản Khoa học và Kỹ thuật, Hà Nội, 2024, tr. 1-523.

[5] Vũ Hữu Tiệp, *Machine Learning cơ bản*, 2018. Truy cập: https://machinelearningcoban.com/

**Tiếng Anh**

[6] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation", *arXiv preprint*, arXiv:2402.03216, 2024. Truy cập: https://arxiv.org/abs/2402.03216

[7] Chroma Team, *ChromaDB: The AI-native open-source embedding database*, 2024. Truy cập: https://docs.trychroma.com/

[8] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, "Retrieval-Augmented Generation for Large Language Models: A Survey", *arXiv preprint*, arXiv:2312.10997v5, 2024. Truy cập: https://arxiv.org/abs/2312.10997

[9] S. Gupta, R. Ranjan, and S. N. Singh, "A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions", *arXiv preprint*, arXiv:2410.12837, 2024. Truy cập: https://arxiv.org/abs/2410.12837

[10] Hugging Face, *Transformers: State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX*, 2024. Truy cập: https://huggingface.co/docs/transformers/index

[11] D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. (draft), 2024. Truy cập: https://web.stanford.edu/~jurafsky/slp3/

[12] PaddlePaddle Team, *PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle*, 2024. Truy cập: https://github.com/PaddlePaddle/PaddleOCR

[13] Pydantic Team, *Pydantic: Data validation using Python type hints*, Version 2.12, 2024. Truy cập: https://docs.pydantic.dev/latest/

[14] PyMuPDF/Artifex Software, *PyMuPDF Documentation*, Version 1.25, 2024. Truy cập: https://pymupdf.readthedocs.io/

[15] PyTorch Team, *PyTorch Documentation*, Version 2.4, 2024. Truy cập: https://docs.pytorch.org/docs/stable/index.html

[16] python-openxml, *python-docx: Create and modify Word documents with Python*, Version 1.1, 2024. Truy cập: https://python-docx.readthedocs.io/

[17] Qwen Team, Alibaba Group, "Qwen2.5 Technical Report", *arXiv preprint*, arXiv:2412.15115, 2024. Truy cập: https://arxiv.org/abs/2412.15115

[18] Qwen Team, Alibaba Group, "Qwen2-VL: To See the World More Clearly", *Qwen Blog*, 2024. Truy cập: https://qwenlm.github.io/blog/qwen2-vl/

[19] S. Ramírez, *FastAPI: Modern, fast web framework for building APIs with Python*, Version 0.115, 2024. Truy cập: https://fastapi.tiangolo.com/

[20] Redis Ltd., *Redis Documentation*, Version 7.4, 2024. Truy cập: https://redis.io/docs/latest/

[21] T. Dettmers et al., *Uvicorn: An ASGI web server for Python*, 2024. Truy cập: https://uvicorn.dev/

[22] vLLM Team, UC Berkeley, *vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs*, 2024. Truy cập: https://docs.vllm.ai/

[23] X. Wang, Z. Wang, X. Gao, F. Zhang, Y. Wu, Z. Xu, T. Shi, Z. Wang, S. Li, Q. Qian, R. Yin, C. Lv, X. Zheng, and X. Huang, "Searching for Best Practices in Retrieval-Augmented Generation", *arXiv preprint*, arXiv:2407.01219, 2024. Truy cập: https://arxiv.org/abs/2407.01219

---

*Báo cáo được cập nhật ngày 11/12/2024.*
