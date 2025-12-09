# Kịch Bản Kiểm Thử Hệ Thống RAG Chatbot

Tài liệu này mô tả các kịch bản kiểm thử cho hệ thống RAG Chatbot từ mức độ đơn giản đến phức tạp.

**Hệ thống**: RAG Chatbot cho Embedded Programming
**Base URL**: `http://localhost:8081`
**Ngày tạo**: 2025-12-09

---

## Mục Lục

1. [Cấp độ 1: Kiểm thử Cơ bản (Basic)](#cấp-độ-1-kiểm-thử-cơ-bản-basic)
2. [Cấp độ 2: Kiểm thử Trung bình (Intermediate)](#cấp-độ-2-kiểm-thử-trung-bình-intermediate)
3. [Cấp độ 3: Kiểm thử Nâng cao (Advanced)](#cấp-độ-3-kiểm-thử-nâng-cao-advanced)
4. [Cấp độ 4: Kiểm thử Hiệu năng & Bảo mật](#cấp-độ-4-kiểm-thử-hiệu-năng--bảo-mật)

---

## Cấp độ 1: Kiểm thử Cơ bản (Basic)

### TC-B01: Health Check
**Mục đích**: Kiểm tra hệ thống hoạt động bình thường
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `GET /health` |
| Precondition | Server đang chạy |

**Các bước thực hiện**:
1. Gửi request GET đến `/health`

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response chứa thông tin trạng thái hệ thống

```bash
curl -X GET http://localhost:8081/health
```

---

### TC-B02: Lấy danh sách tài liệu (trống)
**Mục đích**: Kiểm tra API lấy danh sách tài liệu khi chưa có dữ liệu
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `GET /documents` |
| Precondition | Hệ thống mới khởi động, chưa có tài liệu |

**Các bước thực hiện**:
1. Gửi request GET đến `/documents`

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response trả về danh sách rỗng hoặc thông báo không có tài liệu

```bash
curl -X GET http://localhost:8081/documents
```

---

### TC-B03: Upload file TXT đơn giản
**Mục đích**: Kiểm tra chức năng upload file cơ bản
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /upload` |
| Content-Type | `multipart/form-data` |
| File | `test.txt` (< 1MB) |

**Preconditions**:
- Chuẩn bị file `test.txt` với nội dung đơn giản về embedded systems

**Các bước thực hiện**:
1. Tạo file `test.txt` với nội dung:
   ```
   GPIO là viết tắt của General Purpose Input/Output.
   STM32 là dòng vi điều khiển 32-bit của STMicroelectronics.
   I2C là giao thức truyền thông 2 dây.
   ```
2. Upload file lên server

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response chứa thông tin file đã upload (filename, size)

```bash
curl -X POST http://localhost:8081/upload \
  -F "file=@test.txt"
```

---

### TC-B04: Ingest tài liệu đã upload
**Mục đích**: Kiểm tra chức năng ingest (xử lý và lưu trữ) tài liệu
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /ingest` |
| Content-Type | `application/json` |
| Precondition | File đã được upload (TC-B03) |

**Các bước thực hiện**:
1. Gửi request ingest với filename từ TC-B03

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response chứa số lượng chunks đã tạo
- Tài liệu xuất hiện trong `/documents`

```bash
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "test.txt"}'
```

---

### TC-B05: Chat query đơn giản (Tiếng Việt)
**Mục đích**: Kiểm tra chức năng chat cơ bản với câu hỏi tiếng Việt
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /chat` |
| Content-Type | `application/json` |
| Precondition | Đã ingest ít nhất 1 tài liệu |

**Các bước thực hiện**:
1. Gửi câu hỏi đơn giản liên quan đến tài liệu đã ingest

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response chứa câu trả lời liên quan
- Có thông tin `sources` với tài liệu nguồn
- Field `language` = `"vi"`

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "GPIO là gì?"}'
```

---

### TC-B06: Chat query đơn giản (Tiếng Anh)
**Mục đích**: Kiểm tra chức năng chat với câu hỏi tiếng Anh
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /chat` |
| Precondition | Đã ingest tài liệu có nội dung tiếng Anh |

**Các bước thực hiện**:
1. Gửi câu hỏi tiếng Anh

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response trả lời bằng tiếng Anh
- Field `language` = `"en"`

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is GPIO?"}'
```

---

### TC-B07: Xem thống kê hệ thống
**Mục đích**: Kiểm tra API thống kê
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `GET /stats` |

**Các bước thực hiện**:
1. Gửi request GET đến `/stats`

**Kết quả mong đợi**:
- Status code: `200 OK`
- Response chứa thông tin thống kê (số tài liệu, số chunks, cache info)

```bash
curl -X GET http://localhost:8081/stats
```

---

### TC-B08: Xóa tài liệu
**Mục đích**: Kiểm tra chức năng xóa tài liệu
**Độ khó**: ⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `DELETE /documents/{doc_id}` |
| Precondition | Có tài liệu trong hệ thống |

**Các bước thực hiện**:
1. Lấy doc_id từ `/documents`
2. Gửi request DELETE

**Kết quả mong đợi**:
- Status code: `200 OK`
- Tài liệu không còn trong `/documents`

```bash
curl -X DELETE http://localhost:8081/documents/{doc_id}
```

---

## Cấp độ 2: Kiểm thử Trung bình (Intermediate)

### TC-M01: Upload và Ingest file PDF
**Mục đích**: Kiểm tra xử lý file PDF với text và bảng
**Độ khó**: ⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /upload`, `POST /ingest` |
| File | PDF có text + tables (< 10MB, < 50 pages) |

**Preconditions**:
- Chuẩn bị file PDF chứa:
  - Văn bản thông thường
  - Ít nhất 1 bảng dữ liệu
  - Các heading/tiêu đề

**Các bước thực hiện**:
1. Upload file PDF
2. Ingest với semantic chunking enabled
3. Kiểm tra số chunks được tạo

**Kết quả mong đợi**:
- Upload thành công
- Ingest tạo ra nhiều chunks (tùy độ dài tài liệu)
- Bảng được extract đúng

```bash
# Upload
curl -X POST http://localhost:8081/upload -F "file=@stm32_guide.pdf"

# Ingest với semantic chunking
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "stm32_guide.pdf", "use_semantic_chunking": true}'
```

---

### TC-M02: Upload và Ingest file DOCX
**Mục đích**: Kiểm tra xử lý file Word với hình ảnh nhúng
**Độ khó**: ⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| File | DOCX có paragraphs + tables + embedded images |

**Preconditions**:
- Chuẩn bị file DOCX chứa:
  - Đoạn văn bản
  - Bảng dữ liệu
  - Hình ảnh minh họa (sơ đồ mạch, flowchart)

**Các bước thực hiện**:
1. Upload file DOCX
2. Ingest tài liệu
3. Query về nội dung trong hình ảnh

**Kết quả mong đợi**:
- Text và tables được extract
- Hình ảnh được OCR/caption (nếu có vision model)

```bash
curl -X POST http://localhost:8081/upload -F "file=@tutorial.docx"
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "tutorial.docx"}'
```

---

### TC-M03: Upload và xử lý hình ảnh với OCR
**Mục đích**: Kiểm tra OCR trên hình ảnh chứa text
**Độ khó**: ⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| File | JPG/PNG chứa text (datasheet scan, sơ đồ có chú thích) |

**Preconditions**:
- Chuẩn bị hình ảnh:
  - Scan của datasheet page
  - Hoặc sơ đồ mạch có nhãn text

**Các bước thực hiện**:
1. Upload hình ảnh
2. Ingest với OCR
3. Query về nội dung trong hình

**Kết quả mong đợi**:
- OCR nhận diện text trong hình
- Có thể query được nội dung

```bash
curl -X POST http://localhost:8081/upload -F "file=@schematic.png"
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "schematic.png"}'
```

---

### TC-M04: Hybrid Search - Kiểm tra kết hợp Dense và Sparse
**Mục đích**: Kiểm tra hybrid search hoạt động đúng
**Độ khó**: ⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /chat` |
| Parameter | `use_hybrid: true` |

**Preconditions**:
- Đã ingest tài liệu chứa các thuật ngữ kỹ thuật đặc thù (VD: "STM32F103C8T6", "I2C_CR1", "0x40005400")

**Các bước thực hiện**:
1. Query với thuật ngữ kỹ thuật chính xác
2. So sánh kết quả với `use_hybrid: true` và `use_hybrid: false`

**Kết quả mong đợi**:
- Hybrid search tìm được kết quả tốt hơn cho thuật ngữ kỹ thuật
- Response chứa `hybrid_search: true` trong retrieval_info

```bash
# Hybrid search (default)
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Địa chỉ thanh ghi I2C_CR1 là gì?", "use_hybrid": true}'

# Dense only
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Địa chỉ thanh ghi I2C_CR1 là gì?", "use_hybrid": false}'
```

---

### TC-M05: Kiểm tra Source Attribution
**Mục đích**: Xác nhận sources được trả về chính xác
**Độ khó**: ⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Precondition | Ingest nhiều tài liệu khác nhau |

**Các bước thực hiện**:
1. Ingest 3 tài liệu với nội dung khác nhau:
   - `gpio_guide.txt`: Về GPIO
   - `i2c_tutorial.txt`: Về I2C
   - `spi_reference.txt`: Về SPI
2. Query về I2C cụ thể

**Kết quả mong đợi**:
- Sources chỉ chứa `i2c_tutorial.txt` (hoặc chủ yếu)
- Không mix sources không liên quan

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Cách cấu hình I2C master mode?"}'
```

**Kiểm tra response**:
```json
{
  "sources": [
    {"source": "i2c_tutorial.txt", "score": 0.85, "chunk_index": 2}
  ]
}
```

---

### TC-M06: Kiểm tra Query Caching
**Mục đích**: Xác nhận cache hoạt động đúng
**Độ khó**: ⭐⭐

**Các bước thực hiện**:
1. Gửi query lần 1, ghi nhận thời gian response
2. Gửi query giống hệt lần 2
3. So sánh thời gian

**Kết quả mong đợi**:
- Lần 2 nhanh hơn đáng kể (cache hit)
- `retrieval_info` có thể chỉ ra cache được sử dụng

```bash
# Query lần 1
time curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "GPIO output mode là gì?"}'

# Query lần 2 (giống hệt)
time curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "GPIO output mode là gì?"}'
```

---

### TC-M07: Clear Cache
**Mục đích**: Kiểm tra chức năng xóa cache
**Độ khó**: ⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Endpoint | `POST /cache/clear` |

**Các bước thực hiện**:
1. Gửi query để tạo cache
2. Clear cache
3. Gửi query giống để verify cache đã xóa

**Kết quả mong đợi**:
- Clear thành công
- Query sau đó chậm như lần đầu (cache miss)

```bash
curl -X POST http://localhost:8081/cache/clear
```

---

### TC-M08: Tùy chỉnh Top-K Results
**Mục đích**: Kiểm tra parameter top_k
**Độ khó**: ⭐⭐

**Các bước thực hiện**:
1. Query với `top_k: 3`
2. Query với `top_k: 10`
3. So sánh số sources trả về

**Kết quả mong đợi**:
- Số sources tương ứng với top_k (có thể ít hơn nếu không đủ relevant)

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Các chế độ hoạt động của timer?", "top_k": 3}'

curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Các chế độ hoạt động của timer?", "top_k": 10}'
```

---

### TC-M09: Semantic Chunking vs Simple Chunking
**Mục đích**: So sánh chất lượng chunking
**Độ khó**: ⭐⭐

**Preconditions**:
- Chuẩn bị tài liệu có cấu trúc rõ ràng (headings, code blocks, tables)

**Các bước thực hiện**:
1. Upload cùng 1 file 2 lần với tên khác
2. Ingest lần 1 với `use_semantic_chunking: true`
3. Ingest lần 2 với `use_semantic_chunking: false`
4. So sánh số chunks và chất lượng response

**Kết quả mong đợi**:
- Semantic chunking giữ nguyên code blocks, tables
- Simple chunking có thể cắt giữa các logical units

```bash
# Semantic chunking
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "doc_semantic.txt", "use_semantic_chunking": true}'

# Simple chunking
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "doc_simple.txt", "use_semantic_chunking": false}'
```

---

## Cấp độ 3: Kiểm thử Nâng cao (Advanced)

### TC-A01: Query không có context liên quan
**Mục đích**: Kiểm tra xử lý khi không tìm thấy context phù hợp
**Độ khó**: ⭐⭐⭐

**Preconditions**:
- Hệ thống chỉ có tài liệu về STM32

**Các bước thực hiện**:
1. Query về topic hoàn toàn khác (VD: "Cách nấu phở")

**Kết quả mong đợi**:
- Hệ thống nhận ra không có context liên quan
- Trả về thông báo phù hợp thay vì hallucinate
- `context_used: false` trong response

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Cách nấu phở bò?"}'
```

---

### TC-A02: Relevance Threshold Testing
**Mục đích**: Kiểm tra ngưỡng relevance score
**Độ khó**: ⭐⭐⭐

**Các bước thực hiện**:
1. Query với nội dung hơi liên quan (edge case)
2. Kiểm tra scores trong sources
3. Verify chunks với score < 0.4 bị loại bỏ

**Kết quả mong đợi**:
- Chỉ sources có score >= 0.4 được trả về
- `docs_found` vs `docs_relevant` khác nhau

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Embedded systems programming techniques"}'
```

---

### TC-A03: Long Document Processing (50+ pages)
**Mục đích**: Kiểm tra xử lý tài liệu dài
**Độ khó**: ⭐⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| File | PDF 50-100 pages |
| Limit | MAX_PDF_PAGES = 100 |

**Các bước thực hiện**:
1. Upload PDF dài (50-100 pages)
2. Ingest và đo thời gian
3. Kiểm tra số chunks tạo ra
4. Query về nội dung ở giữa và cuối tài liệu

**Kết quả mong đợi**:
- Ingest thành công (có thể mất vài phút)
- Có thể retrieve content từ bất kỳ phần nào của tài liệu

```bash
curl -X POST http://localhost:8081/upload -F "file=@large_manual.pdf"
curl -X POST http://localhost:8081/ingest \
  -H "Content-Type: application/json" \
  -d '{"filename": "large_manual.pdf", "use_semantic_chunking": true}'
```

---

### TC-A04: Xử lý file vượt giới hạn page
**Mục đích**: Kiểm tra giới hạn số trang
**Độ khó**: ⭐⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| File | PDF > 100 pages |

**Các bước thực hiện**:
1. Upload PDF > 100 pages

**Kết quả mong đợi**:
- Hệ thống từ chối hoặc chỉ xử lý 100 trang đầu
- Error message rõ ràng

---

### TC-A05: Xử lý file vượt giới hạn size
**Mục đích**: Kiểm tra giới hạn dung lượng file
**Độ khó**: ⭐⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| File | File > 50MB |
| Limit | MAX_FILE_SIZE_MB = 50 |

**Các bước thực hiện**:
1. Upload file > 50MB

**Kết quả mong đợi**:
- Status code: `413 Payload Too Large` hoặc tương tự
- Error message về giới hạn file size

---

### TC-A06: Complex Query - Multi-hop Reasoning
**Mục đích**: Kiểm tra khả năng trả lời câu hỏi phức tạp
**Độ khó**: ⭐⭐⭐

**Preconditions**:
- Ingest tài liệu về cấu hình peripherals

**Các bước thực hiện**:
1. Query yêu cầu kết hợp thông tin từ nhiều chunks:
   ```
   "Để giao tiếp với cảm biến nhiệt độ qua I2C, tôi cần cấu hình những
   thanh ghi nào và clock speed bao nhiêu là phù hợp?"
   ```

**Kết quả mong đợi**:
- Response kết hợp thông tin từ nhiều nguồn
- Sources chứa nhiều chunks relevant

---

### TC-A07: Code Block Preservation
**Mục đích**: Kiểm tra semantic chunking giữ nguyên code blocks
**Độ khó**: ⭐⭐⭐

**Preconditions**:
- Tài liệu chứa code examples dài (>100 lines)

**Các bước thực hiện**:
1. Ingest tài liệu với code
2. Query về function cụ thể trong code
3. Kiểm tra response có code đầy đủ không

**Kết quả mong đợi**:
- Code blocks không bị cắt giữa chừng
- Syntax được preserve

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Cho tôi ví dụ code cấu hình UART?"}'
```

---

### TC-A08: Unicode và Special Characters
**Mục đích**: Kiểm tra xử lý tiếng Việt và ký tự đặc biệt
**Độ khó**: ⭐⭐⭐

**Các bước thực hiện**:
1. Upload tài liệu có tiếng Việt đầy đủ dấu
2. Query với tiếng Việt có dấu
3. Kiểm tra response

**Kết quả mong đợi**:
- Không lỗi encoding
- Response hiển thị đúng tiếng Việt

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Giải thích chế độ ngắt ngoài EXTI?"}'
```

---

### TC-A09: Duplicate Document Detection
**Mục đích**: Kiểm tra xử lý upload trùng lặp
**Độ khó**: ⭐⭐⭐

**Các bước thực hiện**:
1. Upload và ingest file A
2. Upload và ingest file A lần nữa (cùng nội dung)

**Kết quả mong đợi**:
- Hệ thống detect duplicate (qua file_hash)
- Không tạo chunks trùng lặp HOẶC thông báo đã tồn tại

---

### TC-A10: Streaming Response
**Mục đích**: Kiểm tra streaming output
**Độ khó**: ⭐⭐⭐

| Thuộc tính | Giá trị |
|------------|---------|
| Parameter | `stream: true` |

**Các bước thực hiện**:
1. Gửi query với streaming enabled
2. Observe response chunks

**Kết quả mong đợi**:
- Response được stream theo từng phần
- Không phải đợi toàn bộ response

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Giải thích chi tiết về DMA controller?", "stream": true}'
```

---

### TC-A11: Max Tokens Limit
**Mục đích**: Kiểm tra giới hạn độ dài response
**Độ khó**: ⭐⭐⭐

**Các bước thực hiện**:
1. Query yêu cầu câu trả lời dài với `max_tokens: 100`
2. Query với `max_tokens: 2048`
3. So sánh độ dài response

**Kết quả mong đợi**:
- Response bị cắt ngắn khi max_tokens thấp
- Response đầy đủ khi max_tokens cao

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Giải thích chi tiết tất cả các timer modes?", "max_tokens": 100}'
```

---

## Cấp độ 4: Kiểm thử Hiệu năng & Bảo mật

### TC-P01: Concurrent Requests
**Mục đích**: Kiểm tra xử lý nhiều request đồng thời
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Sử dụng tool như `ab` (Apache Bench) hoặc `wrk`
2. Gửi 10-50 concurrent requests

**Kết quả mong đợi**:
- Server không crash
- Response time tăng nhưng vẫn acceptable
- Không có request bị drop

```bash
# Sử dụng Apache Bench
ab -n 50 -c 10 -p query.json -T application/json http://localhost:8081/chat

# query.json:
# {"query": "GPIO là gì?"}
```

---

### TC-P02: Memory Usage Under Load
**Mục đích**: Kiểm tra memory usage khi xử lý nhiều tài liệu
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Monitor memory trước khi upload
2. Upload và ingest 20+ tài liệu liên tục
3. Monitor memory sau

**Kết quả mong đợi**:
- Memory tăng nhưng ổn định
- Không có memory leak

```bash
# Monitor memory
watch -n 1 'ps aux | grep python'
```

---

### TC-P03: Response Time Benchmark
**Mục đích**: Đo baseline response time
**Độ khó**: ⭐⭐⭐⭐

**Metrics cần đo**:
| Operation | Target Time |
|-----------|-------------|
| Health check | < 100ms |
| Simple query (cache miss) | < 5s |
| Simple query (cache hit) | < 500ms |
| Document ingest (10 pages) | < 30s |
| Upload (5MB) | < 5s |

**Các bước thực hiện**:
1. Chạy mỗi operation 10 lần
2. Tính average, P95, P99

---

### TC-P04: Large Scale Document Store
**Mục đích**: Kiểm tra performance với nhiều tài liệu
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Ingest 50+ tài liệu khác nhau
2. Kiểm tra query time
3. Kiểm tra accuracy của retrieval

**Kết quả mong đợi**:
- Query time không tăng đáng kể
- Retrieval vẫn chính xác

---

### TC-S01: SQL/NoSQL Injection
**Mục đích**: Kiểm tra bảo mật injection
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Gửi query với payload injection:
   ```json
   {"query": "'; DROP TABLE documents; --"}
   ```
2. Thử các payload khác

**Kết quả mong đợi**:
- Hệ thống không bị ảnh hưởng
- Response là câu trả lời bình thường hoặc error message an toàn

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "'\'' OR 1=1 --"}'
```

---

### TC-S02: XSS Prevention
**Mục đích**: Kiểm tra XSS trong response
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Upload tài liệu có chứa script tags
   ```html
   <script>alert('XSS')</script>
   ```
2. Query và kiểm tra response trong browser

**Kết quả mong đợi**:
- Script không execute
- Content được escape hoặc sanitize

---

### TC-S03: Path Traversal
**Mục đích**: Kiểm tra path traversal trong filename
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Upload file với filename chứa path traversal:
   ```
   ../../../etc/passwd
   ```

**Kết quả mong đợi**:
- Server từ chối hoặc sanitize filename
- Không có unauthorized file access

---

### TC-S04: File Type Validation
**Mục đích**: Kiểm tra validation file type
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Rename file executable (.exe) thành .txt
2. Upload lên server

**Kết quả mong đợi**:
- Server validate actual content, không chỉ extension
- Từ chối file không hợp lệ

---

### TC-S05: Rate Limiting
**Mục đích**: Kiểm tra rate limiting (nếu có)
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Gửi 100 requests trong 10 giây

**Kết quả mong đợi**:
- Rate limiting kick in
- Status code: `429 Too Many Requests`

---

### TC-S06: Redis Connection Security
**Mục đích**: Kiểm tra Redis không exposed
**Độ khó**: ⭐⭐⭐⭐

**Các bước thực hiện**:
1. Thử connect Redis từ external IP

**Kết quả mong đợi**:
- Redis chỉ accessible từ localhost
- Không có unauthorized access

---

## Checklist Tổng hợp

### Basic Tests (8 tests)
- [ ] TC-B01: Health Check
- [ ] TC-B02: Lấy danh sách tài liệu (trống)
- [ ] TC-B03: Upload file TXT đơn giản
- [ ] TC-B04: Ingest tài liệu đã upload
- [ ] TC-B05: Chat query đơn giản (Tiếng Việt)
- [ ] TC-B06: Chat query đơn giản (Tiếng Anh)
- [ ] TC-B07: Xem thống kê hệ thống
- [ ] TC-B08: Xóa tài liệu

### Intermediate Tests (9 tests)
- [ ] TC-M01: Upload và Ingest file PDF
- [ ] TC-M02: Upload và Ingest file DOCX
- [ ] TC-M03: Upload và xử lý hình ảnh với OCR
- [ ] TC-M04: Hybrid Search
- [ ] TC-M05: Source Attribution
- [ ] TC-M06: Query Caching
- [ ] TC-M07: Clear Cache
- [ ] TC-M08: Top-K Results
- [ ] TC-M09: Semantic vs Simple Chunking

### Advanced Tests (11 tests)
- [ ] TC-A01: Query không có context liên quan
- [ ] TC-A02: Relevance Threshold Testing
- [ ] TC-A03: Long Document Processing
- [ ] TC-A04: File vượt giới hạn page
- [ ] TC-A05: File vượt giới hạn size
- [ ] TC-A06: Complex Query - Multi-hop Reasoning
- [ ] TC-A07: Code Block Preservation
- [ ] TC-A08: Unicode và Special Characters
- [ ] TC-A09: Duplicate Document Detection
- [ ] TC-A10: Streaming Response
- [ ] TC-A11: Max Tokens Limit

### Performance & Security Tests (10 tests)
- [ ] TC-P01: Concurrent Requests
- [ ] TC-P02: Memory Usage Under Load
- [ ] TC-P03: Response Time Benchmark
- [ ] TC-P04: Large Scale Document Store
- [ ] TC-S01: SQL/NoSQL Injection
- [ ] TC-S02: XSS Prevention
- [ ] TC-S03: Path Traversal
- [ ] TC-S04: File Type Validation
- [ ] TC-S05: Rate Limiting
- [ ] TC-S06: Redis Connection Security

---

## Test Data Recommendations

### Sample Files cần chuẩn bị:
1. **test_simple.txt** - File text đơn giản (~1KB)
2. **stm32_guide.pdf** - PDF có text + tables (~5MB, 20 pages)
3. **tutorial.docx** - Word với embedded images (~3MB)
4. **schematic.png** - Sơ đồ mạch có text (~500KB)
5. **large_manual.pdf** - PDF dài (~20MB, 80 pages)
6. **code_examples.txt** - File chứa nhiều code blocks
7. **vietnamese_doc.txt** - Tài liệu tiếng Việt đầy đủ dấu

### Nội dung test về Embedded Systems:
- GPIO configuration
- I2C/SPI/UART protocols
- Timer/Counter modes
- Interrupt handling (NVIC, EXTI)
- DMA operations
- Clock configuration (RCC)
- ADC/DAC operations
- Memory mapping
- Register descriptions

---

*Tài liệu được tạo cho hệ thống RAG Chatbot v2.0.0*
