"""Document ingestion: Parse PDF, DOCX, TXT, Images với Semantic Chunking"""
"""document_ingest.py"""
import pymupdf
from docx import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import os
import re
import tempfile
import uuid
from datetime import datetime
from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR, MAX_FILE_SIZE_MB, MAX_PDF_PAGES, MAX_IMAGE_SIZE_MB

# Patterns để detect boundaries
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`')
HEADING_PATTERN = re.compile(r'^#{1,6}\s+.+$|^[A-Z][^.!?]*:$|^\d+\.\d*\s+[A-Z]', re.MULTILINE)
TABLE_MARKER = re.compile(r'^\s*\|.*\|.*$|^\[Table|^\[Page.*Table', re.MULTILINE)
REGISTER_PATTERN = re.compile(r'Register\s+Description|Bit\s+Field|Address\s+Offset|0x[0-9A-Fa-f]+', re.IGNORECASE)


def compute_file_hash(filepath: str) -> str:
    """Tính MD5 hash của file"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_file_limits(filepath: str, max_size_mb: int = MAX_FILE_SIZE_MB):
    """Kiểm tra giới hạn file"""
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File quá lớn: {file_size_mb:.1f}MB (giới hạn {max_size_mb}MB)")
    return file_size_mb


def detect_segment_type(text: str) -> str:
    """Detect loại segment: code, table, register, heading, text"""
    if CODE_BLOCK_PATTERN.search(text):
        return "code"
    if TABLE_MARKER.search(text):
        return "table"
    if REGISTER_PATTERN.search(text):
        return "register"
    if HEADING_PATTERN.match(text.strip()):
        return "heading"
    return "text"


def find_semantic_boundaries(text: str) -> List[Tuple[int, str]]:
    """
    Tìm các điểm boundary tự nhiên trong text.
    Returns: List of (position, boundary_type)
    """
    boundaries = [(0, "start")]
    
    # Tìm headings
    for match in HEADING_PATTERN.finditer(text):
        boundaries.append((match.start(), "heading"))
    
    # Tìm code blocks
    for match in CODE_BLOCK_PATTERN.finditer(text):
        boundaries.append((match.start(), "code_start"))
        boundaries.append((match.end(), "code_end"))
    
    # Tìm table markers
    for match in TABLE_MARKER.finditer(text):
        boundaries.append((match.start(), "table"))
    
    # Tìm register sections
    for match in REGISTER_PATTERN.finditer(text):
        boundaries.append((match.start(), "register"))
    
    # Tìm paragraph breaks (double newline)
    for match in re.finditer(r'\n\s*\n', text):
        boundaries.append((match.start(), "paragraph"))
    
    # Sort theo position
    boundaries.sort(key=lambda x: x[0])
    
    return boundaries


def semantic_chunk_text(
    text: str, 
    chunk_size: int = CHUNK_SIZE, 
    overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Chunk text theo semantic boundaries.
    Giữ nguyên code blocks, tables, register descriptions.
    """
    if not text.strip():
        return []
    
    boundaries = find_semantic_boundaries(text)
    chunks = []
    
    # Tạo segments từ boundaries
    segments = []
    for i in range(len(boundaries)):
        start = boundaries[i][0]
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        segment_text = text[start:end].strip()
        
        if segment_text:
            segment_type = detect_segment_type(segment_text)
            segments.append({
                "text": segment_text,
                "type": segment_type,
                "start": start,
                "end": end
            })
    
    # Merge segments thành chunks
    current_chunk = ""
    current_types = set()
    current_word_count = 0
    
    for seg in segments:
        seg_words = len(seg["text"].split())
        seg_type = seg["type"]
        
        # Nếu segment là code/table/register và đủ lớn -> chunk riêng
        if seg_type in ["code", "table", "register"] and seg_words > 50:
            # Save current chunk trước
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "types": list(current_types)
                })
            
            # Segment này thành chunk riêng
            chunks.append({
                "text": seg["text"],
                "types": [seg_type]
            })
            
            current_chunk = ""
            current_types = set()
            current_word_count = 0
            continue
        
        # Nếu thêm segment này vượt quá chunk_size
        if current_word_count + seg_words > chunk_size:
            # Save current chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "types": list(current_types)
                })
            
            # Overlap: lấy phần cuối của chunk cũ
            if overlap > 0 and current_chunk:
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) + "\n\n"
                current_chunk = overlap_text
                current_word_count = overlap
            else:
                current_chunk = ""
                current_word_count = 0
            
            current_types = set()
        
        # Thêm segment vào current chunk
        current_chunk += seg["text"] + "\n\n"
        current_types.add(seg_type)
        current_word_count += seg_words
    
    # Save chunk cuối
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "types": list(current_types)
        })
    
    return chunks


def chunk_text(
    text: str, 
    chunk_size: int = CHUNK_SIZE, 
    overlap: int = CHUNK_OVERLAP,
    use_semantic: bool = True
) -> List[str]:
    """
    Wrapper function - trả về list strings để compatible với code cũ.
    """
    if use_semantic:
        chunks = semantic_chunk_text(text, chunk_size, overlap)
        return [c["text"] for c in chunks]
    
    # Fallback: simple word-based chunking
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(words):
            break
    
    return chunks


def format_table(table_data: List[List]) -> str:
    """Format table data thành text"""
    if not table_data:
        return ""
    
    lines = []
    for row in table_data:
        cells = [str(cell).strip() if cell else "" for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(filepath: str, max_pages: int = MAX_PDF_PAGES) -> Dict:
    """Parse PDF: text + tables"""
    file_size_mb = check_file_limits(filepath)
    
    doc = pymupdf.open(filepath)
    page_count = len(doc)
    
    if page_count > max_pages:
        doc.close()
        raise ValueError(f"PDF quá nhiều trang: {page_count} (giới hạn {max_pages} trang)")
    
    print(f"Processing PDF: {page_count} pages, {file_size_mb:.1f}MB")
    
    all_text = []
    tables_text = []
    
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text("text")
        if text.strip():
            all_text.append(f"[Page {page_num + 1}]\n{text}")
        
        # Extract tables
        try:
            tabs = page.find_tables()
            for i, tab in enumerate(tabs):
                table_data = tab.extract()
                if table_data:
                    table_str = format_table(table_data)
                    tables_text.append(f"[Page {page_num + 1} - Table {i + 1}]\n{table_str}")
        except Exception as e:
            print(f"Warning: Could not extract tables from page {page_num + 1}: {e}")
    
    doc.close()
    
    combined = "\n\n".join(all_text)
    if tables_text:
        combined += "\n\n[TABLES]\n" + "\n\n".join(tables_text)
    
    return {
        "text": combined,
        "page_count": page_count,
        "file_size_mb": file_size_mb,
        "has_tables": len(tables_text) > 0
    }


def count_docx_pages(filepath: str) -> int:
    """
    Ước tính số trang của DOCX.
    DOCX không có concept "page" cố định như PDF, 
    nên ước tính dựa trên số từ (~300 từ/trang) và số ảnh/bảng.
    """
    doc = DocxDocument(filepath)
    
    # Đếm tổng số từ
    total_words = 0
    for para in doc.paragraphs:
        total_words += len(para.text.split())
    
    # Đếm số bảng (mỗi bảng ~0.5 trang)
    table_count = len(doc.tables)
    
    # Đếm số ảnh (mỗi ảnh ~0.3 trang)
    image_count = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image_count += 1
    
    # Ước tính: 300 từ/trang + bảng + ảnh
    estimated_pages = (total_words / 300) + (table_count * 0.5) + (image_count * 0.3)
    
    return max(1, int(estimated_pages))


def parse_docx(filepath: str, max_pages: int = MAX_PDF_PAGES) -> Dict:
    """Parse DOCX: text + tables + images"""
    file_size_mb = check_file_limits(filepath)
    
    # Ước tính số trang và kiểm tra giới hạn
    estimated_pages = count_docx_pages(filepath)
    if estimated_pages > max_pages:
        raise ValueError(
            f"DOCX quá dài: ước tính ~{estimated_pages} trang (giới hạn {max_pages} trang). "
            f"Vui lòng chia nhỏ tài liệu."
        )
    
    print(f"Processing DOCX: ~{estimated_pages} pages (estimated), {file_size_mb:.1f}MB")
    
    doc = DocxDocument(filepath)
    paragraphs = []
    tables_text = []
    images_text = []
    
    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            # Detect heading style
            if para.style and para.style.name.startswith('Heading'):
                paragraphs.append(f"## {para.text}")
            else:
                paragraphs.append(para.text)
    
    # Extract tables
    for i, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        if table_data:
            table_str = format_table(table_data)
            tables_text.append(f"[Table {i + 1}]\n{table_str}")
    
    # Extract images
    images_text = extract_images_from_docx(doc, filepath)
    
    # Combine all content
    combined = "\n\n".join(paragraphs)
    if tables_text:
        combined += "\n\n[TABLES]\n" + "\n\n".join(tables_text)
    if images_text:
        combined += "\n\n[IMAGES]\n" + "\n\n".join(images_text)
    
    return {
        "text": combined,
        "paragraph_count": len(paragraphs),
        "file_size_mb": file_size_mb,
        "has_tables": len(tables_text) > 0,
        "has_images": len(images_text) > 0,
        "image_count": len(images_text)
    }


def extract_images_from_docx(doc: DocxDocument, filepath: str) -> List[str]:
    """Extract và xử lý images từ DOCX file"""
    from ocr_utils import process_image
    
    images_text = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Lấy tất cả image relationships từ document
        image_parts = []
        
        # Check main document rels
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_parts.append(rel.target_part)
        
        if not image_parts:
            return []
        
        print(f"Found {len(image_parts)} images in DOCX")
        
        for idx, image_part in enumerate(image_parts):
            try:
                # Get image data
                image_data = image_part.blob
                
                # Determine extension from content type
                content_type = image_part.content_type
                if "png" in content_type:
                    ext = ".png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "gif" in content_type:
                    ext = ".gif"
                elif "bmp" in content_type:
                    ext = ".bmp"
                else:
                    ext = ".png"  # Default
                
                # Save temp image file
                temp_image_path = os.path.join(temp_dir, f"image_{idx}{ext}")
                with open(temp_image_path, "wb") as f:
                    f.write(image_data)
                
                # Check image size
                image_size_mb = len(image_data) / (1024 * 1024)
                if image_size_mb > MAX_IMAGE_SIZE_MB:
                    print(f"Skipping image {idx}: too large ({image_size_mb:.1f}MB)")
                    continue
                
                # Process image with OCR and Vision
                result = process_image(temp_image_path)
                
                if result["combined"] and "[No content extracted" not in result["combined"]:
                    images_text.append(f"[Image {idx + 1}]\n{result['combined']}")
                
            except Exception as e:
                print(f"Error processing image {idx} in DOCX: {e}")
                continue
    
    finally:
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not cleanup temp dir: {e}")
    
    return images_text


def parse_txt(filepath: str) -> Dict:
    """Parse plain text file"""
    file_size_mb = check_file_limits(filepath)
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {
        "text": text,
        "file_size_mb": file_size_mb
    }


def parse_image(filepath: str) -> Dict:
    """Parse image: OCR + Vision caption"""
    from ocr_utils import process_image
    
    file_size_mb = check_file_limits(filepath, max_size_mb=MAX_IMAGE_SIZE_MB)
    
    result = process_image(filepath)
    return {
        "text": result["combined"],
        "ocr_text": result["ocr_text"],
        "caption": result["caption"],
        "file_size_mb": file_size_mb
    }


def ingest_document(filepath: str, use_semantic_chunking: bool = True) -> Dict:
    """
    Ingest một document và trả về chunks + metadata
    """
    path = Path(filepath)
    ext = path.suffix.lower()
    filename = path.name
    
    print(f"Ingesting: {filename}")
    
    # Parse theo loại file
    if ext == ".pdf":
        parsed = parse_pdf(filepath)
        doc_type = "pdf"
    elif ext == ".docx":
        parsed = parse_docx(filepath)
        doc_type = "docx"
    elif ext == ".txt":
        parsed = parse_txt(filepath)
        doc_type = "txt"
    elif ext in [".jpg", ".jpeg", ".png"]:
        parsed = parse_image(filepath)
        doc_type = "image"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Chunk text
    raw_text = parsed.get("text", "")
    chunks = chunk_text(raw_text, use_semantic=use_semantic_chunking)
    
    print(f"Created {len(chunks)} chunks (semantic={use_semantic_chunking})")
    
    # Tạo metadata
    file_hash = compute_file_hash(filepath)
    base_metadata = {
        "source": filename,
        "file_hash": file_hash,
        "doc_type": doc_type,
        "ingested_at": datetime.now().isoformat(),
        "chunking": "semantic" if use_semantic_chunking else "simple"
    }
    
    # Tạo chunk documents
    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_docs.append({
            "text": chunk,
            "metadata": {
                **base_metadata, 
                "chunk_index": i, 
                "total_chunks": len(chunks)
            }
        })
    
    return {
        "doc_id": file_hash,
        "filename": filename,
        "doc_type": doc_type,
        "raw_text": raw_text,
        "chunks": chunk_docs,
        "chunk_count": len(chunks),
        "metadata": base_metadata
    }