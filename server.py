"""FastAPI Server cho Embedded RAG Chatbot"""
"""server.py"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import shutil
import os
import logging
from pathlib import Path

from config import UPLOAD_DIR, BASE_DIR, MAX_PDF_PAGES, MAX_FILE_SIZE_MB, SERVER_HOST, SERVER_PORT, LOG_LEVEL
from document_ingest import ingest_document
from vectorstore_chroma import get_vectorstore
from redis_store import get_redis_store
from rag_pipeline import chat, chat_stream, clear_cache

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Embedded RAG Chatbot",
    version="2.0.0",
    description="RAG system for embedded programming documentation"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Pydantic Models ===

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_tokens: int = Field(default=1024, ge=100, le=4096)
    stream: bool = False
    use_hybrid: bool = True  # Mới: toggle hybrid search


class ChatResponse(BaseModel):
    query: str
    response: str
    language: str
    sources: List[dict]
    context_used: bool
    retrieval_info: Optional[dict] = None


class IngestRequest(BaseModel):
    filename: str
    use_semantic_chunking: bool = True  # Mới: toggle semantic chunking


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str
    chunking_method: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    doc_type: str
    chunk_count: int
    ingested_at: Optional[str] = None


# === Endpoints ===

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        vs = get_vectorstore()
        vs_stats = vs.get_stats()
    except Exception as e:
        vs_stats = {"error": str(e)}
    
    try:
        redis = get_redis_store()
        redis_status = "connected"
        doc_count = len(redis.list_documents())
    except Exception as e:
        redis_status = f"error: {e}"
        doc_count = 0
    
    return {
        "status": "healthy",
        "vectorstore": vs_stats,
        "redis": redis_status,
        "document_count": doc_count,
        "limits": {
            "max_pdf_pages": MAX_PDF_PAGES,
            "max_file_size_mb": MAX_FILE_SIZE_MB
        }
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file"""
    allowed_ext = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed_ext:
        raise HTTPException(400, f"File type not supported. Allowed: {allowed_ext}")
    
    # Check size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File quá lớn: {file_size_mb:.1f}MB (giới hạn {MAX_FILE_SIZE_MB}MB)")
    
    # Save file
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    logger.info(f"Uploaded: {file.filename} ({file_size_mb:.2f}MB)")
    
    return {
        "filename": file.filename,
        "path": str(filepath),
        "size_mb": round(file_size_mb, 2),
        "status": "uploaded"
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest document vào vector store"""
    filepath = UPLOAD_DIR / request.filename
    
    if not filepath.exists():
        raise HTTPException(404, f"File not found: {request.filename}")
    
    try:
        logger.info(f"Ingesting: {request.filename} (semantic={request.use_semantic_chunking})")
        
        result = ingest_document(
            str(filepath), 
            use_semantic_chunking=request.use_semantic_chunking
        )
        
        # Store in Redis
        redis = get_redis_store()
        redis.store_document(result["doc_id"], {
            "filename": result["filename"],
            "doc_type": result["doc_type"],
            "raw_text": result["raw_text"],
            "chunk_count": result["chunk_count"],
            "metadata": result["metadata"]
        })
        
        # Store in ChromaDB
        vs = get_vectorstore()
        vs.add_documents(result["chunks"], use_sparse=True)
        
        # Clear query cache vì có document mới
        clear_cache()
        
        logger.info(f"Ingested: {request.filename} -> {result['chunk_count']} chunks")
        
        return IngestResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            status="ingested",
            chunking_method="semantic" if request.use_semantic_chunking else "simple"
        )
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(500, f"Ingest failed: {str(e)}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint"""
    logger.info(f"Query: {request.query[:100]}... (hybrid={request.use_hybrid})")
    
    if request.stream:
        return StreamingResponse(
            chat_stream(
                request.query, 
                request.top_k, 
                request.max_tokens,
                use_hybrid=request.use_hybrid
            ),
            media_type="text/plain"
        )
    
    result = chat(
        request.query, 
        request.top_k, 
        request.max_tokens,
        use_hybrid=request.use_hybrid
    )
    
    logger.info(
        f"Response generated: {result['retrieval_info'].get('total_time_ms', 0)}ms, "
        f"sources={len(result['sources'])}"
    )
    
    return ChatResponse(**result)


@app.get("/documents")
async def list_documents():
    """List all documents"""
    try:
        redis = get_redis_store()
        docs = redis.get_all_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        logger.error(f"List documents error: {e}")
        return {"documents": [], "count": 0, "error": str(e)}


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document details"""
    redis = get_redis_store()
    doc = redis.get_document(doc_id)
    
    if not doc:
        raise HTTPException(404, "Document not found")
    
    # Không trả về raw_text để response nhẹ hơn
    return {
        "id": doc_id,
        "filename": doc.get("filename"),
        "doc_type": doc.get("doc_type"),
        "chunk_count": doc.get("chunk_count"),
        "metadata": doc.get("metadata")
    }


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document"""
    redis = get_redis_store()
    doc = redis.get_document(doc_id)
    
    if not doc:
        raise HTTPException(404, "Document not found")
    
    filename = doc.get("filename", "")
    deleted_chunks = 0
    
    # Delete from ChromaDB
    try:
        vs = get_vectorstore()
        deleted_chunks = vs.delete_by_source(filename)
    except Exception as e:
        logger.error(f"ChromaDB delete error: {e}")
    
    # Delete from Redis
    try:
        redis.delete_document(doc_id)
    except Exception as e:
        logger.error(f"Redis delete error: {e}")
    
    # Delete file
    filepath = UPLOAD_DIR / filename
    if filepath.exists():
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"File delete error: {e}")
    
    # Clear cache
    clear_cache()
    
    logger.info(f"Deleted: {filename} ({deleted_chunks} chunks)")
    
    return {
        "status": "deleted",
        "doc_id": doc_id,
        "filename": filename,
        "chunks_deleted": deleted_chunks
    }


@app.post("/cache/clear")
async def clear_query_cache():
    """Clear query cache"""
    clear_cache()
    return {"status": "cache_cleared"}


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        vs = get_vectorstore()
        vs_stats = vs.get_stats()
    except Exception as e:
        vs_stats = {"error": str(e)}
    
    try:
        redis = get_redis_store()
        docs = redis.get_all_documents()
        doc_stats = {
            "total_documents": len(docs),
            "by_type": {}
        }
        for doc in docs:
            dtype = doc.get("doc_type", "unknown")
            doc_stats["by_type"][dtype] = doc_stats["by_type"].get(dtype, 0) + 1
    except Exception as e:
        doc_stats = {"error": str(e)}
    
    return {
        "vectorstore": vs_stats,
        "documents": doc_stats
    }


@app.get("/limits")
async def get_limits():
    """Get system limits"""
    return {
        "max_pdf_pages": MAX_PDF_PAGES,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "supported_formats": [".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"]
    }


# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve frontend HTML"""
    frontend_path = BASE_DIR / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return HTMLResponse("<h1>Frontend not found. Create frontend/index.html</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)