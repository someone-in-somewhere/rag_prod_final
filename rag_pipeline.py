"""RAG Pipeline: Retrieve + Generate với caching và retry"""
"""rag_pipeline.py"""
from openai import OpenAI
from typing import List, Dict, Optional, Generator
from functools import lru_cache
import hashlib
import time
import logging

from vectorstore_chroma import get_vectorstore
from config import (
    VLLM_BASE_URL, LLM_MODEL, TOP_K, RELEVANCE_THRESHOLD,
    DENSE_WEIGHT, SPARSE_WEIGHT, QUERY_CACHE_SIZE, ENABLE_QUERY_CACHE,
    MAX_RETRIES, RETRY_DELAY, TEMPERATURE, LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# vLLM client
llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

# Cache cho query embeddings
_query_cache: Dict[str, List[Dict]] = {}
MAX_CACHE_SIZE = QUERY_CACHE_SIZE

SYSTEM_PROMPT_EN = """You are an expert assistant specializing in embedded programming and embedded systems.

IMPORTANT RULES:
- ONLY answer based on the provided context from the knowledge base.
- If the context does not contain relevant information, respond EXACTLY with: "NO_RELEVANT_INFO"
- DO NOT make up or infer information not in the context.
- Always cite which document/source you got the information from.
- Provide code examples only if they exist in the context.
- For technical terms, registers, or configurations, be precise and accurate.

Respond in the same language as the user's question."""

SYSTEM_PROMPT_VI = """Bạn là trợ lý chuyên gia về lập trình nhúng và hệ thống nhúng.

QUY TẮC QUAN TRỌNG:
- CHỈ trả lời dựa trên ngữ cảnh được cung cấp từ cơ sở kiến thức.
- Nếu ngữ cảnh KHÔNG chứa thông tin liên quan, trả lời CHÍNH XÁC: "NO_RELEVANT_INFO"
- KHÔNG ĐƯỢC bịa hoặc suy luận thông tin không có trong ngữ cảnh.
- Luôn trích dẫn nguồn tài liệu mà bạn lấy thông tin.
- Chỉ cung cấp ví dụ code nếu có trong ngữ cảnh.
- Với các thuật ngữ kỹ thuật, thanh ghi, cấu hình, hãy chính xác.

Trả lời bằng ngôn ngữ của câu hỏi."""


def detect_language(text: str) -> str:
    """Detect ngôn ngữ: vi hoặc en"""
    vn_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    text_lower = text.lower()
    vn_count = sum(1 for c in text_lower if c in vn_chars)
    return "vi" if vn_count > 2 else "en"


def _cache_key(query: str, top_k: int, use_hybrid: bool) -> str:
    """Tạo cache key cho query"""
    return hashlib.md5(f"{query}:{top_k}:{use_hybrid}".encode()).hexdigest()


def retrieve_with_cache(
    query: str, 
    top_k: int = TOP_K, 
    use_hybrid: bool = True
) -> List[Dict]:
    """Retrieve với caching"""
    global _query_cache
    
    cache_key = _cache_key(query, top_k, use_hybrid)
    
    # Check cache
    if cache_key in _query_cache:
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return _query_cache[cache_key]
    
    # Retrieve
    vs = get_vectorstore()
    results = vs.search(query, top_k=top_k, use_hybrid=use_hybrid)
    
    # Update cache (với size limit)
    if len(_query_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_query_cache.keys())[:MAX_CACHE_SIZE // 4]
        for k in keys_to_remove:
            del _query_cache[k]
    
    _query_cache[cache_key] = results
    return results


def clear_cache():
    """Clear query cache"""
    global _query_cache
    _query_cache = {}
    logger.info("Query cache cleared")


def format_context(docs: List[Dict]) -> tuple:
    """Format retrieved docs thành context string"""
    if not docs:
        return "No relevant documents found.", False, []
    
    relevant_docs = [d for d in docs if d.get("score", 0) >= RELEVANCE_THRESHOLD]
    
    if not relevant_docs:
        return "No relevant documents found.", False, []
    
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.get("metadata", {}).get("source", "Unknown")
        score = doc.get("score", 0)
        chunk_idx = doc.get("metadata", {}).get("chunk_index", "?")
        text = doc.get("text", "")
        
        # Thêm thông tin về loại score nếu có
        score_info = f"relevance: {score:.2f}"
        if "dense_score" in doc and "sparse_score" in doc:
            score_info += f", dense: {doc['dense_score']:.2f}, sparse: {doc['sparse_score']:.2f}"
        
        context_parts.append(
            f"[{i}] Source: {source} (chunk {chunk_idx}, {score_info})\n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts), True, relevant_docs


def build_prompt(query: str, context: str, lang: str) -> List[Dict]:
    """Build prompt messages"""
    system = SYSTEM_PROMPT_VI if lang == "vi" else SYSTEM_PROMPT_EN
    
    if lang == "vi":
        user_content = f"""Ngữ cảnh từ cơ sở kiến thức:
{context}

---
Câu hỏi: {query}

Hãy trả lời chi tiết dựa trên ngữ cảnh. Nếu không có thông tin liên quan, trả lời "NO_RELEVANT_INFO"."""
    else:
        user_content = f"""Context from knowledge base:
{context}

---
Question: {query}

Provide a detailed answer based on the context. If no relevant information, respond with "NO_RELEVANT_INFO"."""
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]


def generate_with_retry(
    messages: List[Dict], 
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> str:
    """Generate với retry logic"""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            return response.choices[0].message.content
        
        except Exception as e:
            last_error = e
            logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    
    logger.error(f"Generation failed after {MAX_RETRIES} attempts: {last_error}")
    raise last_error


def chat(
    query: str, 
    top_k: int = TOP_K, 
    max_tokens: int = 1024,
    use_hybrid: bool = True
) -> Dict:
    """
    Main chat function.
    Returns: {query, response, language, sources, context_used, retrieval_info}
    """
    start_time = time.time()
    lang = detect_language(query)
    
    # Retrieve
    retrieve_start = time.time()
    retrieved_docs = retrieve_with_cache(query, top_k=top_k, use_hybrid=use_hybrid)
    retrieve_time = time.time() - retrieve_start
    
    # Format context
    context, has_relevant, relevant_docs = format_context(retrieved_docs)
    
    # No relevant info message
    no_info_msg = (
        "Tôi không có thông tin về chủ đề này trong tài liệu hiện tại. "
        "Vui lòng upload tài liệu liên quan hoặc hỏi câu hỏi khác."
    ) if lang == "vi" else (
        "I don't have information about this topic in the current documents. "
        "Please upload relevant documents or ask another question."
    )
    
    if not has_relevant:
        return {
            "query": query,
            "response": no_info_msg,
            "language": lang,
            "sources": [],
            "context_used": False,
            "retrieval_info": {
                "docs_found": len(retrieved_docs),
                "docs_relevant": 0,
                "retrieve_time_ms": int(retrieve_time * 1000),
                "hybrid_search": use_hybrid
            }
        }
    
    # Generate
    generate_start = time.time()
    messages = build_prompt(query, context, lang)
    response = generate_with_retry(messages, max_tokens=max_tokens)
    generate_time = time.time() - generate_start
    
    # Check if model says no info
    if "NO_RELEVANT_INFO" in response:
        return {
            "query": query,
            "response": no_info_msg,
            "language": lang,
            "sources": [],
            "context_used": False,
            "retrieval_info": {
                "docs_found": len(retrieved_docs),
                "docs_relevant": len(relevant_docs),
                "retrieve_time_ms": int(retrieve_time * 1000),
                "generate_time_ms": int(generate_time * 1000),
                "hybrid_search": use_hybrid
            }
        }
    
    # Build sources
    sources = [{
        "source": d.get("metadata", {}).get("source"),
        "score": round(d.get("score", 0), 3),
        "chunk_index": d.get("metadata", {}).get("chunk_index")
    } for d in relevant_docs]
    
    total_time = time.time() - start_time
    
    return {
        "query": query,
        "response": response,
        "language": lang,
        "sources": sources,
        "context_used": True,
        "retrieval_info": {
            "docs_found": len(retrieved_docs),
            "docs_relevant": len(relevant_docs),
            "retrieve_time_ms": int(retrieve_time * 1000),
            "generate_time_ms": int(generate_time * 1000),
            "total_time_ms": int(total_time * 1000),
            "hybrid_search": use_hybrid
        }
    }


def chat_stream(
    query: str, 
    top_k: int = TOP_K, 
    max_tokens: int = 1024,
    use_hybrid: bool = True
) -> Generator[str, None, None]:
    """Streaming chat"""
    lang = detect_language(query)
    
    # Retrieve
    retrieved_docs = retrieve_with_cache(query, top_k=top_k, use_hybrid=use_hybrid)
    context, has_relevant, relevant_docs = format_context(retrieved_docs)
    
    if not has_relevant:
        no_info_msg = (
            "Tôi không có thông tin về chủ đề này trong tài liệu hiện tại. "
            "Vui lòng upload tài liệu liên quan hoặc hỏi câu hỏi khác."
        ) if lang == "vi" else (
            "I don't have information about this topic in the current documents. "
            "Please upload relevant documents or ask another question."
        )
        yield no_info_msg
        return
    
    messages = build_prompt(query, context, lang)
    
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        yield f"\n\n[Error: {str(e)}]"


def retrieve(query: str, top_k: int = TOP_K, use_hybrid: bool = True) -> List[Dict]:
    """Direct retrieve function (for debugging)"""
    return retrieve_with_cache(query, top_k=top_k, use_hybrid=use_hybrid)


def generate(messages: List[Dict], max_tokens: int = 1024) -> str:
    """Direct generate function (for debugging)"""
    return generate_with_retry(messages, max_tokens=max_tokens)