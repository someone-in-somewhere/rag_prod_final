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
    MAX_RETRIES, RETRY_DELAY, TEMPERATURE, LOG_LEVEL,
    MAX_CONTEXT_LENGTH, MAX_CHUNK_LENGTH, MAX_TOKENS, MAX_CONVERSATION_HISTORY
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# vLLM client
llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

# Cache cho query embeddings
_query_cache: Dict[str, List[Dict]] = {}
MAX_CACHE_SIZE = QUERY_CACHE_SIZE

# System prompts với giới hạn ngôn ngữ NGHIÊM NGẶT (chỉ Việt hoặc Anh)
SYSTEM_PROMPT_EN = """You are a helpful assistant. Answer based ONLY on the provided context and conversation history. If no relevant info, say "NO_RELEVANT_INFO". Cite sources. Be concise.

CRITICAL: You MUST respond ONLY in English. DO NOT use any other language including Chinese, Japanese, Korean, etc. If you start writing in another language, STOP and rewrite in English."""

SYSTEM_PROMPT_VI = """Bạn là trợ lý hữu ích. Trả lời CHỈ dựa trên ngữ cảnh và lịch sử hội thoại. Nếu không có thông tin, nói "NO_RELEVANT_INFO". Trích nguồn. Ngắn gọn.

QUAN TRỌNG: Bạn PHẢI trả lời CHỈ bằng tiếng Việt. KHÔNG ĐƯỢC sử dụng bất kỳ ngôn ngữ nào khác bao gồm tiếng Trung, tiếng Nhật, tiếng Hàn, v.v. Nếu bạn bắt đầu viết bằng ngôn ngữ khác, DỪNG LẠI và viết lại bằng tiếng Việt."""


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


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length characters"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_context(docs: List[Dict]) -> tuple:
    """Format retrieved docs thành context string với giới hạn độ dài"""
    if not docs:
        return "No relevant documents found.", False, []

    relevant_docs = [d for d in docs if d.get("score", 0) >= RELEVANCE_THRESHOLD]

    if not relevant_docs:
        return "No relevant documents found.", False, []

    context_parts = []
    total_length = 0

    for i, doc in enumerate(relevant_docs, 1):
        # Dừng nếu đã đạt giới hạn context
        if total_length >= MAX_CONTEXT_LENGTH:
            break

        source = doc.get("metadata", {}).get("source", "Unknown")
        chunk_idx = doc.get("metadata", {}).get("chunk_index", "?")
        text = doc.get("text", "")

        # Truncate mỗi chunk
        text = truncate_text(text, MAX_CHUNK_LENGTH)

        # Format ngắn gọn hơn
        chunk_text = f"[{i}] {source}:\n{text}"
        context_parts.append(chunk_text)
        total_length += len(chunk_text)

    return "\n\n".join(context_parts), True, relevant_docs


def build_prompt(query: str, context: str, lang: str, conversation_history: List[Dict] = None) -> List[Dict]:
    """Build prompt messages với conversation history"""
    system = SYSTEM_PROMPT_VI if lang == "vi" else SYSTEM_PROMPT_EN

    messages = [{"role": "system", "content": system}]

    # Thêm conversation history nếu có
    if conversation_history:
        # Giới hạn số tin nhắn history
        history = conversation_history[-MAX_CONVERSATION_HISTORY:]
        for msg in history:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"][:500]  # Giới hạn mỗi tin nhắn 500 chars
                })

    # Thêm context và query hiện tại
    if lang == "vi":
        user_content = f"""Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {query}"""
    else:
        user_content = f"""Context from documents:
{context}

Question: {query}"""

    messages.append({"role": "user", "content": user_content})

    return messages


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
    max_tokens: int = MAX_TOKENS,
    use_hybrid: bool = True,
    conversation_history: List[Dict] = None
) -> Dict:
    """
    Main chat function với conversation memory.
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

    # Generate với conversation history
    generate_start = time.time()
    messages = build_prompt(query, context, lang, conversation_history)
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
    max_tokens: int = MAX_TOKENS,
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