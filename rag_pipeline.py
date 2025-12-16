"""
RAG Pipeline Module
===================
Module xử lý RAG (Retrieval-Augmented Generation) với caching và retry.

RAG Pipeline Flow:
1. User gửi query
2. Detect ngôn ngữ (vi/en)
3. Retrieve: Tìm top-k documents liên quan từ vector store
4. Filter: Lọc documents có score >= threshold
5. Build prompt: Tạo prompt với context từ retrieved docs
6. Generate: Gọi LLM để sinh câu trả lời
7. Return: Trả về response + sources

Caching:
- Query cache: Lưu kết quả retrieval theo query hash
- FIFO eviction: Khi cache đầy, xóa entries cũ nhất
- Clear cache khi có document mới được ingest

Retry Logic:
- Generate có thể fail do network/server issues
- Retry với exponential backoff (MAX_RETRIES lần)

Language Support:
- Tự động detect ngôn ngữ từ query
- System prompt và response message theo ngôn ngữ

Sử dụng:
    from rag_pipeline import chat, chat_stream, retrieve

    # Chat thường
    result = chat("GPIO là gì?", top_k=5)
    print(result["response"])

    # Streaming chat
    for chunk in chat_stream("Explain I2C protocol"):
        print(chunk, end="")

    # Debug retrieval
    docs = retrieve("UART configuration")
    for d in docs:
        print(f"{d['score']:.3f}: {d['text'][:100]}...")
"""

from openai import OpenAI
from typing import List, Dict, Optional, Generator
import hashlib
import time
import logging
from datetime import datetime

from vectorstore_chroma import get_vectorstore
from config import (
    VLLM_BASE_URL, LLM_MODEL, TOP_K, RELEVANCE_THRESHOLD,
    DENSE_WEIGHT, SPARSE_WEIGHT, QUERY_CACHE_SIZE, ENABLE_QUERY_CACHE,
    MAX_RETRIES, RETRY_DELAY, TEMPERATURE, LOG_LEVEL,
    DEBUG_RETRIEVAL, DEBUG_GENERATION, DEBUG_CONTEXT
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


def log_debug(flag: bool, prefix: str, message: str):
    """
    Log debug có điều kiện.

    Args:
        flag: Debug flag từ config (DEBUG_RETRIEVAL, DEBUG_GENERATION, etc.)
        prefix: Prefix cho log (emoji + category)
        message: Nội dung log
    """
    if flag:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {prefix}: {message}")


# vLLM client - kết nối đến vLLM server qua OpenAI-compatible API
llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

# Cache cho query embeddings
# Key: hash của (query, top_k, use_hybrid)
# Value: List[Dict] - retrieved documents
_query_cache: Dict[str, List[Dict]] = {}
MAX_CACHE_SIZE = QUERY_CACHE_SIZE


# ============================================
# System Prompts
# ============================================

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


# ============================================
# Helper Functions
# ============================================

def detect_language(text: str) -> str:
    """
    Detect ngôn ngữ của text: Tiếng Việt hoặc Tiếng Anh.

    Phương pháp: Đếm số ký tự tiếng Việt đặc trưng.
    Nếu có > 2 ký tự tiếng Việt -> "vi", ngược lại -> "en"

    Args:
        text: Văn bản cần detect

    Returns:
        str: "vi" hoặc "en"

    Example:
        >>> detect_language("GPIO là gì?")
        'vi'
        >>> detect_language("What is GPIO?")
        'en'
    """
    vn_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    text_lower = text.lower()
    vn_count = sum(1 for c in text_lower if c in vn_chars)
    return "vi" if vn_count > 2 else "en"


def _cache_key(query: str, top_k: int, use_hybrid: bool) -> str:
    """
    Tạo cache key cho query.

    Cache key là MD5 hash của (query, top_k, use_hybrid) để:
    - Đảm bảo key ngắn gọn
    - Tránh special characters trong key

    Args:
        query: Câu query
        top_k: Số kết quả
        use_hybrid: Có dùng hybrid search không

    Returns:
        str: MD5 hash string (32 chars)
    """
    return hashlib.md5(f"{query}:{top_k}:{use_hybrid}".encode()).hexdigest()


# ============================================
# Retrieval Functions
# ============================================

def retrieve_with_cache(
    query: str,
    top_k: int = TOP_K,
    use_hybrid: bool = True
) -> List[Dict]:
    """
    Retrieve documents với caching.

    Quá trình:
    1. Tạo cache key từ query params
    2. Check cache, nếu hit thì return cached results
    3. Nếu miss, gọi vector store search
    4. Lưu kết quả vào cache (với size limit)

    Cache eviction: Simple FIFO - khi cache đầy, xóa 25% entries cũ nhất.

    Args:
        query: Câu query tìm kiếm
        top_k: Số kết quả tối đa (default từ config)
        use_hybrid: Có dùng hybrid search không (default: True)

    Returns:
        List[Dict]: Danh sách documents tìm được
        Mỗi dict có: id, text, score, metadata, (dense_score, sparse_score nếu hybrid)
    """
    global _query_cache

    cache_key = _cache_key(query, top_k, use_hybrid)

    # Check cache
    if ENABLE_QUERY_CACHE and cache_key in _query_cache:
        log_debug(DEBUG_RETRIEVAL, "🔍 RETRIEVE", f"Cache HIT for: '{query[:50]}...'")
        return _query_cache[cache_key]

    log_debug(DEBUG_RETRIEVAL, "🔍 RETRIEVE", f"Cache MISS, searching: '{query[:50]}...'")

    # Retrieve từ vector store
    vs = get_vectorstore()
    results = vs.search(query, top_k=top_k, use_hybrid=use_hybrid)

    # Log top results khi DEBUG
    if DEBUG_RETRIEVAL and results:
        log_debug(DEBUG_RETRIEVAL, "🔍 RETRIEVE", f"Found {len(results)} docs:")
        for i, doc in enumerate(results[:3]):  # Top 3
            source = doc.get("metadata", {}).get("source", "?")
            score = doc.get("score", 0)
            text_preview = doc.get("text", "")[:80].replace("\n", " ")
            log_debug(
                DEBUG_RETRIEVAL, "🔍 RETRIEVE",
                f"  [{i+1}] {score:.3f} | {source} | {text_preview}..."
            )

    # Update cache (với size limit)
    if ENABLE_QUERY_CACHE:
        if len(_query_cache) >= MAX_CACHE_SIZE:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(_query_cache.keys())[:MAX_CACHE_SIZE // 4]
            for k in keys_to_remove:
                del _query_cache[k]
            log_debug(DEBUG_RETRIEVAL, "🔍 RETRIEVE", f"Cache eviction: removed {len(keys_to_remove)} entries")

        _query_cache[cache_key] = results

    return results


def clear_cache():
    """
    Clear query cache.

    Gọi khi:
    - Có document mới được ingest
    - Document bị xóa
    - User yêu cầu clear cache

    Side effects:
    - Reset _query_cache về empty dict
    """
    global _query_cache
    _query_cache = {}
    logger.info("Query cache cleared")
    log_debug(DEBUG_RETRIEVAL, "🔍 RETRIEVE", "Cache cleared")


# ============================================
# Context Formatting
# ============================================

def format_context(docs: List[Dict]) -> tuple:
    """
    Format retrieved docs thành context string cho LLM.

    Quá trình:
    1. Lọc docs có score >= RELEVANCE_THRESHOLD
    2. Format mỗi doc với source, score, text
    3. Join tất cả docs với separator

    Args:
        docs: List documents từ retrieval

    Returns:
        tuple: (context_string, has_relevant_docs, relevant_docs_list)
        - context_string: Formatted context hoặc "No relevant documents found."
        - has_relevant_docs: True nếu có ít nhất 1 doc relevant
        - relevant_docs_list: List các docs đã lọc

    Example output:
        [1] Source: gpio.pdf (chunk 5, relevance: 0.85)
        GPIO (General Purpose Input/Output) là...

        ---

        [2] Source: gpio.pdf (chunk 6, relevance: 0.82)
        Để cấu hình GPIO mode...
    """
    if not docs:
        return "No relevant documents found.", False, []

    # Filter theo relevance threshold
    relevant_docs = [d for d in docs if d.get("score", 0) >= RELEVANCE_THRESHOLD]

    if not relevant_docs:
        log_debug(
            DEBUG_CONTEXT, "📋 CONTEXT",
            f"No docs above threshold {RELEVANCE_THRESHOLD}, all scores: "
            f"{[d.get('score', 0):.3f for d in docs[:5]]}"
        )
        return "No relevant documents found.", False, []

    log_debug(
        DEBUG_CONTEXT, "📋 CONTEXT",
        f"Filtered {len(relevant_docs)}/{len(docs)} docs (threshold={RELEVANCE_THRESHOLD})"
    )

    # Format từng doc
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.get("metadata", {}).get("source", "Unknown")
        score = doc.get("score", 0)
        chunk_idx = doc.get("metadata", {}).get("chunk_index", "?")
        text = doc.get("text", "")

        # Thêm thông tin về loại score nếu có (hybrid search)
        score_info = f"relevance: {score:.2f}"
        if "dense_score" in doc and "sparse_score" in doc:
            score_info += f", dense: {doc['dense_score']:.2f}, sparse: {doc['sparse_score']:.2f}"

        context_parts.append(
            f"[{i}] Source: {source} (chunk {chunk_idx}, {score_info})\n{text}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    # Log context khi DEBUG_CONTEXT
    if DEBUG_CONTEXT:
        log_debug(DEBUG_CONTEXT, "📋 CONTEXT", f"Context length: {len(context_str)} chars")
        log_debug(DEBUG_CONTEXT, "📋 CONTEXT", f"Context preview:\n{context_str[:500]}...")

    return context_str, True, relevant_docs


# ============================================
# Prompt Building
# ============================================

def build_prompt(query: str, context: str, lang: str) -> List[Dict]:
    """
    Build prompt messages cho LLM.

    Cấu trúc messages:
    1. System message: Hướng dẫn role và rules
    2. User message: Context + Question

    Args:
        query: Câu hỏi của user
        context: Context đã format từ retrieved docs
        lang: Ngôn ngữ ("vi" hoặc "en")

    Returns:
        List[Dict]: Messages cho OpenAI-compatible API
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]

    # Log prompt khi DEBUG_GENERATION
    if DEBUG_GENERATION:
        log_debug(DEBUG_GENERATION, "⚡ GENERATE", f"Prompt length: {len(user_content)} chars")
        log_debug(DEBUG_GENERATION, "⚡ GENERATE", f"System prompt: {system[:100]}...")

    return messages


# ============================================
# Generation Functions
# ============================================

def generate_with_retry(
    messages: List[Dict],
    max_tokens: int = 1024,
    temperature: float = TEMPERATURE
) -> str:
    """
    Generate response từ LLM với retry logic.

    Quá trình:
    1. Gọi vLLM qua OpenAI-compatible API
    2. Nếu fail, retry với exponential backoff
    3. Sau MAX_RETRIES lần fail, raise exception

    Retry delays: RETRY_DELAY * attempt (1s, 2s, 3s, ...)

    Args:
        messages: List messages (system + user)
        max_tokens: Số tokens tối đa cho response (default: 1024)
        temperature: Sampling temperature (default từ config)

    Returns:
        str: Generated response từ LLM

    Raises:
        Exception: Nếu fail sau MAX_RETRIES lần

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = generate_with_retry(messages, max_tokens=100)
    """
    last_error = None

    log_debug(
        DEBUG_GENERATION, "⚡ GENERATE",
        f"Calling LLM: model={LLM_MODEL}, max_tokens={max_tokens}, temp={temperature}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()

            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )

            result = response.choices[0].message.content
            elapsed = time.time() - start_time

            log_debug(
                DEBUG_GENERATION, "⚡ GENERATE",
                f"Response received: {len(result)} chars in {elapsed:.2f}s"
            )

            # Log response preview
            if DEBUG_GENERATION:
                log_debug(
                    DEBUG_GENERATION, "⚡ GENERATE",
                    f"Response preview: {result[:200]}..."
                )

            return result

        except Exception as e:
            last_error = e
            logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
            log_debug(
                DEBUG_GENERATION, "⚡ GENERATE",
                f"Attempt {attempt + 1} failed: {e}"
            )

            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_DELAY * (attempt + 1)
                log_debug(DEBUG_GENERATION, "⚡ GENERATE", f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

    logger.error(f"Generation failed after {MAX_RETRIES} attempts: {last_error}")
    raise last_error


# ============================================
# Main Chat Functions
# ============================================

def chat(
    query: str,
    top_k: int = TOP_K,
    max_tokens: int = 1024,
    use_hybrid: bool = True
) -> Dict:
    """
    Main chat function - Xử lý query và trả về response.

    RAG Pipeline:
    1. Detect ngôn ngữ
    2. Retrieve top-k docs từ vector store
    3. Filter docs theo relevance threshold
    4. Build prompt với context
    5. Generate response từ LLM
    6. Return kết quả với metadata

    Args:
        query: Câu hỏi của user
        top_k: Số docs retrieve (default từ config)
        max_tokens: Max tokens cho response (default: 1024)
        use_hybrid: Có dùng hybrid search không (default: True)

    Returns:
        Dict với các keys:
        - query: Câu query gốc
        - response: Câu trả lời từ LLM
        - language: "vi" hoặc "en"
        - sources: List sources được sử dụng
        - context_used: True nếu có sử dụng context
        - retrieval_info: Dict với timing và stats:
            - docs_found: Số docs tìm được
            - docs_relevant: Số docs vượt threshold
            - retrieve_time_ms: Thời gian retrieve (ms)
            - generate_time_ms: Thời gian generate (ms)
            - total_time_ms: Tổng thời gian (ms)
            - hybrid_search: Có dùng hybrid không

    Example:
        >>> result = chat("GPIO là gì?")
        >>> print(result["response"])
        GPIO (General Purpose Input/Output) là các chân đa năng...
        >>> print(result["sources"])
        [{"source": "stm32.pdf", "score": 0.85, "chunk_index": 5}]
    """
    start_time = time.time()
    lang = detect_language(query)

    log_debug(DEBUG_GENERATION, "⚡ CHAT", f"Query: '{query[:80]}...', lang={lang}")

    # Retrieve
    retrieve_start = time.time()
    retrieved_docs = retrieve_with_cache(query, top_k=top_k, use_hybrid=use_hybrid)
    retrieve_time = time.time() - retrieve_start

    log_debug(
        DEBUG_RETRIEVAL, "🔍 RETRIEVE",
        f"Retrieved {len(retrieved_docs)} docs in {retrieve_time*1000:.0f}ms"
    )

    # Format context
    context, has_relevant, relevant_docs = format_context(retrieved_docs)

    # No info message theo ngôn ngữ
    no_info_msg = (
        "Tôi không có thông tin về chủ đề này trong tài liệu hiện tại. "
        "Vui lòng upload tài liệu liên quan hoặc hỏi câu hỏi khác."
    ) if lang == "vi" else (
        "I don't have information about this topic in the current documents. "
        "Please upload relevant documents or ask another question."
    )

    # Nếu không có context relevant
    if not has_relevant:
        log_debug(DEBUG_GENERATION, "⚡ CHAT", "No relevant context, returning no_info message")
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

    log_debug(
        DEBUG_GENERATION, "⚡ GENERATE",
        f"Generated response in {generate_time*1000:.0f}ms"
    )

    # Check if model says no info
    if "NO_RELEVANT_INFO" in response:
        log_debug(DEBUG_GENERATION, "⚡ CHAT", "LLM returned NO_RELEVANT_INFO")
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

    # Build sources list
    sources = [{
        "source": d.get("metadata", {}).get("source"),
        "score": round(d.get("score", 0), 3),
        "chunk_index": d.get("metadata", {}).get("chunk_index")
    } for d in relevant_docs]

    total_time = time.time() - start_time

    log_debug(
        DEBUG_GENERATION, "⚡ CHAT",
        f"Chat completed: {total_time*1000:.0f}ms total, "
        f"{len(sources)} sources used"
    )

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
    """
    Streaming chat - Trả về response từng chunk.

    Tương tự chat() nhưng yield từng token thay vì trả về toàn bộ.
    Dùng cho real-time display trong UI.

    Args:
        query: Câu hỏi của user
        top_k: Số docs retrieve
        max_tokens: Max tokens cho response
        use_hybrid: Có dùng hybrid search không

    Yields:
        str: Từng chunk của response

    Example:
        >>> for chunk in chat_stream("GPIO là gì?"):
        ...     print(chunk, end="", flush=True)
    """
    lang = detect_language(query)

    log_debug(DEBUG_GENERATION, "⚡ STREAM", f"Stream query: '{query[:50]}...'")

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
        log_debug(DEBUG_GENERATION, "⚡ STREAM", "Starting stream generation...")

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        log_debug(DEBUG_GENERATION, "⚡ STREAM", "Stream completed")

    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        log_debug(DEBUG_GENERATION, "⚡ STREAM", f"Stream error: {e}")
        yield f"\n\n[Error: {str(e)}]"


# ============================================
# Debug/Utility Functions
# ============================================

def retrieve(query: str, top_k: int = TOP_K, use_hybrid: bool = True) -> List[Dict]:
    """
    Direct retrieve function (for debugging).

    Wrapper đơn giản của retrieve_with_cache, dùng để test retrieval
    độc lập với generation.

    Args:
        query: Câu query
        top_k: Số kết quả
        use_hybrid: Có dùng hybrid search không

    Returns:
        List[Dict]: Retrieved documents
    """
    return retrieve_with_cache(query, top_k=top_k, use_hybrid=use_hybrid)


def generate(messages: List[Dict], max_tokens: int = 1024) -> str:
    """
    Direct generate function (for debugging).

    Wrapper đơn giản của generate_with_retry, dùng để test generation
    độc lập với retrieval.

    Args:
        messages: Messages cho LLM
        max_tokens: Max tokens

    Returns:
        str: Generated response
    """
    return generate_with_retry(messages, max_tokens=max_tokens)
