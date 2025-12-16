"""
BGE-M3 Embedding Module
=======================
Module xử lý embedding văn bản sử dụng model BGE-M3.

BGE-M3 là model embedding đa ngôn ngữ hỗ trợ:
- Dense embedding: Vector 1024 chiều, dùng cho semantic similarity
- Sparse embedding (Lexical weights): Tương tự BM25, dùng cho keyword matching
- ColBERT vectors: Multi-vector representation (không dùng trong hệ thống này)

Hybrid Search:
- Kết hợp dense + sparse để cải thiện retrieval
- Dense: Tốt cho semantic (nghĩa tương đồng)
- Sparse: Tốt cho keyword exact match (thuật ngữ kỹ thuật, tên thanh ghi)

Singleton Pattern:
- Class Embedder sử dụng singleton để tránh load model nhiều lần
- Model được cache trong _instance

Sử dụng:
    from embedder import get_embedder
    embedder = get_embedder()

    # Embed nhiều văn bản
    result = embedder.embed(["text1", "text2"], return_sparse=True)
    # result["dense"] = np.ndarray shape (2, 1024)
    # result["sparse"] = [{"token_id": weight}, ...]

    # Embed một query
    query_vec = embedder.embed_query("tìm kiếm gì đó", return_sparse=True)
    # query_vec["dense"] = np.ndarray shape (1024,)
    # query_vec["sparse"] = {"token_id": weight}
"""

from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict
import numpy as np
from datetime import datetime

from config import EMBEDDING_MODEL, DEBUG_EMBEDDING


def log_embedding_debug(message: str):
    """
    Log debug cho embedding process.

    Chỉ hiển thị khi DEBUG_EMBEDDING=True trong config.
    Hữu ích để theo dõi:
    - Số lượng text được embed
    - Kích thước vector output
    - Thời gian embedding

    Args:
        message: Nội dung log debug
    """
    if DEBUG_EMBEDDING:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🔢 EMBED: {message}")


class Embedder:
    """
    BGE-M3 Embedder với hỗ trợ Dense và Sparse vectors.

    Singleton class - chỉ có một instance duy nhất được tạo.

    Attributes:
        model: BGEM3FlagModel instance
        dense_dim: Kích thước dense vector (1024 cho BGE-M3)

    Model Configuration:
        - use_fp16=True: Sử dụng FP16 để giảm VRAM (~2GB thay vì 4GB)
        - device="cuda": Chạy trên GPU
    """
    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation.

        Returns:
            Embedder: Instance duy nhất của class
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """
        Khởi tạo BGE-M3 model.

        Quá trình:
        1. Load model từ HuggingFace (hoặc cache local)
        2. Chuyển sang FP16 để tiết kiệm VRAM
        3. Load lên GPU

        VRAM Usage: ~2GB với FP16
        """
        log_embedding_debug(f"Loading BGE-M3 model: {EMBEDDING_MODEL}")
        log_embedding_debug(f"Initializing model with FP16 on CUDA...")

        self.model = BGEM3FlagModel(
            EMBEDDING_MODEL,
            use_fp16=True,
            device="cuda"
        )
        self.dense_dim = 1024  # BGE-M3 dense dimension

        log_embedding_debug(f"BGE-M3 ready - Dense dim: {self.dense_dim}")

    def embed(self, texts: List[str], return_sparse: bool = False) -> Dict:
        """
        Embed một danh sách văn bản.

        Quá trình:
        1. Nhận list strings làm input
        2. Gọi model.encode() để tạo embeddings
        3. Trả về dense vectors (luôn có) và sparse vectors (nếu yêu cầu)

        Args:
            texts: Danh sách văn bản cần embed
            return_sparse: Có trả về sparse vectors không (default: False)

        Returns:
            Dict với các keys:
            - "dense": np.ndarray shape (N, 1024) - Dense vectors
            - "sparse": List[Dict] - Sparse vectors (nếu return_sparse=True)
                        Mỗi dict có dạng {token_id: weight}

        Example:
            >>> embedder = get_embedder()
            >>> result = embedder.embed(["Hello world", "Xin chào"], return_sparse=True)
            >>> result["dense"].shape
            (2, 1024)
            >>> len(result["sparse"])
            2

        Performance:
            - Batch size tự động được model xử lý
            - GPU memory tỷ lệ với số texts
        """
        if not texts:
            log_embedding_debug("Empty input, returning empty arrays")
            return {"dense": np.array([]), "sparse": []}

        log_embedding_debug(f"Embedding {len(texts)} texts, return_sparse={return_sparse}")

        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False  # Không dùng ColBERT vectors
        )

        result = {"dense": np.array(output["dense_vecs"])}

        if return_sparse and "lexical_weights" in output:
            result["sparse"] = output["lexical_weights"]
            log_embedding_debug(
                f"Output: dense shape={result['dense'].shape}, "
                f"sparse vectors={len(result['sparse'])}"
            )
        else:
            log_embedding_debug(f"Output: dense shape={result['dense'].shape}")

        return result

    def embed_query(self, query: str, return_sparse: bool = False) -> Dict:
        """
        Embed một query (câu hỏi/tìm kiếm).

        Wrapper của embed() cho single query, trả về vector thay vì list.

        Args:
            query: Câu query cần embed
            return_sparse: Có trả về sparse vector không

        Returns:
            Dict với các keys:
            - "dense": np.ndarray shape (1024,) - Dense vector
            - "sparse": Dict {token_id: weight} - Sparse vector (nếu return_sparse=True)

        Example:
            >>> vec = embedder.embed_query("GPIO là gì?", return_sparse=True)
            >>> vec["dense"].shape
            (1024,)
            >>> isinstance(vec["sparse"], dict)
            True
        """
        log_embedding_debug(f"Embedding query: '{query[:50]}...'")

        result = self.embed([query], return_sparse=return_sparse)
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0] if result.get("sparse") else {}
        }

    def embed_dense(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts và chỉ trả về dense vectors.

        Backward compatible function cho code cũ không dùng hybrid search.

        Args:
            texts: Danh sách văn bản cần embed

        Returns:
            np.ndarray: Dense vectors shape (N, 1024)
        """
        log_embedding_debug(f"embed_dense: {len(texts)} texts (dense only)")
        return self.embed(texts, return_sparse=False)["dense"]

    def embed_query_dense(self, query: str) -> np.ndarray:
        """
        Embed query và chỉ trả về dense vector.

        Backward compatible function cho code cũ không dùng hybrid search.

        Args:
            query: Câu query cần embed

        Returns:
            np.ndarray: Dense vector shape (1024,)
        """
        log_embedding_debug(f"embed_query_dense: '{query[:50]}...' (dense only)")
        return self.embed_query(query, return_sparse=False)["dense"]


def get_embedder() -> Embedder:
    """
    Factory function để lấy Embedder instance.

    Sử dụng function này thay vì gọi Embedder() trực tiếp
    để đảm bảo singleton pattern.

    Returns:
        Embedder: Singleton instance của Embedder

    Example:
        >>> embedder = get_embedder()
        >>> embedder2 = get_embedder()
        >>> embedder is embedder2
        True
    """
    return Embedder()
