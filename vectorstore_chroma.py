"""
ChromaDB Vector Store Module
============================
Module quản lý vector store với hỗ trợ Hybrid Search (Dense + Sparse).

Kiến trúc:
- ChromaDB: Lưu trữ dense vectors và metadata (persistent)
- In-memory Sparse Index: Inverted index cho sparse vectors (rebuild khi khởi động)

Hybrid Search:
- Dense Search: Cosine similarity trên ChromaDB
- Sparse Search: BM25-like scoring trên inverted index
- Combined Score: dense_weight * dense_score + sparse_weight * sparse_score

Tại sao Hybrid?
- Dense: Tốt cho semantic similarity (câu có nghĩa giống nhau)
- Sparse: Tốt cho exact keyword match (tên thanh ghi, thuật ngữ kỹ thuật)
- Ví dụ: "GPIOA_ODR" sẽ match tốt hơn với sparse search

Singleton Pattern:
- Class VectorStore sử dụng singleton
- ChromaDB client được khởi tạo một lần duy nhất

Sử dụng:
    from vectorstore_chroma import get_vectorstore

    vs = get_vectorstore()

    # Thêm documents
    chunks = [{"text": "...", "metadata": {...}}, ...]
    vs.add_documents(chunks, use_sparse=True)

    # Tìm kiếm
    results = vs.search("GPIO pin configuration", top_k=5, use_hybrid=True)
    # results = [{"id": ..., "text": ..., "score": ..., "metadata": ...}, ...]
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
import json
from datetime import datetime

from config import (
    CHROMA_DIR, CHROMA_COLLECTION, TOP_K,
    DENSE_WEIGHT, SPARSE_WEIGHT, DEBUG_RETRIEVAL
)
from embedder import get_embedder


def log_retrieval_debug(message: str):
    """
    Log debug cho retrieval/search process.

    Chỉ hiển thị khi DEBUG_RETRIEVAL=True trong config.
    Hữu ích để theo dõi:
    - Số documents tìm được
    - Scores của từng kết quả
    - Thời gian search

    Args:
        message: Nội dung log debug
    """
    if DEBUG_RETRIEVAL:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🔍 RETRIEVAL: {message}")


class VectorStore:
    """
    Vector Store với Hybrid Search support.

    Singleton class quản lý ChromaDB và sparse index.

    Attributes:
        client: ChromaDB PersistentClient
        collection: ChromaDB Collection cho dense vectors
        embedder: BGE-M3 Embedder instance
        sparse_index: Dict[str, List[tuple]] - Inverted index cho sparse search
                      {token: [(doc_id, weight), ...]}
        doc_sparse: Dict[str, Dict] - Sparse vectors của từng document
                    {doc_id: {token: weight, ...}}

    Storage:
        - Dense vectors + metadata: ChromaDB (persistent on disk)
        - Sparse index: In-memory (rebuild từ metadata khi khởi động)
    """
    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation.

        Returns:
            VectorStore: Instance duy nhất của class
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_store()
        return cls._instance

    def _init_store(self):
        """
        Khởi tạo ChromaDB và sparse index.

        Quá trình:
        1. Tạo ChromaDB PersistentClient (lưu trên disk)
        2. Lấy hoặc tạo collection với cosine distance
        3. Load embedder (BGE-M3)
        4. Rebuild sparse index từ metadata trong ChromaDB
        """
        print(f"Initializing ChromaDB at: {CHROMA_DIR}")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedder = get_embedder()

        # Collection cho dense vectors (cosine similarity)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}  # HNSW index với cosine distance
        )

        # In-memory sparse index (simple inverted index)
        # token -> [(doc_id, weight), ...]
        self.sparse_index: Dict[str, List[tuple]] = {}
        # doc_id -> sparse_vector
        self.doc_sparse: Dict[str, Dict] = {}

        self._load_sparse_index()
        print(f"Collection '{CHROMA_COLLECTION}' ready, docs: {self.collection.count()}")
        log_retrieval_debug(
            f"Initialized: {self.collection.count()} docs, "
            f"{len(self.sparse_index)} unique tokens in sparse index"
        )

    def _load_sparse_index(self):
        """
        Load sparse index từ metadata trong ChromaDB.

        Sparse vectors được lưu dưới dạng JSON string trong metadata
        của mỗi document. Khi khởi động, rebuild inverted index từ
        tất cả documents.

        Quá trình:
        1. Lấy tất cả documents từ ChromaDB
        2. Parse sparse_vector từ metadata
        3. Build inverted index: token -> [(doc_id, weight), ...]
        """
        try:
            all_docs = self.collection.get(include=["metadatas"])

            for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
                if metadata and "sparse_vector" in metadata:
                    sparse = json.loads(metadata["sparse_vector"])
                    self.doc_sparse[doc_id] = sparse

                    for token, weight in sparse.items():
                        if token not in self.sparse_index:
                            self.sparse_index[token] = []
                        self.sparse_index[token].append((doc_id, weight))

            log_retrieval_debug(
                f"Loaded sparse index: {len(self.doc_sparse)} docs, "
                f"{len(self.sparse_index)} unique tokens"
            )
        except Exception as e:
            print(f"Warning: Could not load sparse index: {e}")

    def add_documents(self, chunks: List[Dict[str, Any]], use_sparse: bool = True) -> int:
        """
        Thêm documents vào vector store.

        Quá trình:
        1. Embed tất cả texts (dense + optional sparse)
        2. Serialize sparse vectors vào metadata
        3. Update in-memory sparse index
        4. Add vào ChromaDB

        Args:
            chunks: List các chunks, mỗi chunk là dict:
                    {"text": str, "metadata": dict}
            use_sparse: Có tạo sparse vectors không (default: True)

        Returns:
            int: Số chunks đã thêm

        Example:
            >>> chunks = [
            ...     {"text": "GPIO configuration...", "metadata": {"source": "doc.pdf"}},
            ...     {"text": "Timer setup...", "metadata": {"source": "doc.pdf"}}
            ... ]
            >>> count = vs.add_documents(chunks, use_sparse=True)
            >>> print(f"Added {count} chunks")
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        log_retrieval_debug(f"Adding {len(chunks)} documents, use_sparse={use_sparse}")

        # Embed với cả dense và sparse
        embeddings_result = self.embedder.embed(texts, return_sparse=use_sparse)
        dense_embeddings = embeddings_result["dense"].tolist()

        # Xử lý sparse vectors
        if use_sparse and embeddings_result.get("sparse"):
            sparse_vectors = embeddings_result["sparse"]

            for i, (doc_id, sparse) in enumerate(zip(ids, sparse_vectors)):
                # Chuyển sparse vector thành serializable format
                sparse_dict = {str(k): float(v) for k, v in sparse.items()}
                metadatas[i]["sparse_vector"] = json.dumps(sparse_dict)

                # Update in-memory index
                self.doc_sparse[doc_id] = sparse_dict
                for token, weight in sparse_dict.items():
                    if token not in self.sparse_index:
                        self.sparse_index[token] = []
                    self.sparse_index[token].append((doc_id, weight))

        # Add vào ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=dense_embeddings,
            documents=texts,
            metadatas=metadatas
        )

        log_retrieval_debug(
            f"Added {len(chunks)} docs, total: {self.collection.count()}"
        )

        return len(chunks)

    def _sparse_search(self, query_sparse: Dict, top_k: int) -> Dict[str, float]:
        """
        Tìm kiếm bằng sparse vector (BM25-like).

        Scoring:
        - Với mỗi token trong query, tìm documents chứa token đó
        - Score = sum(query_weight * doc_weight) cho mỗi token match

        Args:
            query_sparse: Sparse vector của query {token: weight}
            top_k: Số kết quả tối đa (không dùng ở đây, để filter sau)

        Returns:
            Dict[str, float]: {doc_id: score} cho tất cả docs có match
        """
        scores = {}

        for token, query_weight in query_sparse.items():
            token_str = str(token)
            if token_str in self.sparse_index:
                for doc_id, doc_weight in self.sparse_index[token_str]:
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += query_weight * doc_weight

        log_retrieval_debug(
            f"Sparse search: {len(query_sparse)} query tokens, "
            f"matched {len(scores)} docs"
        )

        return scores

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        use_hybrid: bool = True,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT
    ) -> List[Dict]:
        """
        Hybrid search: kết hợp Dense và Sparse.

        Quá trình:
        1. Embed query (dense + sparse nếu hybrid)
        2. Dense search qua ChromaDB (cosine similarity)
        3. Sparse search qua inverted index (nếu hybrid)
        4. Normalize và combine scores
        5. Sort và trả về top_k

        Scoring formula (hybrid):
            combined_score = dense_weight * normalized_dense + sparse_weight * normalized_sparse

        Args:
            query: Câu query tìm kiếm
            top_k: Số kết quả trả về (default từ config)
            use_hybrid: Có dùng hybrid search không (default: True)
            dense_weight: Trọng số cho dense score (default: 0.7)
            sparse_weight: Trọng số cho sparse score (default: 0.3)
                          dense_weight + sparse_weight nên = 1.0

        Returns:
            List[Dict]: Danh sách kết quả, mỗi item có:
            - id: Document ID
            - text: Nội dung document
            - score: Combined score (hoặc dense score nếu không hybrid)
            - metadata: Metadata của document
            - dense_score: (chỉ có nếu hybrid) Dense similarity score
            - sparse_score: (chỉ có nếu hybrid) Sparse match score

        Example:
            >>> results = vs.search("GPIO input mode", top_k=5, use_hybrid=True)
            >>> for r in results:
            ...     print(f"{r['score']:.3f} - {r['metadata']['source']}")
        """
        log_retrieval_debug(
            f"Search: '{query[:50]}...', top_k={top_k}, "
            f"hybrid={use_hybrid}, weights=({dense_weight}/{sparse_weight})"
        )

        # Embed query
        query_result = self.embedder.embed_query(query, return_sparse=use_hybrid)
        query_dense = query_result["dense"].tolist()

        # Dense search qua ChromaDB
        # Lấy nhiều hơn nếu hybrid để có thể merge với sparse results
        # QUAN TRỌNG: n_results phải <= số docs trong collection, nếu không HNSW sẽ lỗi
        collection_count = self.collection.count()
        if collection_count == 0:
            log_retrieval_debug("Collection empty, returning empty results")
            return []

        n_results = min(top_k * 2 if use_hybrid else top_k, collection_count)

        dense_results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Parse dense results
        dense_scores = {}
        doc_data = {}

        for i in range(len(dense_results["ids"][0])):
            doc_id = dense_results["ids"][0][i]
            # ChromaDB trả về distance, convert thành similarity
            # cosine distance = 1 - cosine similarity
            dense_scores[doc_id] = 1 - dense_results["distances"][0][i]
            doc_data[doc_id] = {
                "id": doc_id,
                "text": dense_results["documents"][0][i],
                "metadata": dense_results["metadatas"][0][i]
            }

        log_retrieval_debug(f"Dense search: found {len(dense_scores)} docs")

        if not use_hybrid or not query_result.get("sparse"):
            # Chỉ dùng dense - trả về kết quả trực tiếp
            docs = []
            for doc_id in dense_results["ids"][0][:top_k]:
                doc = doc_data[doc_id]
                doc["score"] = dense_scores[doc_id]
                # Remove sparse_vector từ metadata để response gọn hơn
                if "sparse_vector" in doc["metadata"]:
                    del doc["metadata"]["sparse_vector"]
                docs.append(doc)

            log_retrieval_debug(
                f"Dense-only results: {len(docs)} docs, "
                f"top score={docs[0]['score']:.3f}" if docs else "no results"
            )

            return docs

        # Sparse search
        sparse_scores = self._sparse_search(query_result["sparse"], top_k * 2)

        # Normalize scores (để kết hợp được)
        max_dense = max(dense_scores.values()) if dense_scores else 1.0
        max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0

        # Combine scores
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}

        for doc_id in all_doc_ids:
            d_score = dense_scores.get(doc_id, 0.0) / max_dense
            s_score = sparse_scores.get(doc_id, 0.0) / max_sparse
            combined_scores[doc_id] = dense_weight * d_score + sparse_weight * s_score

        # Sort và lấy top_k
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Fetch document data nếu chưa có (từ sparse-only matches)
        missing_ids = [doc_id for doc_id, _ in sorted_docs if doc_id not in doc_data]
        if missing_ids:
            missing_data = self.collection.get(
                ids=missing_ids,
                include=["documents", "metadatas"]
            )
            for i, doc_id in enumerate(missing_data["ids"]):
                doc_data[doc_id] = {
                    "id": doc_id,
                    "text": missing_data["documents"][i],
                    "metadata": missing_data["metadatas"][i]
                }

        # Build result với full scoring info
        docs = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_data:
                doc = doc_data[doc_id].copy()
                doc["score"] = score
                doc["dense_score"] = dense_scores.get(doc_id, 0.0)
                doc["sparse_score"] = sparse_scores.get(doc_id, 0.0)
                # Remove sparse_vector từ metadata
                if "sparse_vector" in doc["metadata"]:
                    del doc["metadata"]["sparse_vector"]
                docs.append(doc)

        # Log top-k results khi DEBUG_RETRIEVAL
        if DEBUG_RETRIEVAL and docs:
            log_retrieval_debug(f"Hybrid results ({len(docs)} docs):")
            for i, d in enumerate(docs[:5]):  # Log top 5
                source = d["metadata"].get("source", "?")
                log_retrieval_debug(
                    f"  [{i+1}] score={d['score']:.3f} "
                    f"(dense={d['dense_score']:.3f}, sparse={d['sparse_score']:.3f}) "
                    f"- {source}"
                )

        return docs

    def delete_by_source(self, source: str) -> int:
        """
        Xóa documents theo source file.

        Quá trình:
        1. Tìm tất cả documents có metadata.source = source
        2. Remove từ sparse index (in-memory)
        3. Remove từ ChromaDB

        Args:
            source: Tên file nguồn (vd: "document.pdf")

        Returns:
            int: Số chunks đã xóa
        """
        try:
            results = self.collection.get(
                where={"source": source},
                include=["metadatas"]
            )

            if results["ids"]:
                log_retrieval_debug(f"Deleting {len(results['ids'])} docs with source={source}")

                # Remove từ sparse index
                for doc_id in results["ids"]:
                    if doc_id in self.doc_sparse:
                        sparse = self.doc_sparse[doc_id]
                        for token in sparse.keys():
                            if token in self.sparse_index:
                                self.sparse_index[token] = [
                                    (d, w) for d, w in self.sparse_index[token]
                                    if d != doc_id
                                ]
                        del self.doc_sparse[doc_id]

                # Remove từ ChromaDB
                self.collection.delete(ids=results["ids"])
                print(f"Deleted {len(results['ids'])} chunks with source={source}")
                return len(results["ids"])

            return 0
        except Exception as e:
            print(f"Delete error: {e}")
            return 0

    def get_stats(self) -> Dict:
        """
        Lấy thống kê của collection.

        Returns:
            Dict với các thông tin:
            - total_documents: Tổng số documents
            - collection_name: Tên collection
            - sparse_index_tokens: Số unique tokens trong sparse index
            - hybrid_enabled: Luôn True (hệ thống hỗ trợ hybrid)
        """
        return {
            "total_documents": self.collection.count(),
            "collection_name": CHROMA_COLLECTION,
            "sparse_index_tokens": len(self.sparse_index),
            "hybrid_enabled": True
        }


def get_vectorstore() -> VectorStore:
    """
    Factory function để lấy VectorStore instance.

    Sử dụng function này thay vì gọi VectorStore() trực tiếp
    để đảm bảo singleton pattern.

    Returns:
        VectorStore: Singleton instance của VectorStore

    Example:
        >>> vs = get_vectorstore()
        >>> vs2 = get_vectorstore()
        >>> vs is vs2
        True
    """
    return VectorStore()
