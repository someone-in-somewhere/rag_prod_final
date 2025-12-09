"""ChromaDB Vector Store với Hybrid Search (Dense + Sparse)"""
"""vectorstore_chroma.py"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
import json
from config import CHROMA_DIR, CHROMA_COLLECTION, TOP_K, DENSE_WEIGHT, SPARSE_WEIGHT
from embedder import get_embedder


class VectorStore:
    """Vector store với hybrid search support"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_store()
        return cls._instance
    
    def _init_store(self):
        print(f"Initializing ChromaDB at: {CHROMA_DIR}")
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedder = get_embedder()
        
        # Collection cho dense vectors
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        
        # In-memory sparse index (simple inverted index)
        self.sparse_index: Dict[str, List[tuple]] = {}  # token -> [(doc_id, weight)]
        self.doc_sparse: Dict[str, Dict] = {}  # doc_id -> sparse_vector
        
        self._load_sparse_index()
        print(f"Collection '{CHROMA_COLLECTION}' ready, docs: {self.collection.count()}")
    
    def _load_sparse_index(self):
        """Load sparse index từ metadata trong ChromaDB"""
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
        except Exception as e:
            print(f"Warning: Could not load sparse index: {e}")
    
    def add_documents(self, chunks: List[Dict[str, Any]], use_sparse: bool = True) -> int:
        """
        Thêm documents vào vector store.
        chunks: [{"text": str, "metadata": dict}, ...]
        """
        if not chunks:
            return 0
        
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        
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
        
        return len(chunks)
    
    def _sparse_search(self, query_sparse: Dict, top_k: int) -> Dict[str, float]:
        """Tìm kiếm bằng sparse vector (BM25-like)"""
        scores = {}
        
        for token, query_weight in query_sparse.items():
            token_str = str(token)
            if token_str in self.sparse_index:
                for doc_id, doc_weight in self.sparse_index[token_str]:
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += query_weight * doc_weight
        
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
        Hybrid search: kết hợp dense và sparse.
        dense_weight + sparse_weight = 1.0
        """
        # Embed query
        query_result = self.embedder.embed_query(query, return_sparse=use_hybrid)
        query_dense = query_result["dense"].tolist()

        # Kiểm tra collection có documents không
        doc_count = self.collection.count()
        if doc_count == 0:
            return []

        # Dense search qua ChromaDB
        n_results = min(top_k * 2, doc_count) if use_hybrid else min(top_k, doc_count)
        dense_results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=max(n_results, 1),  # Đảm bảo >= 1
            include=["documents", "metadatas", "distances"]
        )
        
        # Parse dense results
        dense_scores = {}
        doc_data = {}
        
        for i in range(len(dense_results["ids"][0])):
            doc_id = dense_results["ids"][0][i]
            dense_scores[doc_id] = 1 - dense_results["distances"][0][i]  # cosine similarity
            doc_data[doc_id] = {
                "id": doc_id,
                "text": dense_results["documents"][0][i],
                "metadata": dense_results["metadatas"][0][i]
            }
        
        if not use_hybrid or not query_result.get("sparse"):
            # Chỉ dùng dense
            docs = []
            for doc_id in dense_results["ids"][0][:top_k]:
                doc = doc_data[doc_id]
                doc["score"] = dense_scores[doc_id]
                # Remove sparse_vector từ metadata để response gọn hơn
                if "sparse_vector" in doc["metadata"]:
                    del doc["metadata"]["sparse_vector"]
                docs.append(doc)
            return docs
        
        # Sparse search
        sparse_scores = self._sparse_search(query_result["sparse"], top_k * 2)
        
        # Normalize scores
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
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Fetch document data nếu chưa có
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
        
        # Build result
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
        
        return docs
    
    def delete_by_source(self, source: str) -> int:
        """Xóa documents theo source file"""
        try:
            results = self.collection.get(
                where={"source": source},
                include=["metadatas"]
            )
            
            if results["ids"]:
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
        """Lấy thống kê collection"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": CHROMA_COLLECTION,
            "sparse_index_tokens": len(self.sparse_index),
            "hybrid_enabled": True
        }


def get_vectorstore() -> VectorStore:
    return VectorStore()