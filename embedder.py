"""BGE-M3 Embedding module - Dense + Sparse hybrid"""
"""embedder.py"""
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Tuple
import numpy as np
from config import EMBEDDING_MODEL


class Embedder:
    """BGE-M3 embedder với hỗ trợ dense và sparse vectors"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance
    
    def _init_model(self):
        print(f"Loading BGE-M3 model: {EMBEDDING_MODEL}")
        self.model = BGEM3FlagModel(
            EMBEDDING_MODEL,
            use_fp16=True,
            device="cuda"
        )
        self.dense_dim = 1024  # BGE-M3 dense dimension
        print(f"BGE-M3 ready - Dense dim: {self.dense_dim}")
    
    def embed(self, texts: List[str], return_sparse: bool = False) -> Dict:
        """
        Embed list texts.
        Returns: {
            "dense": np.ndarray,  # (N, 1024)
            "sparse": List[Dict]  # optional, list of {token_id: weight}
        }
        """
        if not texts:
            return {"dense": np.array([]), "sparse": []}
        
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False
        )
        
        result = {"dense": np.array(output["dense_vecs"])}
        
        if return_sparse and "lexical_weights" in output:
            result["sparse"] = output["lexical_weights"]
        
        return result
    
    def embed_query(self, query: str, return_sparse: bool = False) -> Dict:
        """Embed single query"""
        result = self.embed([query], return_sparse=return_sparse)
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0] if result.get("sparse") else {}
        }
    
    def embed_dense(self, texts: List[str]) -> np.ndarray:
        """Backward compatible - chỉ trả về dense vectors"""
        return self.embed(texts, return_sparse=False)["dense"]
    
    def embed_query_dense(self, query: str) -> np.ndarray:
        """Backward compatible - chỉ trả về dense vector"""
        return self.embed_query(query, return_sparse=False)["dense"]


def get_embedder() -> Embedder:
    return Embedder()