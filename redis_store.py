"""Redis storage cho raw documents với connection pooling"""
"""redis_store.py"""
import redis
from redis import ConnectionPool
import json
import logging
from typing import Optional, List, Dict
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger(__name__)


class RedisStore:
    """Redis store với connection pooling và error handling"""
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_redis()
        return cls._instance
    
    def _init_redis(self):
        """Initialize Redis với connection pool"""
        logger.info(f"Connecting to Redis: {REDIS_HOST}:{REDIS_PORT}")
        
        self._pool = ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            max_connections=10,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        self.client = redis.Redis(connection_pool=self._pool)
        
        try:
            self.client.ping()
            logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def _get_client(self) -> redis.Redis:
        """Get Redis client với reconnection"""
        try:
            self.client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            logger.warning("Redis reconnecting...")
            self.client = redis.Redis(connection_pool=self._pool)
        return self.client
    
    def store_document(self, doc_id: str, data: Dict) -> bool:
        """Lưu raw document data"""
        try:
            client = self._get_client()
            key = f"doc:{doc_id}"
            
            # Store document data
            client.set(key, json.dumps(data, ensure_ascii=False))
            
            # Add to index
            client.sadd("doc:index", doc_id)
            
            # Store searchable metadata separately
            meta_key = f"doc:meta:{doc_id}"
            metadata = {
                "filename": data.get("filename", ""),
                "doc_type": data.get("doc_type", ""),
                "chunk_count": data.get("chunk_count", 0),
                "ingested_at": data.get("metadata", {}).get("ingested_at", "")
            }
            client.hset(meta_key, mapping=metadata)
            
            logger.debug(f"Stored document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Redis store error: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Lấy document theo ID"""
        try:
            client = self._get_client()
            key = f"doc:{doc_id}"
            data = client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Lấy metadata nhẹ của document"""
        try:
            client = self._get_client()
            meta_key = f"doc:meta:{doc_id}"
            data = client.hgetall(meta_key)
            if data:
                data["id"] = doc_id
                # Convert chunk_count to int
                if "chunk_count" in data:
                    data["chunk_count"] = int(data["chunk_count"])
            return data if data else None
        except Exception as e:
            logger.error(f"Redis get metadata error: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Xóa document"""
        try:
            client = self._get_client()
            
            # Delete document data
            key = f"doc:{doc_id}"
            client.delete(key)
            
            # Delete metadata
            meta_key = f"doc:meta:{doc_id}"
            client.delete(meta_key)
            
            # Remove from index
            client.srem("doc:index", doc_id)
            
            logger.debug(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def list_documents(self) -> List[str]:
        """Liệt kê tất cả document IDs"""
        try:
            client = self._get_client()
            return list(client.smembers("doc:index"))
        except Exception as e:
            logger.error(f"Redis list error: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict]:
        """Lấy tất cả documents với metadata (lightweight)"""
        docs = []
        for doc_id in self.list_documents():
            # Lấy metadata nhẹ thay vì full document
            meta = self.get_document_metadata(doc_id)
            if meta:
                docs.append(meta)
        return docs
    
    def get_all_documents_full(self) -> List[Dict]:
        """Lấy tất cả documents với full data"""
        docs = []
        for doc_id in self.list_documents():
            doc = self.get_document(doc_id)
            if doc:
                doc["id"] = doc_id
                docs.append(doc)
        return docs
    
    def search_by_filename(self, filename_pattern: str) -> List[Dict]:
        """Tìm documents theo filename pattern"""
        results = []
        for doc_id in self.list_documents():
            meta = self.get_document_metadata(doc_id)
            if meta and filename_pattern.lower() in meta.get("filename", "").lower():
                results.append(meta)
        return results
    
    def get_stats(self) -> Dict:
        """Lấy thống kê Redis"""
        try:
            client = self._get_client()
            doc_count = client.scard("doc:index")
            info = client.info("memory")
            
            return {
                "document_count": doc_count,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected": True
            }
        except Exception as e:
            return {"error": str(e), "connected": False}


def get_redis_store() -> RedisStore:
    """Get singleton RedisStore instance"""
    return RedisStore()