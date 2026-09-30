"""
Embedding and Vector Storage Service
Uses langchain-ollama and langchain-chroma for modern integration.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings


class EmbeddingServiceError(Exception):
    """Raised when embedding operations fail."""
    pass


class EmbeddingService:
    """Service for managing embeddings and vector storage."""
    
    def __init__(self):
        """Initialize embedding service with Ollama and ChromaDB."""
        try:
            # Initialize Ollama embeddings (modern package)
            self.embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_EMBEDDING_MODEL,
            )
            
            # Initialize ChromaDB (modern package)
            self.vectorstore = Chroma(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )
            
            # Track collection statistics
            self.collection_stats = {
                "total_documents": 0,
                "total_chunks": 0,
                "last_updated": None
            }
            
        except Exception as e:
            raise EmbeddingServiceError(
                f"Failed to initialize embedding service: {str(e)}"
            )
    
    def check_connection(self) -> bool:
        """Verify connection to Ollama embedding service."""
        try:
            # Test embedding generation
            test_text = "connection test"
            test_embedding = self.embeddings.embed_query(test_text)
            
            # Verify we got a valid embedding
            if test_embedding and len(test_embedding) > 0:
                return True
            return False
            
        except Exception as e:
            raise EmbeddingServiceError(
                f"Failed to connect to Ollama: {str(e)}"
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store with embeddings.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs added
            
        Raises:
            EmbeddingServiceError: If adding fails
        """
        if not documents:
            return []
        
        try:
            # Generate unique IDs for each document
            ids = [
                doc.metadata.get("chunk_id", f"doc_{i}_{datetime.now().timestamp()}")
                for i, doc in enumerate(documents)
            ]
            
            # Add to vector store
            self.vectorstore.add_documents(
                documents=documents,
                ids=ids
            )
            
            # Update statistics
            self._update_stats(len(documents))
            
            return ids
            
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to add documents: {str(e)}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter
            )
            
            # Convert to Document list with scores in metadata
            documents = []
            for doc, score in results:
                doc.metadata["relevance_score"] = score
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise EmbeddingServiceError(f"Search failed: {str(e)}")
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        """
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter
            )
            return results
            
        except Exception as e:
            raise EmbeddingServiceError(f"Search with scores failed: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if successful
        """
        try:
            # Delete by metadata filter
            self.vectorstore._collection.delete(
                where={"document_id": document_id}
            )
            return True
            
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to delete document: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get total number of chunks in the collection."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            count = self.get_document_count()
            
            return {
                "total_chunks": count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "persist_directory": settings.CHROMA_PERSIST_DIR,
                "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
                "last_updated": self.collection_stats.get("last_updated")
            }
            
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to get stats: {str(e)}")
    
    def _update_stats(self, chunks_added: int):
        """Update internal statistics."""
        self.collection_stats["total_chunks"] += chunks_added
        self.collection_stats["last_updated"] = datetime.now().isoformat()
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete all documents
            self.vectorstore._collection.delete(where={})
            self.collection_stats["total_chunks"] = 0
            return True
            
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to clear collection: {str(e)}")
    
    def get_retriever(self, k: Optional[int] = None):
        """
        Get a retriever object for use with RAG chains.
        
        Args:
            k: Number of documents to retrieve (defaults to settings)
            
        Returns:
            A retriever object
        """
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k or settings.RETRIEVER_K,
                "fetch_k": settings.RETRIEVER_FETCH_K
            }
        )