"""
Query and Response Service
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config import settings
from app.services.embedding_service import EmbeddingService

# Configure logging
logger = logging.getLogger(__name__)


class QueryServiceError(Exception):
    """Raised when query processing fails."""
    pass


class QueryService:
    """Service for processing queries using RAG with modern LCEL approach."""
    
    # Modern prompt template using ChatPromptTemplate
    PROMPT_TEMPLATE = """You are an expert document analyst. Use the following pieces of context to answer the question at the end.

If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide citations to the specific documents and sections you used for your answer.
Be concise but thorough in your response.

Context:
{context}

Question: {question}

Helpful Answer (with citations):"""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize query service.
        
        Args:
            embedding_service: Initialized EmbeddingService instance
        """
        self.embedding_service = embedding_service
        self.llm = self._initialize_llm()
        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        
        # Query statistics
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0
        }
        
        logger.info("QueryService initialized with LCEL approach")
    
    def _initialize_llm(self) -> ChatOllama:
        """Initialize local LLM via Ollama."""
        try:
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                num_ctx=settings.LLM_CONTEXT_WINDOW,
            )
        except Exception as e:
            raise QueryServiceError(f"Failed to initialize LLM: {str(e)}")
    
    def check_llm_connection(self) -> bool:
        """Verify connection to Ollama LLM."""
        try:
            response = self.llm.invoke("Hello, this is a test.")
            return bool(response and response.content)
        except Exception as e:
            logger.error(f"LLM connection check failed: {e}")
            return False
    
    def create_rag_chain(self, k: Optional[int] = None):
        """
        Create a RAG chain using LCEL (LangChain Expression Language).
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            A tuple of (rag_chain, retriever)
        """
        # Get retriever from vector store
        retriever = self.embedding_service.get_retriever(k)
        
        # Format documents for context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Modern LCEL chain
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever
    
    def answer_query(
        self,
        query: str,
        include_sources: bool = True,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG.
        
        Args:
            query: User question
            include_sources: Whether to include source citations
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and sources
            
        Raises:
            QueryServiceError: If query processing fails
        """
        start_time = time.time()
        
        try:
            # Update stats
            self.query_stats["total_queries"] += 1
            
            # Create RAG chain
            rag_chain, retriever = self.create_rag_chain(k)
            
            # Get response using LCEL
            answer = rag_chain.invoke(query)
            
            # Calculate timing
            response_time = time.time() - start_time
            
            # Get sources if requested
            sources = []
            if include_sources:
                # Retrieve documents for source information
                docs = retriever.invoke(query)
                
                for doc in docs:
                    sources.append({
                        "content": self._truncate_content(doc.page_content, 300),
                        "metadata": doc.metadata,
                        "relevance_score": doc.metadata.get("relevance_score", 0)
                    })
            
            # Update stats
            self.query_stats["successful_queries"] += 1
            self._update_avg_response_time(response_time)
            
            # Prepare response
            result = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(response_time, 3),
                "model_used": settings.OLLAMA_LLM_MODEL,
                "success": True
            }
            
            return result
            
        except Exception as e:
            # Update stats
            self.query_stats["failed_queries"] += 1
            logger.error(f"Query failed: {e}", exc_info=True)
            
            raise QueryServiceError(f"Query failed: {str(e)}")
    
    def batch_answer_queries(
        self, 
        queries: List[str],
        include_sources: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple queries in batch.
        """
        results = []
        
        for query in queries:
            try:
                result = self.answer_query(query, include_sources)
                results.append(result)
            except QueryServiceError as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query processing statistics."""
        return {
            **self.query_stats,
            "embedding_stats": self.embedding_service.get_collection_stats(),
            "llm_model": settings.OLLAMA_LLM_MODEL
        }
    
    def _truncate_content(self, content: str, max_length: int = 300) -> str:
        """Truncate content to specified length with ellipsis."""
        if len(content) <= max_length:
            return content
        
        return content[:max_length] + "..."
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time."""
        current_avg = self.query_stats["avg_response_time"]
        total_queries = self.query_stats["total_queries"]
        
        if total_queries > 0:
            new_avg = (
                (current_avg * (total_queries - 1) + response_time) / total_queries
            )
            self.query_stats["avg_response_time"] = round(new_avg, 3)