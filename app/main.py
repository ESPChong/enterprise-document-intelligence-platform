"""
Enterprise Document Intelligence Platform - Main Application
"""

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

from app.config import settings
from app.services.document_service import DocumentService, DocumentProcessingError
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.query_service import QueryService, QueryServiceError

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize services
    app.state.document_service = DocumentService()
    
    try:
        app.state.embedding_service = EmbeddingService()
        app.state.query_service = QueryService(app.state.embedding_service)
        
        # Check connections
        if app.state.embedding_service.check_connection():
            logger.info("✅ Ollama embedding service connected")
        else:
            logger.warning("⚠️ Ollama embedding service not responding")
            
        if app.state.query_service.check_llm_connection():
            logger.info("✅ Ollama LLM connected")
        else:
            logger.warning("⚠️ Ollama LLM not responding")
            
    except Exception as e:
        logger.warning(f"Service initialization warning: {e}")
        # Initialize with basic services
        app.state.embedding_service = None
        app.state.query_service = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade RAG system for enterprise document processing",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Middleware for request logging ----
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(duration)
    
    return response


# ---- Exception handlers ----
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ---- Health Check ----
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "document_service": True,
            "embedding_service": app.state.embedding_service is not None,
            "query_service": app.state.query_service is not None
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# ---- Document Routes ----
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    """
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.pdf', '.docx', '.txt']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: PDF, DOCX, TXT"
            )
        
        # Check file size
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_ext
        ) as tmp_file:
            tmp_file.write(contents)
            temp_path = tmp_file.name
        
        try:
            # Process document
            chunks = app.state.document_service.process_file(temp_path)
            
            # Add to vector store if available
            if app.state.embedding_service:
                doc_ids = app.state.embedding_service.add_documents(chunks)
            else:
                doc_ids = []
            
            # Get statistics
            stats = app.state.document_service.get_document_stats(chunks)
            
            return {
                "status": "success",
                "filename": file.filename,
                "document_id": stats.get("document_id"),
                "chunks_processed": len(chunks),
                "total_characters": stats.get("total_size_chars"),
                "avg_chunk_size": stats.get("avg_chunk_size"),
                "message": "Document processed successfully"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
        
    except HTTPException:
        raise
    except DocumentProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/documents/stats")
async def get_document_stats():
    """Get vector store statistics."""
    try:
        if not app.state.embedding_service:
            raise HTTPException(
                status_code=503,
                detail="Embedding service not available"
            )
        
        stats = app.state.embedding_service.get_collection_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        if not app.state.embedding_service:
            raise HTTPException(
                status_code=503,
                detail="Embedding service not available"
            )
        
        success = app.state.embedding_service.delete_document(document_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Document {document_id} deleted",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Query Routes ----
@app.post("/queries/ask")
async def ask_question(query: dict):
    """
    Answer a question using RAG.
    """
    try:
        if not app.state.query_service:
            raise HTTPException(
                status_code=503,
                detail="Query service not available"
            )
        
        question = query.get("question", "").strip()
        
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question is required"
            )
        
        # Get answer
        result = app.state.query_service.answer_query(question)
        
        return result
        
    except HTTPException:
        raise
    except QueryServiceError as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queries/stats")
async def get_query_stats():
    """Get query processing statistics."""
    try:
        if not app.state.query_service:
            raise HTTPException(
                status_code=503,
                detail="Query service not available"
            )
        
        stats = app.state.query_service.get_query_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )