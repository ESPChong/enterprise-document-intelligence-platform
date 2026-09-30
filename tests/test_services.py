"""
Tests for Document Intelligence Platform services.
"""

import pytest
import os
import tempfile
from pathlib import Path

from app.services.document_service import DocumentService, DocumentProcessingError


class TestDocumentService:
    """Test document processing service."""
    
    @pytest.fixture
    def document_service(self):
        return DocumentService()
    
    @pytest.fixture
    def sample_txt_file(self):
        """Create a sample text file for testing."""
        content = """This is a test document for the Enterprise Document Intelligence Platform.
        It contains multiple sentences. This is another sentence.

        This is a new paragraph with more content. The system should be able to
        process this and split it into appropriate chunks. This is more content
        to ensure we have enough text for proper chunking.

        Additional paragraph to make the document longer and ensure proper
        chunking behavior during testing. The system should handle this
        correctly and create multiple chunks from this content."""
        
        # Create a temporary file and properly close it before yielding
        fd, temp_path = tempfile.mkstemp(suffix='.txt')
        
        try:
            # Write content to file
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()  # Ensure content is written to disk
                os.fsync(f.fileno())  # Force write to disk
            
            yield temp_path
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_file_creation(self, sample_txt_file):
        """Verify the test fixture creates a non-empty file."""
        assert os.path.exists(sample_txt_file)
        assert os.path.getsize(sample_txt_file) > 0
        
        with open(sample_txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 0
        assert "test document" in content
    
    def test_validate_file_supported_txt(self, document_service, sample_txt_file):
        """Test file validation with supported format."""
        assert document_service.validate_file(sample_txt_file) is True
    
    def test_validate_file_unsupported(self, document_service):
        """Test file validation with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            unsupported_path = f.name
        
        try:
            assert document_service.validate_file(unsupported_path) is False
        finally:
            os.unlink(unsupported_path)
    
    def test_load_document_txt(self, document_service, sample_txt_file):
        """Test loading a text document."""
        documents = document_service.load_document(sample_txt_file)
        
        assert len(documents) > 0
        assert "test document" in documents[0].page_content
        assert documents[0].metadata["file_type"] == ".txt"
        assert "document_id" in documents[0].metadata
        assert documents[0].metadata["file_size_bytes"] > 0
    
    def test_chunk_documents(self, document_service, sample_txt_file):
        """Test document chunking."""
        documents = document_service.load_document(sample_txt_file)
        chunks = document_service.chunk_documents(documents)
        
        assert len(chunks) > 0
        assert all("chunk_id" in chunk.metadata for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)
        assert all("total_chunks" in chunk.metadata for chunk in chunks)
        assert all(chunk.page_content for chunk in chunks)  # No empty chunks
    
    def test_process_file_complete(self, document_service, sample_txt_file):
        """Test complete file processing pipeline."""
        chunks = document_service.process_file(sample_txt_file)
        
        assert len(chunks) > 0
        
        stats = document_service.get_document_stats(chunks)
        assert stats["total_chunks"] > 0
        assert stats["total_size_chars"] > 0
        assert "document_id" in stats
        assert "source_file" in stats
    
    def test_get_document_stats_empty(self, document_service):
        """Test statistics with no chunks."""
        stats = document_service.get_document_stats([])
        assert stats["total_chunks"] == 0
        assert stats["total_size_chars"] == 0
    
    def test_unsupported_file_type(self, document_service):
        """Test error handling for unsupported file types."""
        with pytest.raises(DocumentProcessingError):
            document_service.load_document("test.xyz")


class TestEmbeddingService:
    """Test embedding service."""
    
    @pytest.fixture
    def embedding_service(self):
        try:
            from app.services.embedding_service import EmbeddingService
            return EmbeddingService()
        except Exception:
            pytest.skip("Ollama or ChromaDB not available")
    
    def test_check_connection(self, embedding_service):
        """Test Ollama connection."""
        try:
            connected = embedding_service.check_connection()
            assert isinstance(connected, bool)
        except Exception:
            pytest.skip("Ollama not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])