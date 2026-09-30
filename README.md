# Enterprise Document Intelligence Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.5-1C3C3C.svg?logo=langchain)](https://python.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade RAG system for enterprise document processing with 100% local LLM inference**

A comprehensive solution for intelligent document analysis that processes business documents (PDF, DOCX, TXT) and provides context-aware answers with full citations—all while maintaining complete data privacy through local model execution.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Data Flow](#data-flow)
- [Production Considerations](#production-considerations)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Usage](#usage)
- [Project Status](#project-status)
- [Roadmap](#roadmap)
- [License](#license)


## Project Overview

### What This System Does

The Enterprise Document Intelligence Platform is a production-grade question-answering system that allows businesses to upload documents, automatically extract their content, and then ask natural language questions about those documents. The system provides accurate answers with citations back to the original source material, making it suitable for compliance-heavy environments like banking, legal, and finance.

The core problem this solves is that enterprises have thousands of documents (PDFs, Word files, text reports) containing critical information, but that information is locked away and difficult to access. Instead of having employees manually search through documents, this system uses AI to understand the content and answer questions directly—while keeping all data completely local for privacy and security.

In practical terms, a user could upload an annual report, a set of contracts, or compliance documents, then ask questions like "What was the revenue growth in Q3?" or "What are the termination clauses in this agreement?" and receive precise answers with references to the exact pages and sections where the information was found.

### Business Problem Solved

Hong Kong enterprises across banking, retail, property, and logistics sectors struggle with:
- **Document overload**: Thousands of PDFs, contracts, and reports with critical information buried inside
- **Compliance requirements**: Need for auditable answers with provenance
- **Data privacy concerns**: Cannot use cloud APIs for sensitive documents
- **Efficiency gaps**: Manual document review is slow and error-prone

### The Solution

| Business Need | Our Solution |
|--------------|--------------|
| **Document Processing** | Automated multi-format ingestion with intelligent chunking |
| **Intelligent Retrieval** | Hybrid semantic + keyword search with relevance scoring |
| **Reasoning & Analysis** | Context-aware answers with proper citations |
| **Audit & Compliance** | Full traceability from answer to source document |
| **Security** | 100% local processing—no data leaves your machine |


## Features

### Core Capabilities

- **Multi-format Document Support**: Process PDF, DOCX, and TXT files with automatic format detection
- **Intelligent Chunking**: Contextual text splitting with overlap strategies for optimal retrieval
- **Hybrid Search**: Combine semantic and keyword search for better accuracy
- **Citation Tracking**: Every answer includes source references with page numbers
- **Local Processing**: All data stays on your machine—no cloud APIs required
- **Fast Inference**: Optimized for Apple Silicon (M1/M2/M3) with quantized models

### Technical Features

- **Performance Monitoring**: Built-in metrics for latency, token usage, and retrieval accuracy
- **Error Handling**: Graceful degradation with fallback models
- **Async Processing**: Background document processing for large files
- **Analytics Dashboard**: Usage statistics and query analytics
- **Modern UI**: Clean, responsive interface with dark mode support


## Tech Stack

### Core AI Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Runtime** | Ollama | Local model execution |
| **Orchestration** | LangChain | AI workflow management |
| **Vector DB** | ChromaDB | Embedding storage & search |
| **Embeddings** | nomic-embed-text | Text vectorization |

### Application Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI | REST API framework |
| **Frontend** | Streamlit | Web interface |
| **Validation** | Pydantic | Data validation & settings |
| **Testing** | pytest | Test suite |

### Development Tools

| Component           | Technology   | Purpose               |
| ------------------- | ------------ | --------------------- |
| **Package Manager** | pip + venv   | Dependency management |
| **Task Runner**     | Make         | Build automation      |
| **Version Control** | Git + GitHub | Source control        |
| **Documentation**   | Markdown     | Project documentation |


## Architecture Overview

The system follows a layered architecture with four main components that work together to process documents and answer queries.

### 1. Frontend Layer (Streamlit)

The frontend is a web-based interface built with Streamlit that provides two primary functions: document upload and question answering. Users can drag and drop files (PDF, DOCX, TXT), monitor processing status, and type questions in a chat-like interface. The frontend communicates with the backend via HTTP requests and displays answers along with their source citations.

### 2. Backend Layer (FastAPI)

The backend is a REST API built with FastAPI that serves as the system's control plane. It handles file uploads, orchestrates the processing pipeline, manages queries, and returns results. The API includes health monitoring endpoints, request validation, error handling, and is designed to be deployed independently from the frontend. This separation allows the backend to serve multiple clients (web, mobile, or other services) and enables horizontal scaling in production.

### 3. AI Core (LangChain + Ollama)

This is where the actual intelligence happens. The AI core has three main responsibilities:

**Document Processing**: When a document is uploaded, the system extracts text, splits it into overlapping chunks (typically 1,000 characters with 200-character overlap), and attaches metadata like the source filename, page number, and processing timestamp. This chunking strategy ensures that relevant context isn't lost at chunk boundaries while keeping each piece small enough for efficient processing.

**Embedding and Storage**: Each text chunk is converted into a numerical vector (embedding) using the `nomic-embed-text` model running locally via Ollama. These vectors capture the semantic meaning of the text, allowing the system to find conceptually similar content even when exact keywords don't match. The vectors are stored in ChromaDB, a lightweight vector database that supports fast similarity search.

**Query Processing**: When a user asks a question, the system converts the question into an embedding, searches the vector database for the most relevant document chunks, and passes those chunks along with the original question to the LLM (Llama 3 8B, running locally). The LLM generates an answer based only on the provided context, which reduces hallucinations and ensures answers are grounded in the actual documents.

### 4. Infrastructure Layer

The infrastructure consists of two local services:

**Ollama** manages the LLM and embedding models. It handles model loading, inference, and memory management. Because it runs entirely on your machine, no data ever leaves your system. The models are quantized (reduced precision) versions that fit within 16GB of RAM while maintaining good quality.

**ChromaDB** provides persistent storage for the document embeddings and metadata. It supports filtering by document source, date, or other metadata fields, which enables features like "search only within this specific contract" or "only show results from last quarter's reports."## Architecture Overview


## Data Flow

### Document Ingestion Flow

When a user uploads a document, it flows through these steps:

1. The frontend sends the file to the backend via a POST request
2. The backend saves the file temporarily and determines its format (PDF, DOCX, TXT)
3. The appropriate loader extracts text content (PyPDF for PDFs, python-docx for Word files)
4. The text is split into overlapping chunks with metadata attached
5. Each chunk is sent to Ollama to generate an embedding vector
6. The vectors and metadata are stored in ChromaDB
7. The temporary file is deleted, and the backend confirms success to the frontend

### Query Processing Flow

When a user asks a question:

1. The frontend sends the question to the backend
2. The backend converts the question into an embedding vector
3. ChromaDB performs a similarity search to find the top 5 most relevant document chunks
4. The system constructs a prompt containing the question and the relevant context chunks
5. The prompt is sent to the LLM (Llama 3 8B) which generates an answer
6. The answer is returned along with metadata about which documents and sections were used
7. The frontend displays the answer and its citations

## Production Considerations

This architecture is designed to be production-ready while remaining runnable on consumer hardware. Key production features include:

- **Persistent storage**: ChromaDB saves to disk, so documents don't need to be re-processed after restarts
- **Error handling**: The API includes proper error responses and logging
- **Monitoring**: Health check endpoints allow orchestration systems to verify the service is running
- **Separation of concerns**: The API and frontend are separate services that can be scaled independently
- **Local-first design**: All processing happens on-device, eliminating network latency and privacy concerns

The system can be extended with authentication, rate limiting, and Docker containerization for deployment in enterprise environments, but the core architecture remains the same.


## Quick Start

### Prerequisites

- **Python 3.9+**
- **macOS with Apple Silicon** (M1/M2/M3) or **Linux**
- **16GB RAM** (minimum)
- **Ollama** installed ([Download here](https://ollama.ai))

Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/enterprise-document-intelligence.git
cd enterprise-document-intelligence

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
make install

# 4. Download AI models
make models

# 5. Start development servers
make dev
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "app_name": "Enterprise Document Intelligence",
#   "version": "1.0.0"
# }
```


## Detailed Setup

### Step 1: Environment Setup

```bash
# Create project directory
mkdir enterprise-document-intelligence
cd enterprise-document-intelligence

# Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Ollama Configuration

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama server
ollama serve

# Pull required models
ollama pull llama3:8b-instruct-q4_0      # Primary LLM (~4.5GB)
ollama pull nomic-embed-text              # Embedding model (~0.5GB)

# Optional: Lightweight backup model
ollama pull phi3:mini-3.8b-q4_0           # Quick tasks (~2.5GB)

# Verify models are available
ollama list
```

### Step 3: Project Configuration

Create a `.env` file in the project root:

```env
# Application Settings
APP_NAME="Enterprise Document Intelligence"
APP_VERSION="1.0.0"
DEBUG=True

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3:8b-instruct-q4_0
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=30

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=enterprise_documents

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENTS=1000
MAX_FILE_SIZE_MB=50

# Security (for future implementation)
SECRET_KEY=your-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Step 4: Verify Setup

```bash
# Run the test suite
make test

# Start the backend
make run

# In another terminal, test the health endpoint
curl http://localhost:8000/health
```


## Usage

### Starting the Application

```bash
# Development mode (auto-reload)
make dev

# Production mode
make run

# Start only frontend
make frontend

# Start only backend
make run
```

### API Examples

Complete API Usage Examples
#### Upload a Document

```bash
# Upload a PDF file
curl -X POST "http://localhost:8000/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

#### Ask a Question

```bash
# Query the system
curl -X POST "http://localhost:8000/queries/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key financial metrics from the annual report?"}'
```

#### Response Format

```json
{
  "answer": "Based on the annual report, the key financial metrics are...",
  "sources": [
    {
      "document": "annual_report_2024.pdf",
      "page": 15,
      "relevance_score": 0.92
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "processing_time_ms": 1200
}
```


## Project Status

### Completed Features

- [x] Project structure and configuration
- [x] FastAPI backend with health endpoints
- [x] Development environment setup
- [x] Makefile for task automation
- [x] Git repository initialization
- [x] Environment configuration management

### In Progress

- [ ] Document processing service
- [ ] Embedding and vector storage service
- [ ] Query and response service
- [ ] API routes for documents and queries
- [ ] Streamlit frontend
- [ ] Comprehensive test suite

### Planned Features

- [ ] Multi-agent workflows with LangGraph
- [ ] Fine-tuning support for domain-specific models
- [ ] User authentication and role-based access
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests


## Roadmap

### Phase 2: Core Features (Current)

**Document Processing Service**
- Multi-format loader support (PDF, DOCX, TXT)
- Intelligent text chunking with overlap
- Metadata extraction and validation
- Background processing for large files

**Embedding & Storage Service**
- Ollama integration for local embeddings
- ChromaDB vector store configuration
- Metadata filtering and search
- Similarity search with relevance scoring

**Query Service**
- RAG implementation with LangChain
- Citation tracking and source attribution
- Response validation and error handling
- Performance optimization

### Phase 3: Advanced Features

**Multi-Agent Workflows**
- LangGraph-based agent orchestration
- Tool integration (web search, calculators)
- Human-in-the-loop approval processes
- Complex reasoning pipelines

**Domain-Specific Tuning**
- Fine-tuning support for specialized models
- Domain adaptation for finance, legal, healthcare
- Custom evaluation metrics
- Performance benchmarking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.