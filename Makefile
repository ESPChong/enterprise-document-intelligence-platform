# ==============================================
# Development Commands
# ==============================================

.PHONY: help install run frontend dev test clean format lint ollama models setup

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make run         - Start FastAPI backend"
	@echo "  make frontend    - Start Streamlit frontend"
	@echo "  make dev         - Start both backend & frontend"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean cache files"
	@echo "  make ollama      - Start Ollama server"
	@echo "  make models      - Download Ollama models"

install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py

dev:
	@echo "Starting development servers..."
	@make run &
	@make frontend &
	@wait

test:
	pytest tests/ -v --cov=app

test-services:
	pytest tests/test_services.py -v

ollama:
	ollama serve

check-ollama:
	curl -f http://localhost:11434/api/tags || echo "❌ Ollama not running"

models:
	ollama pull llama3:8b-instruct-q4_0
	ollama pull nomic-embed-text
	@echo "✅ Models downloaded!"

setup: install models
	@echo "🎉 Setup complete! Run 'make dev' to start."

# Start all services for development
dev-all:
	@echo "Starting all services..."
	@ollama serve &
	@uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@streamlit run frontend/app.py &
	@wait

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -f .coverage
	@echo "🧹 Cleaned!"



