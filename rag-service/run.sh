#!/bin/bash

# RAG Service Startup Script

echo "🚀 Starting RAG Web Crawler Service..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip and try again."
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    
    # Install Playwright browsers
    echo "🌐 Installing Playwright browsers..."
    python3 -m playwright install chromium
else
    echo "❌ requirements.txt not found. Please run this script from the rag-service directory."
    exit 1
fi

# Check if Ollama is running
echo "🤖 Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    echo "   ollama pull llama3.2"
    echo ""
    echo "Continuing anyway - the service will start but queries may fail..."
fi

# Create data directory
mkdir -p data

# Start the service
echo "🚀 Starting FastAPI server..."
echo "📖 API documentation will be available at: http://localhost:8000/docs"
echo "🔍 Health check available at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Run the application
python3 -m app.main
