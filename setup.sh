#!/bin/bash

# SoF Event Extractor - Development Setup Script

echo "🚢 Setting up SoF Event Extractor Development Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8+ and try again.${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 16+ and try again.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed. Please install npm and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Backend Setup
echo -e "${BLUE}🔧 Setting up Backend...${NC}"
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install spaCy English model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit backend/.env file with your API keys before running the application${NC}"
fi

# Create directories
mkdir -p uploads results

echo -e "${GREEN}✅ Backend setup complete${NC}"
echo ""

# Frontend Setup
echo -e "${BLUE}🔧 Setting up Frontend...${NC}"
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo -e "${GREEN}✅ Frontend setup complete${NC}"
echo ""

# Final instructions
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Edit backend/.env file with your API keys:"
echo "   - OPENAI_API_KEY (required for GPT extraction)"
echo "   - AZURE_COGNITIVE_KEY (optional, for advanced OCR)"
echo ""
echo "2. Start the backend server:"
echo -e "   ${YELLOW}cd backend && source venv/bin/activate && uvicorn app:app --reload${NC}"
echo ""
echo "3. In a new terminal, start the frontend:"
echo -e "   ${YELLOW}cd frontend && npm start${NC}"
echo ""
echo "4. Open your browser to http://localhost:3000"
echo ""
echo -e "${BLUE}🚢 Happy Maritime Document Processing!${NC}"
