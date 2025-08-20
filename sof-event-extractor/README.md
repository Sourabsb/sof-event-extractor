# SoF Event Extractor 🚢

## Abstract

The **SoF Event Extractor** is an AI-powered maritime document processing system designed to automatically extract structured port events from unstructured Statement of Facts (SoF) documents. Built for the Integrated Maritime Exchange (IME) hackathon, this solution transforms manual document processing into an intelligent, automated workflow.

## Problem Statement

Maritime professionals spend countless hours manually extracting port events and timestamps from Statement of Facts documents (PDFs, Word docs, scanned images). This manual process is:
- Time-consuming and error-prone
- Difficult to scale across multiple documents
- Lacks standardized output formats
- Creates bottlenecks in maritime operations

## Solution

Our AI-powered system automatically:
- Extracts all port events with precise start/end timestamps
- Converts unstructured data into structured JSON/CSV formats
- Provides optional timeline visualization
- Supports multiple document formats (PDF, DOCX, scanned images)

## 🌊 Features

- **Smart Document Processing**: Handles PDF, DOCX, and scanned documents
- **AI-Powered Extraction**: Uses regex patterns, spaCy NLP, and GPT fallback
- **OCR Support**: Processes scanned documents with Azure Cognitive Services
- **Multiple Export Formats**: Download as JSON or CSV
- **Timeline Visualization**: Interactive event timeline (optional)
- **Maritime UI Theme**: Clean, professional interface with navy blue palette
- **Real-time Processing**: Fast, efficient document processing
- **Secure Upload**: Safe file handling with validation

## 🏗️ Architecture

![Architecture Diagram](demo_video_architecture.png)

### System Components:

1. **Frontend**: React + TailwindCSS
2. **Backend**: FastAPI (Python)
3. **AI Pipeline**: 
   - PDF/DOCX parsers
   - OCR processing (Azure Cognitive/Pytesseract)
   - Regex + spaCy + GPT fallback
4. **Storage**: Local uploads with optional Azure Blob
5. **Export**: Pandas-based CSV/JSON generation

## 🚀 Tech Stack

- **Frontend**: React 18, TailwindCSS, Axios
- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: spaCy, OpenAI GPT, Azure Cognitive Services
- **Document Processing**: PyPDF2, python-docx, Pillow
- **OCR**: Azure Computer Vision / Pytesseract
- **Export**: Pandas, JSON
- **Styling**: Inter font, Maritime color palette

## 📋 Setup Instructions

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
AZURE_COGNITIVE_KEY=your_azure_key
AZURE_COGNITIVE_ENDPOINT=your_azure_endpoint
```

5. Run the backend server:
```bash
uvicorn app:app --reload --port 8000
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

## 🎯 How to Run

1. **Start Backend**: Follow backend setup and run `uvicorn app:app --reload`
2. **Start Frontend**: Follow frontend setup and run `npm start`
3. **Upload Document**: Drag & drop or select your SoF document
4. **Process**: AI automatically extracts events and timestamps
5. **Export**: Download results as JSON or CSV
6. **Visualize**: View timeline representation (optional)

## 📊 API Endpoints

- `POST /api/upload` - Upload and process document
- `GET /api/result/{job_id}` - Get processing results
- `POST /api/export/{job_id}` - Export data (CSV/JSON)
- `GET /api/status/{job_id}` - Check processing status

## 🎬 Demo Video

[Demo Video Link](https://your-demo-video-link.com)

## 🔧 Development

### Project Structure
```
sof-event-extractor/
├── frontend/          # React application
├── backend/           # FastAPI server
├── scripts/           # Demo scripts
└── README.md          # This file
```

### Color Palette
- Navy Blue: `#001F3F`
- White: `#FFFFFF`
- Light Blue: `#F0F5FA`
- Accent Blue: `#0074D9`

## 🚢 Maritime Integration

This system is designed for easy integration with existing maritime workflows:
- RESTful API for system integration
- Standardized JSON/CSV output formats
- Scalable architecture for enterprise deployment
- Docker containerization ready

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 IME Hackathon Submission

Built for the Integrated Maritime Exchange hackathon, demonstrating how AI can revolutionize maritime document processing and operational efficiency.

---

**Team**: Maritime AI Solutions
**Contact**: [Your Contact Information]
**Demo**: [Live Demo URL]
