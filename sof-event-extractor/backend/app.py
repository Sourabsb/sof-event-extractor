"""
SoF Event Extractor Backend API
FastAPI application for processing maritime Statement of Facts documents
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

# Import our utility modules
from utils.pdf_parser import PDFParser
from utils.docx_parser import DocxParser
from utils.gpt_extractor import GPTExtractor
from utils.gemini_extractor import GeminiEmbeddingExtractor
# from utils.ocr_module import OCRProcessor  # Temporarily disabled

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SoF Event Extractor API",
    description="AI-powered maritime document processing for Statement of Facts",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize processors
pdf_parser = PDFParser()
docx_parser = DocxParser()

# Initialize extractors with error handling
try:
    gpt_extractor = GPTExtractor()
    print("✅ GPT Extractor initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: GPT Extractor failed to initialize: {e}")
    gpt_extractor = None

try:
    gemini_extractor = GeminiEmbeddingExtractor()
    print("✅ Gemini Embedding Extractor initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Gemini Extractor failed to initialize: {e}")
    gemini_extractor = None

ocr_processor = None  # OCRProcessor()  # Temporarily disabled

# In-memory job storage (use database in production)
jobs = {}

class JobStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "SoF Event Extractor API is running", "status": "healthy"}

@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload and process a Statement of Facts document
    Supports PDF, DOCX, and image files
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.tiff'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize job status
        jobs[job_id] = {
            "status": JobStatus.PROCESSING,
            "filename": file.filename,
            "file_path": str(file_path),
            "created_at": datetime.now().isoformat(),
            "events": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(process_document, job_id, file_path, file_extension)
        
        return {
            "job_id": job_id,
            "status": JobStatus.PROCESSING,
            "message": "Document uploaded successfully. Processing started."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def validate_maritime_events(events: List[Dict], text: str) -> bool:
    """
    Validate if extracted events contain meaningful maritime information
    
    Args:
        events: List of extracted events
        text: Original document text for additional validation
        
    Returns:
        bool: True if events contain valid maritime data, False otherwise
    """
    if not events:
        return False
    
    # Check for maritime-specific keywords in the document
    maritime_keywords = [
        'vessel', 'ship', 'maritime', 'nautical', 'port', 'harbor', 'harbour',
        'navigation', 'collision', 'grounding', 'incident', 'accident',
        'latitude', 'longitude', 'coordinates', 'knots', 'bearing',
        'crew', 'pilot', 'captain', 'bridge', 'engine', 'cargo',
        'statement of facts', 'sof', 'marine', 'ocean', 'sea', 'berth',
        'anchor', 'mooring', 'tug', 'arrival', 'departure', 'loading',
        'discharge', 'docking', 'berthing', 'terminal', 'wharf'
    ]
    
    text_lower = text.lower()
    maritime_keyword_count = sum(1 for keyword in maritime_keywords if keyword in text_lower)
    
    # More lenient requirement: at least 1 maritime keyword OR proper timestamps
    has_maritime_keywords = maritime_keyword_count >= 1
    
    # Check if events have meaningful content (not just system errors)
    valid_events = 0
    for event in events:
        event_type = event.get('event', '').lower()
        description = event.get('description', '').lower()
        
        # Skip system errors and configuration errors
        if any(word in event_type for word in ['error', 'warning', 'system', 'configuration']):
            continue
        
        # Count events with any timestamp or location information
        start_time = event.get('start', '')
        location = event.get('location', '')
        
        # Accept events with either timestamps or specific locations
        has_timestamp = start_time and start_time not in ['Not Available', 'Unknown', '']
        has_location = location and location not in ['Not Available', 'Unknown', '', 'Port']
        
        # Check for maritime event types (more inclusive)
        maritime_event_types = [
            'grounding', 'collision', 'fire', 'explosion', 'machinery',
            'cargo', 'structural', 'oil spill', 'overboard', 'navigation',
            'weather', 'port', 'incident', 'accident', 'damage',
            'arrival', 'departure', 'berthing', 'unberthing', 'loading',
            'discharge', 'pilot', 'anchor', 'mooring', 'tug'
        ]
        
        has_maritime_content = any(
            maritime_type in event_type or maritime_type in description
            for maritime_type in maritime_event_types
        )
        
        # Accept events that have either maritime content OR proper timestamps/locations
        if has_maritime_content or has_timestamp or has_location:
            valid_events += 1
    
    # More lenient validation: require either maritime keywords OR valid events
    has_valid_content = has_maritime_keywords or valid_events > 0
    
    logger.info(f"Validation: keywords={maritime_keyword_count}, valid_events={valid_events}, result={has_valid_content}")
    return has_valid_content

async def process_document(job_id: str, file_path: Path, file_extension: str):
    """
    Background task to process the document and extract events
    """
    try:
        # Extract text based on file type
        if file_extension == '.pdf':
            text = pdf_parser.extract_text(file_path)
        elif file_extension in ['.docx', '.doc']:
            text = docx_parser.extract_text(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff']:
            # text = await ocr_processor.extract_text_from_image(file_path)  # Temporarily disabled
            raise Exception("OCR processing temporarily disabled. Please use PDF or DOCX files.")
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
        
        if not text or len(text.strip()) < 10:
            raise Exception("No readable text found in document")
        
        # Extract events using available extractors (Gemini primary, GPT fallback)
        events = []
        
        try:
            # Try Gemini extraction first for better accuracy and speed
            if gemini_extractor and gemini_extractor.gemini_available:
                events = await gemini_extractor.extract_events_with_gemini(text)
                logger.info(f"Gemini extraction completed: {len(events)} events found")
            else:
                logger.info("Gemini extractor not available, skipping...")
            
            # If Gemini extraction yields few results or is unavailable, use GPT
            if len(events) < 3 and gpt_extractor:
                gpt_events = await gpt_extractor.extract_events(text)
                logger.info(f"GPT extraction completed: {len(gpt_events)} events found")
                
                # Merge results, prioritizing Gemini events
                combined_events = events + gpt_events
                
                # Remove duplicates based on event type and timing
                unique_events = []
                seen_events = set()
                
                for event in combined_events:
                    event_key = f"{event.get('event', '').lower()}_{event.get('start', '')}"
                    if event_key not in seen_events:
                        unique_events.append(event)
                        seen_events.add(event_key)
                
                events = unique_events[:10]  # Limit to top 10 events
            
            # Validate if the extracted events contain meaningful maritime data
            if not validate_maritime_events(events, text):
                logger.warning("No valid maritime events found in document")
                events = [{
                    "event": "Document Validation Warning",
                    "description": "The uploaded document does not contain required maritime event details (event types, start/end times, or locations). Please ensure the document is a proper Statement of Facts or maritime incident report.",
                    "start": "Not Available",
                    "end": "Not Available", 
                    "location": "Not Available",
                    "severity": "Warning",
                    "suggestion": "Upload a document containing maritime incidents with proper timestamps and event descriptions"
                }]
                
        except Exception as extraction_error:
            logger.warning(f"Primary extraction failed: {extraction_error}")
            
            # Final fallback: try GPT if available
            if gpt_extractor:
                try:
                    logger.info("Using GPT as final fallback...")
                    events = await gpt_extractor.extract_events(text)
                except Exception as gpt_error:
                    logger.error(f"GPT fallback also failed: {gpt_error}")
                    # Return validation warning instead of mock events
                    events = [{
                        "event": "Document Validation Warning",
                        "description": "The uploaded document does not contain recognizable maritime event details (event types, start/end times, or locations). Please ensure the document is a proper Statement of Facts or maritime incident report.",
                        "start": "Not Available",
                        "end": "Not Available", 
                        "location": "Not Available",
                        "severity": "Warning",
                        "suggestion": "Upload a document containing maritime incidents with proper timestamps and event descriptions"
                    }]
            else:
                # No extractors available
                events = [{
                    "event": "Document Validation Warning",
                    "description": "The uploaded document does not contain recognizable maritime event details. Please ensure the document is a proper Statement of Facts or maritime incident report with timestamps and event descriptions.",
                    "start": "Not Available",
                    "end": "Not Available",
                    "location": "Not Available",
                    "severity": "Warning",
                    "suggestion": "Upload a document containing maritime incidents with proper timestamps and event descriptions"
                }]
        
        # Save results
        result_file = RESULTS_DIR / f"{job_id}_results.json"
        with open(result_file, 'w') as f:
            json.dump(events, f, indent=2)
        
        # Update job status
        jobs[job_id].update({
            "status": JobStatus.COMPLETED,
            "events": events,
            "processed_at": datetime.now().isoformat(),
            "result_file": str(result_file)
        })
        
    except Exception as e:
        jobs[job_id].update({
            "status": JobStatus.FAILED,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """
    Get processing results for a specific job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == JobStatus.PROCESSING:
        return {
            "job_id": job_id,
            "status": JobStatus.PROCESSING,
            "message": "Document is still being processed"
        }
    elif job["status"] == JobStatus.FAILED:
        return {
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "error": job["error"]
        }
    else:
        return {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "filename": job["filename"],
            "events": job["events"],
            "processed_at": job["processed_at"]
        }

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """
    Get processing status for a specific job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "created_at": job["created_at"]
    }

@app.post("/api/export/{job_id}")
async def export_data(job_id: str, export_type: str = "csv"):
    """
    Export processed events as CSV or JSON
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    events = job["events"]
    if not events:
        raise HTTPException(status_code=404, detail="No events found")
    
    try:
        if export_type.lower() == "csv":
            # Convert to DataFrame and export as CSV
            df = pd.DataFrame(events)
            csv_file = RESULTS_DIR / f"{job_id}_export.csv"
            df.to_csv(csv_file, index=False)
            
            return FileResponse(
                csv_file,
                media_type='text/csv',
                filename=f"sof_events_{job_id[:8]}.csv"
            )
        
        elif export_type.lower() == "json":
            # Export as JSON
            json_file = RESULTS_DIR / f"{job_id}_export.json"
            with open(json_file, 'w') as f:
                json.dump(events, f, indent=2)
            
            return FileResponse(
                json_file,
                media_type='application/json',
                filename=f"sof_events_{job_id[:8]}.json"
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid export type. Use 'csv' or 'json'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/jobs")
async def list_jobs():
    """
    List all processing jobs (for debugging/admin)
    """
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "filename": job["filename"],
                "created_at": job["created_at"]
            }
            for job_id, job in jobs.items()
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
