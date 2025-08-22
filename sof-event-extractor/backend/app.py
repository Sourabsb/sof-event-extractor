"""
SoF Event Extractor Backend API
FastAPI application for processing maritime Statement of Facts documents
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path

# Import the new pipeline modules
from sof_pipeline import (
    IngestedDoc,
    process_uploaded_files,
    extract_events_and_summary,
    calculate_laytime,
    LaytimeResult,
    safe_float
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

# In-memory job storage (use database in production)
jobs = {}

class JobStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check for required API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. LLM extraction will not be available.")

print("✅ Maritime pipeline initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "SoF Event Extractor API is running", "status": "healthy"}

@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload and process a Statement of Facts document
    Supports PDF, DOCX, TXT, and image files
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg', '.tiff'}
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
            "summary": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(process_document_new_pipeline, job_id, file_path, file.filename)
        
        return {
            "job_id": job_id,
            "status": JobStatus.PROCESSING,
            "message": "Document uploaded successfully. Processing started."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# File upload wrapper to adapt to the new pipeline format
class FileUploadWrapper:
    def __init__(self, filepath: Path, filename: str):
        self.filepath = filepath
        self.name = filename
        
    def read(self):
        return self.filepath.read_bytes()

async def process_document_new_pipeline(job_id: str, file_path: Path, filename: str):
    """
    Background task to process the document using the new pipeline
    """
    try:
        logger.info(f"Starting processing for job {job_id}: {filename}")
        
        # Create a file wrapper for the new pipeline
        file_wrapper = FileUploadWrapper(file_path, filename)
        
        # Process the file using the new pipeline
        docs = process_uploaded_files([file_wrapper])
        
        if not docs:
            raise Exception("No text could be extracted from the uploaded document(s).")
        
        logger.info(f"Extracted text from {len(docs)} document(s)")
        
        # Extract events and summary using the new pipeline
        if not GROQ_API_KEY:
            raise Exception("GROQ_API_KEY is not configured. Please set the API key in the environment.")
        
        events_df, summary = extract_events_and_summary(docs, GROQ_API_KEY)
        
        logger.info(f"Extraction completed: {len(events_df)} events, summary keys: {list(summary.keys())}")
        
        # Convert DataFrame to list of dictionaries for JSON serialization
        if not events_df.empty:
            # Ensure datetime columns are properly formatted
            events_df['start_time_iso'] = pd.to_datetime(events_df['start_time_iso'], errors='coerce')
            events_df['end_time_iso'] = pd.to_datetime(events_df['end_time_iso'], errors='coerce')
            
            # Format datetime columns for JSON serialization
            events_df['start_time_formatted'] = events_df['start_time_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')
            events_df['end_time_formatted'] = events_df['end_time_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Replace NaT and NaN with None for JSON serialization
            events_df = events_df.where(pd.notnull(events_df), None)
            
            events_list = events_df.to_dict('records')
        else:
            events_list = []
        
        # Save results
        result_data = {
            "events": events_list,
            "summary": summary,
            "events_count": len(events_list)
        }
        
        result_file = RESULTS_DIR / f"{job_id}_results.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Update job status
        jobs[job_id].update({
            "status": JobStatus.COMPLETED,
            "events": events_list,
            "summary": summary,
            "events_df": events_df,  # Keep DataFrame for calculations
            "processed_at": datetime.now().isoformat(),
            "result_file": str(result_file)
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
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
            "summary": job.get("summary", {}),
            "processed_at": job["processed_at"]
        }

@app.post("/api/calculate-laytime/{job_id}")
async def calculate_laytime_endpoint(job_id: str, summary_data: Dict[str, Any]):
    """
    Calculate laytime based on extracted events and user-provided summary data
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    try:
        # Get the events DataFrame
        events_df = job.get("events_df")
        if events_df is None or events_df.empty:
            raise HTTPException(status_code=400, detail="No events available for calculation")
        
        # Perform laytime calculation
        result = calculate_laytime(summary_data, events_df)
        
        # Convert result to JSON-serializable format
        result_dict = {
            "laytime_allowed_days": result.laytime_allowed_days,
            "laytime_consumed_days": result.laytime_consumed_days,
            "laytime_saved_days": result.laytime_saved_days,
            "demurrage_due": result.demurrage_due,
            "dispatch_due": result.dispatch_due,
            "calculation_log": result.calculation_log,
            "events": result.events_df.to_dict('records') if not result.events_df.empty else []
        }
        
        # Save laytime results
        result_file = RESULTS_DIR / f"{job_id}_laytime.json"
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Update job with laytime results
        jobs[job_id]["laytime_result"] = result_dict
        
        return result_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Laytime calculation failed: {str(e)}")

@app.put("/api/update-events/{job_id}")
async def update_events(job_id: str, events: List[Dict[str, Any]]):
    """
    Update events for a job (after user edits)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    try:
        # Convert events list back to DataFrame
        events_df = pd.DataFrame(events)
        
        # Ensure proper datetime formatting
        if 'start_time_iso' in events_df.columns:
            events_df['start_time_iso'] = pd.to_datetime(events_df['start_time_iso'], errors='coerce')
        if 'end_time_iso' in events_df.columns:
            events_df['end_time_iso'] = pd.to_datetime(events_df['end_time_iso'], errors='coerce')
        
        # Update job data
        jobs[job_id]["events"] = events
        jobs[job_id]["events_df"] = events_df
        
        return {
            "job_id": job_id,
            "status": "success",
            "message": "Events updated successfully",
            "events_count": len(events)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update events: {str(e)}")

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
async def export_data(job_id: str, export_type: str = "csv", data_type: str = "events"):
    """
    Export processed events or laytime results as CSV or JSON
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    try:
        if data_type == "events":
            data = job["events"]
            if not data:
                raise HTTPException(status_code=404, detail="No events found")
            filename_prefix = "sof_events"
        elif data_type == "laytime":
            laytime_result = job.get("laytime_result")
            if not laytime_result:
                raise HTTPException(status_code=404, detail="No laytime calculation results found. Please calculate laytime first.")
            data = laytime_result["events"]
            filename_prefix = "laytime_calculation"
        else:
            raise HTTPException(status_code=400, detail="Invalid data_type. Use 'events' or 'laytime'")
        
        if export_type.lower() == "csv":
            # Convert to DataFrame and export as CSV
            df = pd.DataFrame(data)
            csv_file = RESULTS_DIR / f"{job_id}_{data_type}_export.csv"
            df.to_csv(csv_file, index=False)
            
            return FileResponse(
                csv_file,
                media_type='text/csv',
                filename=f"{filename_prefix}_{job_id[:8]}.csv"
            )
        
        elif export_type.lower() == "json":
            # Export as JSON
            json_file = RESULTS_DIR / f"{job_id}_{data_type}_export.json"
            with open(json_file, 'w') as f:
                if data_type == "laytime":
                    json.dump(job["laytime_result"], f, indent=2, default=str)
                else:
                    json.dump(data, f, indent=2, default=str)
            
            return FileResponse(
                json_file,
                media_type='application/json',
                filename=f"{filename_prefix}_{job_id[:8]}.json"
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
                "created_at": job["created_at"],
                "has_laytime_result": "laytime_result" in job
            }
            for job_id, job in jobs.items()
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
