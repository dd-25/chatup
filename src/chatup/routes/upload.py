from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import tempfile
import os
from src.chatup.graph.data_ingestion import run_ingestion
from src.chatup.constants import UPLOAD_SETTINGS

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


class UploadResponse(BaseModel):
    """Response model for file upload endpoints."""
    status: str
    message: str
    filename: str
    namespace: str
    chunks_generated: int
    processing_stats: Dict[str, Any]
    error_details: Optional[str] = None


class UploadErrorResponse(BaseModel):
    """Error response model for upload failures."""
    status: str
    error: str
    filename: str
    namespace: str
    details: Optional[str] = None


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    namespace: str = Query(default="default", description="Namespace for document storage")
):
    """
    Upload and process a document through the LangGraph data ingestion pipeline.
    
    Supports multiple file formats:
    - PDF documents (.pdf)
    - Word documents (.docx)
    - Text files (.txt)
    - JSON files (.json)
    - CSV/TSV files (.csv, .tsv)
    
    Args:
        file: The uploaded file
        namespace: Pinecone namespace for storage (default: "default")
        
    Returns:
        Detailed processing results including statistics and chunk information
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    MAX_FILE_SIZE = UPLOAD_SETTINGS.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
    
    logger.info(f"Starting upload processing for file: {file.filename}")
    
    try:
        # Use SpooledTemporaryFile for memory-efficient file handling
        # Files smaller than max_size stay in memory, larger ones are written to disk
        with tempfile.SpooledTemporaryFile(
            max_size=UPLOAD_SETTINGS.DISK_THRESHOLD_MB * 1024 * 1024,  # 10MB threshold for memory vs disk
            mode='w+b'
        ) as spooled_file:
            
            # Stream file content into SpooledTemporaryFile
            file_size = 0
            chunk_size = UPLOAD_SETTINGS.STREAMING_CHUNK_SIZE_KB * 1024  # Convert KB to bytes
            
            # Reset file pointer to beginning
            await file.seek(0)
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                    
                file_size += len(chunk)
                
                # Check file size during streaming to prevent memory issues
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size ({file_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
                    )
                
                spooled_file.write(chunk)
            
            # Validate file size
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Empty file is not allowed"
                )
            
            # Validate file extension
            file_extension = '.' + file.filename.split('.')[-1].lower()
            
            if file_extension not in UPLOAD_SETTINGS.ALLOWED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Allowed formats: {', '.join(UPLOAD_SETTINGS.ALLOWED_FILE_TYPES)}"
                )
            
            # Reset spooled file pointer to beginning for reading
            spooled_file.seek(0)
            
            # Read file content efficiently
            file_bytes = spooled_file.read()
            
            # Log file processing info
            is_in_memory = not spooled_file._rolled
            storage_location = "memory" if is_in_memory else "temporary disk"
            logger.info(f"Processing {file.filename} ({file_size} bytes) stored in {storage_location} in namespace '{namespace}'")
            
            # Process through LangGraph pipeline
            result = run_ingestion(file_bytes, file.filename, namespace)
            
            if result['success']:
                # Successful processing
                response_data = UploadResponse(
                    status="success",
                    message=f"File '{file.filename}' processed successfully",
                    filename=file.filename,
                    namespace=namespace,
                    chunks_generated=len(result.get('chunks', [])),
                    processing_stats=result.get('processing_stats', {})
                )
                
                logger.info(f"Successfully processed {file.filename}: {len(result.get('chunks', []))} chunks generated")
                return JSONResponse(
                    status_code=200,
                    content=response_data.dict()
                )
            
            else:
                # Processing failed
                error_message = result.get('error_message', 'Unknown processing error')
                logger.error(f"Processing failed for {file.filename}: {error_message}")
                
                error_response = UploadErrorResponse(
                    status="error",
                    error=error_message,
                    filename=file.filename,
                    namespace=namespace,
                    details=str(result.get('processing_stats', {}))
                )
                
                return JSONResponse(
                    status_code=422,
                    content=error_response.dict()
                )
        
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error during file upload for {file.filename}:")
        
        error_response = UploadErrorResponse(
            status="error",
            error="Internal server error during file processing",
            filename=file.filename or "unknown",
            namespace=namespace,
            details=str(e)
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.get("/health")
async def upload_health():
    """Health check endpoint for upload service."""
    return {
        "status": "healthy",
        "service": "file_upload",
        "supported_formats": list(UPLOAD_SETTINGS.ALLOWED_FILE_TYPES),
        "max_file_size_mb": UPLOAD_SETTINGS.MAX_FILE_SIZE_MB,
        "memory_threshold_mb": 10  # SpooledFile memory threshold
    }


@router.get("/formats")
async def supported_formats():
    """Get information about supported file formats."""
    return {
        "supported_formats": {
            "pdf": {
                "description": "Portable Document Format",
                "extensions": [".pdf"],
                "notes": "Supports text extraction from PDF documents"
            },
            "docx": {
                "description": "Microsoft Word Document",
                "extensions": [".docx"],
                "notes": "Supports modern Word document format"
            },
            "text": {
                "description": "Plain Text",
                "extensions": [".txt"],
                "notes": "UTF-8 encoded text files"
            },
            "json": {
                "description": "JavaScript Object Notation",
                "extensions": [".json"],
                "notes": "Structured data with intelligent key-value extraction"
            },
            "csv": {
                "description": "Comma-Separated Values",
                "extensions": [".csv", ".tsv"],
                "notes": "Tabular data converted to readable format"
            }
        },
        "limits": {
            "max_file_size_mb": UPLOAD_SETTINGS.MAX_FILE_SIZE_MB,
            "memory_threshold_mb": UPLOAD_SETTINGS.DISK_THRESHOLD_MB,
            "supported_encodings": ["UTF-8"],
            "streaming_chunk_size_kb": UPLOAD_SETTINGS.STREAMING_CHUNK_SIZE_KB,
        }
    }