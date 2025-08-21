from fastapi import APIRouter, UploadFile, File
import logging
from src.chatup.graph.data_ingestion import run_ingestion

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...), namespace: str = "default"):
    try:
        file_bytes = await file.read()
        chunks = run_ingestion(file_bytes, file.filename, namespace)
        return {"status": "success", "num_chunks": len(chunks)}
    except Exception as e:
        logging.exception("Error during file upload:")
        return {"status": "error", "message": str(e)}