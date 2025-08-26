from fastapi import APIRouter
from src.chatup.db.pinecone import check_pinecone_connection

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "ok"}

@router.get("/pinecone")
def pinecone_health_check():
    """
    Health check for Pinecone connection.
    """
    if check_pinecone_connection():
        return {"status": "ok"}
    else:
        return {"status": "error"}