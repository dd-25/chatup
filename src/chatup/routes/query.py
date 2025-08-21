from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def query(query: str):
    # invoke Query Graph
    return {"response": "Response from the llm not implemented"}