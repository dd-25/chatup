from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
from chatup.graph.retrieval import run_retrieval
from chatup.agents.general import ask_agent

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    strategy: Optional[str] = "semantic"
    top_k: Optional[int] = 5
    namespace: Optional[str] = "default"
    conversation_history: Optional[List[str]] = None
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool
    response: str
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None
    error_message: Optional[str] = None


@router.post("/", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Process user query through retrieval pipeline and generate response.
    
    Flow:
    1. Run retrieval graph to get relevant chunks
    2. Pass chunks to general agent for response generation
    3. Return structured response with sources and metadata
    """
    try:
        logger.info(f"Processing query: '{request.query[:50]}...'")
        
        # Step 1: Run retrieval graph
        retrieval_result = run_retrieval(
            query=request.query,
            strategy=request.strategy,
            top_k=request.top_k,
            namespace=request.namespace,
            conversation_history=request.conversation_history
        )
        
        # Check if retrieval was successful
        if not retrieval_result.get("success", False):
            error_msg = retrieval_result.get("error_message") or "Unknown retrieval error"
            logger.error(f"Retrieval failed: {error_msg}")
            return QueryResponse(
                success=False,
                response="I'm sorry, I couldn't retrieve relevant information to answer your question.",
                error_message=error_msg
            )
        
        # Extract chunks from retrieval result
        retrieval_data = retrieval_result.get("result")
        if not retrieval_data or not retrieval_data.chunks:
            logger.warning("No chunks found for query")
            return QueryResponse(
                success=True,
                response="I couldn't find any relevant information to answer your question. Could you try rephrasing or providing more context?",
                sources=[],
                metadata={"chunks_found": 0}
            )
        
        chunks = retrieval_data.chunks
        logger.info(f"Retrieved {len(chunks)} chunks for query")
        
        # Step 2: Generate response using general agent
        agent_result = ask_agent(
            query=request.query,
            chunks=chunks,
            conversation_history=request.conversation_history,
            stream=request.stream
        )
        
        # Handle streaming response
        if request.stream:
            # For streaming, we'd need to implement Server-Sent Events (SSE)
            # For now, return a message indicating streaming is not implemented in this endpoint
            return QueryResponse(
                success=False,
                response="Streaming is not implemented in this endpoint. Please use stream=false.",
                error_message="Streaming not supported"
            )
        
        # Check if agent response was successful
        if not agent_result.get("success", False):
            error_msg = agent_result.get("metadata", {}).get("error", "Agent response generation failed")
            logger.error(f"Agent response failed: {error_msg}")
            return QueryResponse(
                success=False,
                response="I encountered an error while generating a response to your question.",
                error_message=error_msg
            )
        
        # Extract response and metadata
        response_text = agent_result.get("response", "")
        agent_metadata = agent_result.get("metadata", {})
        
        # Extract sources from chunks
        sources = list(set(chunk.source for chunk in chunks))
        
        # Combine metadata from both retrieval and agent
        combined_metadata = {
            "retrieval": {
                "strategy": request.strategy,
                "chunks_retrieved": len(chunks),
                "agent_selected_strategy": retrieval_result.get("agent_selected_strategy"),
                "processing_time": retrieval_result.get("processing_time", 0)
            },
            "agent": {
                "model_used": agent_metadata.get("model_used"),
                "generation_time": agent_metadata.get("generation_time", 0),
                "total_tokens": agent_metadata.get("total_tokens", 0),
                "temperature": agent_metadata.get("temperature")
            },
            "total_processing_time": retrieval_result.get("processing_time", 0) + agent_metadata.get("generation_time", 0)
        }
        
        logger.info(f"Query processed successfully in {combined_metadata['total_processing_time']:.2f}s")
        
        return QueryResponse(
            success=True,
            response=response_text,
            sources=sources,
            metadata=combined_metadata
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return QueryResponse(
            success=False,
            response="I'm sorry, I encountered an unexpected error while processing your question.",
            error_message=str(e)
        )


@router.get("/health")
def health_check():
    """Health check endpoint for the query service."""
    return {"status": "healthy", "service": "query"}