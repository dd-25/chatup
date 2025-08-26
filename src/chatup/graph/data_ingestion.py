from __future__ import annotations
import logging
from typing import List, TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from src.chatup.services.data_ingestion import (
    extract_text_and_create_document,
    chunk_document,
    embed_and_upload
)

# Set up logging
logger = logging.getLogger(__name__)


class IngestionState(TypedDict):
    """State structure for the data ingestion graph."""
    # Input data
    file_bytes: bytes
    filename: str
    namespace: str
    
    # Processing results
    document: Optional[Any]  # LlamaIndex Document
    text_nodes: Optional[List[Any]]  # List of TextNode objects
    chunks: Optional[List[str]]  # Extracted text chunks
    
    # Metadata and status
    success: bool
    error_message: Optional[str]
    processing_stats: Dict[str, Any]


class DataIngestionGraph:
    """
    Robust LangGraph implementation for data ingestion pipeline.
    
    Pipeline stages:
    1. Document Extraction - Convert file bytes to LlamaIndex Document
    2. Text Chunking - Split document into optimal chunks
    3. Embedding & Upload - Generate embeddings and store in Pinecone
    4. Validation - Verify successful processing
    """
    
    def __init__(self):
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for data ingestion."""
        graph = StateGraph(IngestionState)
        
        # Add nodes
        graph.add_node("extract_document", self._extract_document_node)
        graph.add_node("chunk_text", self._chunk_text_node)
        graph.add_node("embed_upload", self._embed_upload_node)
        graph.add_node("validate_results", self._validate_results_node)
        graph.add_node("handle_error", self._handle_error_node)
        
        # Add edges - main flow
        graph.add_edge(START, "extract_document")
        graph.add_conditional_edges(
            "extract_document",
            self._should_continue_after_extraction,
            {
                "continue": "chunk_text",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "chunk_text",
            self._should_continue_after_chunking,
            {
                "continue": "embed_upload",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "embed_upload",
            self._should_continue_after_embedding,
            {
                "continue": "validate_results",
                "error": "handle_error"
            }
        )
        graph.add_edge("validate_results", END)
        graph.add_edge("handle_error", END)
        
        return graph
    
    def _extract_document_node(self, state: IngestionState) -> IngestionState:
        """
        Node 1: Extract text and create LlamaIndex Document.
        Handles multiple file formats (PDF, DOCX, JSON, TXT, etc.)
        """
        try:
            logger.info(f"Extracting document from {state['filename']}")
            
            document = extract_text_and_create_document(
                state["file_bytes"], 
                state["filename"]
            )
            
            # Update processing stats
            stats = {
                "original_size_bytes": len(state["file_bytes"]),
                "extracted_length_chars": len(document.text),
                "file_type": document.metadata.get("file_type", "unknown"),
                "extraction_success": True
            }
            
            return {
                **state,
                "document": document,
                "processing_stats": stats,
                "success": True,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Document extraction failed for {state['filename']}: {str(e)}")
            return {
                **state,
                "document": None,
                "success": False,
                "error_message": f"Document extraction failed: {str(e)}",
                "processing_stats": {"extraction_success": False}
            }
    
    def _chunk_text_node(self, state: IngestionState) -> IngestionState:
        """
        Node 2: Chunk the document into optimal pieces.
        Uses intelligent chunking based on document type.
        """
        try:
            logger.info(f"Chunking document {state['filename']}")
            
            if not state["document"]:
                raise ValueError("No document available for chunking")
            
            text_nodes = chunk_document(state["document"])
            chunks = [node.text for node in text_nodes]
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Update processing stats
            stats = state.get("processing_stats", {})
            stats.update({
                "chunk_count": len(chunks),
                "avg_chunk_length": sum(len(chunk) for chunk in chunks) / len(chunks),
                "chunking_success": True
            })
            
            return {
                **state,
                "text_nodes": text_nodes,
                "chunks": chunks,
                "processing_stats": stats,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Text chunking failed for {state['filename']}: {str(e)}")
            return {
                **state,
                "text_nodes": None,
                "chunks": None,
                "success": False,
                "error_message": f"Text chunking failed: {str(e)}"
            }
    
    def _embed_upload_node(self, state: IngestionState) -> IngestionState:
        """
        Node 3: Generate embeddings and upload to Pinecone.
        Handles batch processing for efficient uploads.
        """
        try:
            logger.info(f"Embedding and uploading {len(state['chunks'])} chunks from {state['filename']}")
            
            if not state["chunks"]:
                raise ValueError("No chunks available for embedding")
            
            embed_and_upload(
                chunks=state["chunks"],
                namespace=state["namespace"],
                source=state["filename"]
            )
            
            # Update processing stats
            stats = state.get("processing_stats", {})
            stats.update({
                "upload_success": True,
                "uploaded_chunk_count": len(state["chunks"])
            })
            
            return {
                **state,
                "processing_stats": stats,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Embedding/upload failed for {state['filename']}: {str(e)}")
            return {
                **state,
                "success": False,
                "error_message": f"Embedding/upload failed: {str(e)}"
            }
    
    def _validate_results_node(self, state: IngestionState) -> IngestionState:
        """
        Node 4: Validate the entire processing pipeline.
        Ensures data integrity and completeness.
        """
        try:
            logger.info(f"Validating processing results for {state['filename']}")
            
            # Validation checks
            validation_results = {
                "has_document": state["document"] is not None,
                "has_chunks": state["chunks"] is not None and len(state["chunks"]) > 0,
                "upload_completed": state.get("processing_stats", {}).get("upload_success", False)
            }
            
            all_valid = all(validation_results.values())
            
            # Update processing stats
            stats = state.get("processing_stats", {})
            stats.update({
                "validation_results": validation_results,
                "pipeline_success": all_valid,
                "final_status": "completed" if all_valid else "partial_failure"
            })
            
            return {
                **state,
                "processing_stats": stats,
                "success": all_valid
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {state['filename']}: {str(e)}")
            return {
                **state,
                "success": False,
                "error_message": f"Validation failed: {str(e)}"
            }
    
    def _handle_error_node(self, state: IngestionState) -> IngestionState:
        """
        Error handling node for graceful failure management.
        Logs errors and provides cleanup.
        """
        logger.error(f"Processing failed for {state['filename']}: {state.get('error_message', 'Unknown error')}")
        
        # Update processing stats with error information
        stats = state.get("processing_stats", {})
        stats.update({
            "pipeline_success": False,
            "final_status": "failed",
            "error_details": state.get("error_message", "Unknown error")
        })
        
        return {
            **state,
            "processing_stats": stats,
            "success": False
        }
    
    # Conditional edge functions
    def _should_continue_after_extraction(self, state: IngestionState) -> str:
        """Check if document extraction was successful."""
        return "continue" if state.get("success", False) and state.get("document") else "error"
    
    def _should_continue_after_chunking(self, state: IngestionState) -> str:
        """Check if text chunking was successful."""
        return "continue" if state.get("success", False) and state.get("chunks") else "error"
    
    def _should_continue_after_embedding(self, state: IngestionState) -> str:
        """Check if embedding and upload was successful."""
        return "continue" if state.get("success", False) else "error"
    
    def run_ingestion(
        self, 
        file_bytes: bytes, 
        filename: str, 
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Run the complete data ingestion pipeline.
        
        Args:
            file_bytes: Raw file content
            filename: Name of the file being processed
            namespace: Pinecone namespace for storage
            
        Returns:
            Dict containing processing results and statistics
        """
        initial_state: IngestionState = {
            "file_bytes": file_bytes,
            "filename": filename,
            "namespace": namespace,
            "document": None,
            "text_nodes": None,
            "chunks": None,
            "success": False,
            "error_message": None,
            "processing_stats": {}
        }
        
        try:
            logger.info(f"Starting ingestion pipeline for {filename}")
            result = self.compiled_graph.invoke(initial_state)
            
            return {
                "success": result.get("success", False),
                "chunks": result.get("chunks", []),
                "processing_stats": result.get("processing_stats", {}),
                "error_message": result.get("error_message"),
                "filename": filename,
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed for {filename}: {str(e)}")
            return {
                "success": False,
                "chunks": [],
                "processing_stats": {"pipeline_success": False, "error_details": str(e)},
                "error_message": f"Pipeline execution failed: {str(e)}",
                "filename": filename,
                "namespace": namespace
            }


# Create global instance
data_ingestion_graph = DataIngestionGraph()


def run_ingestion(file_bytes: bytes, filename: str, namespace: str = "default") -> Dict[str, Any]:
    """
    Convenience function to run data ingestion pipeline.
    
    Args:
        file_bytes: Raw file content
        filename: Name of the file being processed
        namespace: Pinecone namespace for storage
        
    Returns:
        Dict containing processing results and statistics
    """
    return data_ingestion_graph.run_ingestion(file_bytes, filename, namespace)
