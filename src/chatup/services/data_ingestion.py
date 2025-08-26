from __future__ import annotations
import io
import json
import uuid
import logging
from typing import List, Dict, Any
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader, DocxReader
from src.chatup.db.pinecone import upsert_embeddings
from src.chatup.config import settings
from src.chatup.utils.token import count_tokens
from src.chatup.utils.helper import split_sentences, split_json_chunk
from src.chatup.constants import EMBEDDING_SETTINGS, CHUNKING_SETTINGS, UPLOAD_SETTINGS

# Set up logging
logger = logging.getLogger(__name__)


# Initialize LlamaIndex components
embedding_model = OpenAIEmbedding(
    model=EMBEDDING_SETTINGS.EMBED_MODEL,
    api_key=settings.OPENAI_API_KEY
)

# Initialize node parser for chunking with better settings
node_parser = SentenceSplitter(
    chunk_size=CHUNKING_SETTINGS.MAX_TOKENS,
    chunk_overlap=CHUNKING_SETTINGS.OVERLAP_TOKENS,
    paragraph_separator="\n\n",
    secondary_chunking_regex="[.!?]+",
    include_metadata=True,
    include_prev_next_rel=False  # Don't include previous/next relationships to save space
)


def parse_json_content(json_data: Dict[str, Any], parent_key: str = "") -> List[str]:
    """
    Parse JSON content recursively and extract meaningful text chunks.
    Handles nested objects, arrays, and maintains context.
    """
    chunks = []
    
    def extract_from_value(value: Any, key_path: str = "") -> List[str]:
        extracted = []
        
        if isinstance(value, dict):
            # Handle nested objects
            for k, v in value.items():
                new_key_path = f"{key_path}.{k}" if key_path else k
                if isinstance(v, (str, int, float, bool)):
                    # Simple key-value pairs
                    extracted.append(f"{new_key_path}: {v}")
                else:
                    # Nested structures
                    extracted.extend(extract_from_value(v, new_key_path))
                    
        elif isinstance(value, list):
            # Handle arrays
            for i, item in enumerate(value):
                if isinstance(item, (str, int, float, bool)):
                    extracted.append(f"{key_path}[{i}]: {item}")
                else:
                    new_key_path = f"{key_path}[{i}]"
                    extracted.extend(extract_from_value(item, new_key_path))
                    
        else:
            # Simple values
            if key_path:
                extracted.append(f"{key_path}: {value}")
            else:
                extracted.append(str(value))
                
        return extracted
    
    # Extract all key-value pairs
    all_extracts = extract_from_value(json_data, parent_key)
    
    # Group related items into chunks
    current_chunk = ""
    for extract in all_extracts:
        # Check if adding this item would exceed token limit
        test_chunk = current_chunk + "\n" + extract if current_chunk else extract
        
        if count_tokens(test_chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
            current_chunk = test_chunk
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = extract
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_text_and_create_document(file_bytes: bytes, filename: str) -> Document:
    """
    Extract text from different file types using LlamaIndex readers.
    Supports: .txt, .pdf, .docx, .json and more formats.
    Includes fallback methods for problematic files.
    """
    extension = filename.split('.')[-1].lower()
    
    try:
        if extension == "pdf":
            # Create a temporary file-like object for PDF
            file_obj = io.BytesIO(file_bytes)
            file_obj.name = filename  # Add filename to file object for better error reporting
            
            try:
                pdf_reader = PDFReader()
                documents = pdf_reader.load_data(file_obj)
                if not documents:
                    raise ValueError(f"No content extracted from PDF: {filename}")
                # Combine all pages into one document
                combined_text = "\n\n".join([doc.text for doc in documents if doc.text.strip()])
                
                # Validate we got meaningful content
                if not combined_text.strip():
                    raise ValueError(f"PDF appears to be empty or contains only images: {filename}")
                    
            except Exception as pdf_error:
                # Enhanced error handling for PDF processing
                error_msg = str(pdf_error)
                logger.warning(f"LlamaIndex PDF reader failed for {filename}: {error_msg}")
                
                # Try fallback method using PyPDF2
                try:
                    import PyPDF2
                    file_obj.seek(0)  # Reset file pointer
                    pdf_reader_fallback = PyPDF2.PdfReader(file_obj)
                    
                    text_parts = []
                    for page in pdf_reader_fallback.pages:
                        text_parts.append(page.extract_text())
                    
                    combined_text = "\n\n".join(text_parts)
                    
                    if not combined_text.strip():
                        raise ValueError(f"PDF contains no extractable text: {filename}")
                        
                    logger.info(f"Successfully extracted PDF using PyPDF2 fallback: {filename}")
                    
                except ImportError:
                    logger.error("PyPDF2 not available for fallback. Install with: pip install PyPDF2")
                    raise ValueError(f"PDF processing failed and no fallback available: {filename}") from pdf_error
                except Exception as fallback_error:
                    logger.error(f"Both LlamaIndex and PyPDF2 failed for {filename}")
                    if "RetryError" in error_msg or "TypeError" in error_msg:
                        raise ValueError(f"PDF processing failed - file may be corrupted, password-protected, or contain only images: {filename}") from pdf_error
                    else:
                        raise ValueError(f"Error reading PDF {filename}: {error_msg}") from pdf_error
            
        elif extension == "docx":
            # Create a temporary file-like object for DOCX
            file_obj = io.BytesIO(file_bytes)
            file_obj.name = filename  # Add filename for better error reporting
            
            try:
                docx_reader = DocxReader()
                documents = docx_reader.load_data(file_obj)
                if not documents:
                    raise ValueError(f"No content extracted from DOCX: {filename}")
                combined_text = "\n\n".join([doc.text for doc in documents if doc.text.strip()])
                
                # Validate we got meaningful content
                if not combined_text.strip():
                    raise ValueError(f"DOCX appears to be empty: {filename}")
                    
            except Exception as docx_error:
                raise ValueError(f"Error reading DOCX {filename}: {str(docx_error)}") from docx_error
            
        elif extension == "json":
            # Parse JSON content
            try:
                json_str = file_bytes.decode("utf-8")
                json_data = json.loads(json_str)
                
                # Extract meaningful chunks from JSON
                json_chunks = parse_json_content(json_data)
                if not json_chunks:
                    raise ValueError(f"No meaningful content extracted from JSON: {filename}")
                combined_text = "\n\n".join(json_chunks)
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {filename}: {str(e)}")
            
        elif extension == "txt":
            combined_text = file_bytes.decode("utf-8")
            
        elif extension in ["csv", "tsv"]:
            # Basic CSV/TSV support
            try:
                text_content = file_bytes.decode("utf-8")
                lines = text_content.split('\n')
                # Convert CSV to readable format
                formatted_lines = []
                for i, line in enumerate(lines[:100]):  # Limit to first 100 rows for safety
                    if line.strip():
                        if extension == "csv":
                            fields = line.split(',')
                        else:  # tsv
                            fields = line.split('\t')
                        if i == 0:
                            formatted_lines.append(f"Headers: {', '.join(fields)}")
                        else:
                            formatted_lines.append(f"Row {i}: {', '.join(fields)}")
                combined_text = "\n".join(formatted_lines)
            except UnicodeDecodeError:
                raise ValueError(f"Unable to decode CSV/TSV file: {filename}")
            
        else:
            # For other formats, try to decode as text
            try:
                combined_text = file_bytes.decode("utf-8")
                # Try to parse as JSON if it looks like JSON
                if combined_text.strip().startswith(('{', '[')):
                    try:
                        json_data = json.loads(combined_text)
                        json_chunks = parse_json_content(json_data)
                        combined_text = "\n\n".join(json_chunks)
                    except json.JSONDecodeError:
                        # Not valid JSON, treat as regular text
                        pass
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type: {extension}")
        
        # Validate that we extracted some content
        if not combined_text or not combined_text.strip():
            raise ValueError(f"No text content extracted from file: {filename}")
        
        # Create LlamaIndex Document
        document = Document(
            text=combined_text,
            metadata={
                "filename": filename, 
                "file_type": extension,
                "original_size": len(file_bytes),
                "extracted_length": len(combined_text)
            }
        )
        
        logger.info(f"Successfully processed {filename}: extracted {len(combined_text)} characters")
        return document
        
    except Exception as e:
        logger.error(f"Document extraction failed for {filename}: {str(e)}")
        raise ValueError(f"Error processing file {filename}: {str(e)}") from e


def chunk_document(document: Document) -> List[TextNode]:
    """
    Chunk the document into smaller pieces using LlamaIndex's SentenceSplitter.
    Ensures chunks are properly sized for Pinecone metadata limits.
    Handles different document types intelligently.
    Returns a list of TextNode objects.
    """
    # Check if document is JSON-based (already pre-chunked)
    file_type = document.metadata.get("file_type", "")
    
    if file_type == "json":
        # For JSON, the text is already intelligently pre-chunked
        # Split by double newlines to get individual chunks
        json_chunks = document.text.split("\n\n")
        valid_nodes = []
        
        for chunk_text in json_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
                
            # Validate token count
            if count_tokens(chunk_text) <= CHUNKING_SETTINGS.MAX_TOKENS:
                chunk_node = TextNode(
                    text=chunk_text,
                    metadata=document.metadata.copy()
                )
                valid_nodes.append(chunk_node)
            else:
                # If a JSON chunk is still too large, split it using JSON-aware splitting
                json_sub_chunks = split_json_chunk(chunk_text)
                
                for sub_chunk in json_sub_chunks:
                    if sub_chunk.strip():
                        chunk_node = TextNode(
                            text=sub_chunk.strip(),
                            metadata=document.metadata.copy()
                        )
                        valid_nodes.append(chunk_node)
        
        return valid_nodes
    
    else:
        # For all other document types (PDF, DOCX, TXT), use standard chunking
        nodes = node_parser.get_nodes_from_documents([document])
        
        # Filter out empty nodes and ensure proper sizing
        valid_nodes = []
        for node in nodes:
            # Skip empty or very small chunks
            if not node.text.strip() or len(node.text.strip()) < 10:
                continue
            
            # Ensure chunk is within token limit
            if count_tokens(node.text) > CHUNKING_SETTINGS.MAX_TOKENS:
                # If still too large, split further by sentences
                sentences = split_sentences(node.text)
                current_chunk = ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    if count_tokens(test_chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
                        current_chunk = test_chunk
                    else:
                        if current_chunk.strip():
                            # Create a new TextNode with proper metadata
                            chunk_node = TextNode(
                                text=current_chunk.strip(),
                                metadata=node.metadata.copy() if node.metadata else {}
                            )
                            valid_nodes.append(chunk_node)
                        current_chunk = sentence
                
                # Add the last chunk if it has content
                if current_chunk.strip():
                    chunk_node = TextNode(
                        text=current_chunk.strip(),
                        metadata=node.metadata.copy() if node.metadata else {}
                    )
                    valid_nodes.append(chunk_node)
            else:
                valid_nodes.append(node)
        
        return valid_nodes


def embed_and_upload(
    chunks: List[str],
    namespace: str = "default",
    source: str = "unknown",
    id_prefix: str = "chunk",
):
    """
    Create embeddings for chunks and upload to Pinecone in batches.
    Assumes chunks are already properly sized.
    """
    if not chunks:
        return
    
    all_vectors = []
    
    # First, create all vectors
    for chunk in chunks:
        # Validate chunk size (should already be properly sized from chunking)
        if count_tokens(chunk) > CHUNKING_SETTINGS.MAX_TOKENS:
            logger.error(f"Chunk too large: {count_tokens(chunk)} tokens, {len(chunk)} chars")
            raise ValueError(f"Chunk too large ({count_tokens(chunk)} tokens). Fix chunking logic.")
        
        # Check chunk size in bytes to prevent Pinecone metadata limit issues
        chunk_bytes = len(chunk.encode('utf-8'))
        if chunk_bytes > UPLOAD_SETTINGS.MAX_CHUNK_SIZE_BYTES:  # 3MB safety limit (Pinecone limit is 4MB total)
            logger.error(f"Chunk too large in bytes: {chunk_bytes} bytes")
            raise ValueError(f"Chunk too large ({chunk_bytes} bytes). Reduce chunk size.")
        
        logger.info(f"Processing chunk: {count_tokens(chunk)} tokens, {chunk_bytes} bytes")
        
        vec = embedding_model.get_text_embedding(chunk)
        all_vectors.append(
            {
                "id": f"{id_prefix}-{str(uuid.uuid4())[:8]}",
                "values": vec,
                "metadata": {
                    "text": chunk,
                    "source": source,
                    "namespace": namespace,
                    "chunk_length": count_tokens(chunk),
                    "chunk_bytes": chunk_bytes,
                },
            }
        )
    
    # Now upload vectors in batches
    logger.info(f"Uploading {len(all_vectors)} vectors in batches...")
    
    current_batch = []
    current_batch_size = 0
    
    for i, vector in enumerate(all_vectors):
        # Estimate vector size: embeddings (1536 floats * 4 bytes) + metadata
        vector_size = len(vector["values"]) * 4 + vector["metadata"]["chunk_bytes"] + 200
        
        # Check if adding this vector would exceed limits
        if (current_batch_size + vector_size > UPLOAD_SETTINGS.MAX_VECTORS_BATCH_SIZE_BYTES or 
            len(current_batch) >= UPLOAD_SETTINGS.MAX_VECTORS_PER_BATCH):
            
            # Upload current batch
            if current_batch:
                logger.info(f"Uploading batch of {len(current_batch)} vectors ({current_batch_size:,} bytes)")
                try:
                    upsert_embeddings(vectors=current_batch, namespace=namespace)
                    logger.info(f"Successfully uploaded batch of {len(current_batch)} vectors")
                except Exception as e:
                    logger.error(f"Failed to upload batch: {str(e)}")
                    raise
                
                # Reset batch
                current_batch = []
                current_batch_size = 0
        
        # Add vector to current batch
        current_batch.append(vector)
        current_batch_size += vector_size
    
    # Upload final batch
    if current_batch:
        logger.info(f"Uploading final batch of {len(current_batch)} vectors ({current_batch_size:,} bytes)")
        try:
            upsert_embeddings(vectors=current_batch, namespace=namespace)
            logger.info(f"Successfully uploaded final batch of {len(current_batch)} vectors")
        except Exception as e:
            logger.error(f"Failed to upload final batch: {str(e)}")
            raise
    
    logger.info(f"Completed uploading all {len(all_vectors)} vectors for {source}")


def ingest_file(
    file_bytes: bytes,
    filename: str,
    namespace: str = "default",
) -> List[str]:
    """
    Main ingestion function using LlamaIndex.
    Extracts text, chunks it, and uploads to Pinecone.
    """
    try:
        logger.info(f"Starting ingestion for file: {filename} ({len(file_bytes)} bytes)")
        
        # Extract text and create document
        document = extract_text_and_create_document(file_bytes, filename)
        logger.info(f"Document extracted: {len(document.text)} characters")
        
        # Chunk the document
        text_nodes = chunk_document(document)
        logger.info(f"Document chunked into {len(text_nodes)} nodes")
        
        # Log chunk sizes for debugging
        for i, node in enumerate(text_nodes[:5]):  # Log first 5 chunks
            chunk_tokens = count_tokens(node.text)
            chunk_bytes = len(node.text.encode('utf-8'))
            logger.info(f"Chunk {i+1}: {chunk_tokens} tokens, {chunk_bytes} bytes")

        # TODO: Data Classification Strategy
        # Consider implementing a data classifier agent to categorize content.
        # Current approach: Use namespace-based classification where different
        # data types are stored in separate namespaces for better organization
        # and retrieval filtering. Future enhancement could include automatic
        # content categorization before chunking or per-chunk classification.

        # Convert TextNodes to strings for embedding
        chunks = [node.text for node in text_nodes]
        
        # Final validation before embedding
        total_chunks = len(chunks)
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            chunk_bytes = len(chunk.encode('utf-8'))
            
            if chunk_tokens > CHUNKING_SETTINGS.MAX_TOKENS:
                logger.error(f"Chunk {i+1}/{total_chunks} too large: {chunk_tokens} tokens")
                continue
                
            if chunk_bytes > 3_000_000:  # 3MB safety limit
                logger.error(f"Chunk {i+1}/{total_chunks} too large: {chunk_bytes} bytes")
                continue
                
            valid_chunks.append(chunk)
        
        logger.info(f"Valid chunks for upload: {len(valid_chunks)}/{total_chunks}")
        
        # Embed and upload to Pinecone
        embed_and_upload(valid_chunks, namespace=namespace, source=filename)
        
        logger.info(f"Successfully ingested {filename}: {len(valid_chunks)} chunks uploaded")
        return valid_chunks
        
    except Exception as e:
        logger.error(f"Processing failed for {filename}: {str(e)}")
        raise
