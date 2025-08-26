from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cohere
from llama_index.embeddings.openai import OpenAIEmbedding
from src.chatup.db.pinecone import query_embeddings
from src.chatup.config import settings
from src.chatup.constants import RETRIEVER_SETTINGS, PINECONE_SETTINGS
from src.chatup.utils.token import count_tokens
from src.chatup.utils.embeddings import embed_texts

logger = logging.getLogger(__name__)

# Initialize Cohere client if API key is provided
cohere_client = None
if cohere and settings.COHERE_API_KEY:
    try:
        cohere_client = cohere.Client(settings.COHERE_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize Cohere client: {e}")
        cohere_client = None

class RetrievalStrategy(Enum):
    """Different retrieval strategies for various use cases."""
    SEMANTIC = "semantic"          # Pure semantic similarity
    HYBRID = "hybrid"             # Semantic + keyword matching
    MULTI_QUERY = "multi_query"   # Multiple query variations
    CONTEXTUAL = "contextual"     # Context-aware retrieval
    RERANK = "rerank"             # Retrieval with re-ranking


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata and scores."""
    text: str
    source: str
    score: float
    chunk_id: str
    file_type: str
    chunk_length: int
    namespace: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "file_type": self.file_type,
            "chunk_length": self.chunk_length,
            "namespace": self.namespace,
            "metadata": self.metadata
        }


@dataclass
class RetrievalResult:
    """Complete retrieval result with chunks and metadata."""
    chunks: List[RetrievedChunk]
    query: str
    strategy: RetrievalStrategy
    total_found: int
    retrieval_time: float
    context_length: int
    
    def get_context_text(self, max_tokens: Optional[int] = None) -> str:
        """Get concatenated context text from all chunks."""
        context_parts = []
        current_tokens = 0
        
        for chunk in self.chunks:
            chunk_tokens = chunk.chunk_length
            if max_tokens and current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(f"Source: {chunk.source}\n{chunk.text}")
            current_tokens += chunk_tokens
            
        return "\n\n---\n\n".join(context_parts)
    
    def get_sources(self) -> List[str]:
        """Get unique sources from retrieved chunks."""
        return list(set(chunk.source for chunk in self.chunks))


class DocumentRetriever:
    """
    Robust document retriever with multiple strategies and optimizations.
    Handles various retrieval scenarios for RAG applications.
    """
    
    def __init__(
        self,
        default_top_k: int = PINECONE_SETTINGS.TOP_K,
        default_namespace: str = "default",
        similarity_threshold: float = RETRIEVER_SETTINGS.SIMILARITY_THRESHOLD,
        max_context_tokens: int = RETRIEVER_SETTINGS.MAX_TOKENS
    ):
        self.default_top_k = default_top_k
        self.default_namespace = default_namespace
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens
        
    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Main retrieval method that routes to specific strategies.
        
        Args:
            query: User query string
            strategy: Retrieval strategy to use
            top_k: Number of chunks to retrieve
            namespace: Pinecone namespace to search
            filters: Additional filters for retrieval
            **kwargs: Strategy-specific parameters
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.default_top_k
        namespace = namespace or self.default_namespace
        
        try:
            if strategy == RetrievalStrategy.SEMANTIC:
                chunks = self._semantic_retrieval(query, top_k, namespace, filters)
            elif strategy == RetrievalStrategy.HYBRID:
                chunks = self._hybrid_retrieval(query, top_k, namespace, filters, **kwargs)
            elif strategy == RetrievalStrategy.MULTI_QUERY:
                chunks = self._multi_query_retrieval(query, top_k, namespace, filters, **kwargs)
            elif strategy == RetrievalStrategy.CONTEXTUAL:
                chunks = self._contextual_retrieval(query, top_k, namespace, filters, **kwargs)
            elif strategy == RetrievalStrategy.RERANK:
                chunks = self._rerank_retrieval(query, top_k, namespace, filters, **kwargs)
            else:
                raise ValueError(f"Unknown retrieval strategy: {strategy}")
            
            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in chunks 
                if chunk.score >= self.similarity_threshold
            ]
            
            retrieval_time = time.time() - start_time
            context_length = sum(chunk.chunk_length for chunk in filtered_chunks)
            
            return RetrievalResult(
                chunks=filtered_chunks,
                query=query,
                strategy=strategy,
                total_found=len(filtered_chunks),
                retrieval_time=retrieval_time,
                context_length=context_length
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {repr(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise the exception instead of returning empty result
            raise e
    
    def _semantic_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedChunk]:
        """Basic semantic similarity retrieval."""
        # Check index stats before querying
        from src.chatup.db.pinecone import check_index_stats
        logger.info(f"Checking index stats for namespace '{namespace}'...")
        has_vectors = check_index_stats(namespace)
        
        if not has_vectors:
            logger.warning(f"No vectors found in namespace '{namespace}' - returning empty results")
            return []
        
        # Create query embedding - now handles single string correctly
        logger.info(f"Creating embedding for query: '{query[:50]}...'")
        query_vector = embed_texts(query)
        logger.info(f"Query embedding created with dimension: {len(query_vector)}")
        
        # Query Pinecone
        logger.info(f"Querying Pinecone with top_k={top_k}, namespace='{namespace}'")
        results = query_embeddings(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace
        )
        
        logger.info(f"Pinecone returned {len(results.matches) if hasattr(results, 'matches') else 0} matches")
        return self._process_pinecone_results(results)
    
    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        keyword_boost: float = RETRIEVER_SETTINGS.KEYWORD_BOOST,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Hybrid retrieval combining semantic and keyword matching."""
        # Get semantic results
        semantic_chunks = self._semantic_retrieval(query, top_k * 2, namespace, filters)
        
        # Extract keywords from query
        query_keywords = set(query.lower().split())
        
        # Boost scores for keyword matches
        for chunk in semantic_chunks:
            text_words = set(chunk.text.lower().split())
            keyword_overlap = len(query_keywords.intersection(text_words))
            keyword_ratio = keyword_overlap / len(query_keywords) if query_keywords else 0
            
            # Apply keyword boost
            chunk.score += keyword_ratio * keyword_boost
        
        # Re-sort by updated scores and return top_k
        semantic_chunks.sort(key=lambda x: x.score, reverse=True)
        return semantic_chunks[:top_k]
    
    def _multi_query_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        query_variations: Optional[List[str]] = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Multi-query retrieval using query variations."""
        if query_variations is None:
            query_variations = self._generate_query_variations(query)
        
        all_chunks = []
        seen_chunk_ids = set()
        
        # Retrieve for each query variation
        for variation in [query] + query_variations:
            chunks = self._semantic_retrieval(variation, top_k // 2, namespace, filters)
            
            for chunk in chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
        
        # Sort by score and return top_k
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        return all_chunks[:top_k]
    
    def _contextual_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        conversation_history: Optional[List[str]] = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Context-aware retrieval considering conversation history."""
        # If we have conversation history, create a contextual query
        if conversation_history:
            # Combine recent history with current query
            context = " ".join(conversation_history[-3:])  # Last 3 messages
            contextual_query = f"Context: {context}\nQuestion: {query}"
        else:
            contextual_query = query
        
        return self._semantic_retrieval(contextual_query, top_k, namespace, filters)
    
    def _rerank_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        rerank_factor: int = RETRIEVER_SETTINGS.RERANK_FACTOR,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieval with Cohere reranking for improved relevance.
        Falls back to basic semantic retrieval if Cohere is not available.
        """
        # Retrieve more candidates for reranking
        initial_top_k = min(top_k * rerank_factor, RETRIEVER_SETTINGS.COHERE_TOP_N)
        candidates = (self.retrieve(query, strategy, initial_top_k, namespace, filters)).chunks
        
        if not candidates:
            return []
        
        # If Cohere is not available, return top candidates by original score
        if not cohere_client:
            logger.warning("Cohere reranker not available, falling back to semantic scores")
            return candidates[:top_k]
        
        try:
            # Prepare documents for Cohere reranking
            documents = [chunk.text for chunk in candidates]
            
            # Use Cohere reranker
            rerank_response = cohere_client.rerank(
                model=RETRIEVER_SETTINGS.COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=False  # We already have the documents
            )
            
            # Map reranked results back to our chunks
            reranked_chunks = []
            for result in rerank_response.results:
                original_chunk = candidates[result.index]
                # Update the score with Cohere's relevance score
                original_chunk.score = result.relevance_score
                reranked_chunks.append(original_chunk)
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Fall back to original candidates
            return candidates[:top_k]
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for multi-query retrieval."""
        variations = []
        
        # Question variations
        if not query.endswith('?'):
            variations.append(f"{query}?")
        
        # Add context words
        context_words = ["explain", "describe", "what is", "how to", "why"]
        for word in context_words:
            if word not in query.lower():
                variations.append(f"{word} {query}")
        
        # Synonyms and paraphrases (simple approach)
        synonym_map = {
            "how": "what way",
            "what": "which",
            "why": "reason for",
            "when": "time of"
        }
        
        for original, synonym in synonym_map.items():
            if original in query.lower():
                variations.append(query.lower().replace(original, synonym))
        
        return variations[:3]  # Limit to 3 variations
    
    def _process_pinecone_results(self, results) -> List[RetrievedChunk]:
        """Convert Pinecone results to RetrievedChunk objects."""
        chunks = []
        
        if not results:
            logger.warning("No results returned from Pinecone")
            return chunks
            
        if not hasattr(results, 'matches'):
            logger.warning("Results object has no 'matches' attribute")
            logger.info(f"Results object type: {type(results)}")
            logger.info(f"Results object: {results}")
            return chunks
        
        logger.info(f"Processing {len(results.matches)} Pinecone matches")
        
        for i, match in enumerate(results.matches):
            metadata = match.metadata or {}
            logger.info(f"Match {i+1}: ID={match.id}, Score={match.score:.4f}, Source={metadata.get('source', 'unknown')}")
            
            chunk = RetrievedChunk(
                text=metadata.get("text", ""),
                source=metadata.get("source", "unknown"),
                score=float(match.score),
                chunk_id=match.id,
                file_type=metadata.get("file_type", "unknown"),
                chunk_length=metadata.get("chunk_length", 0),
                namespace=metadata.get("namespace", "default"),
                metadata=metadata
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} RetrievedChunk objects")
        return chunks
    
    def search_by_source(
        self,
        source_filename: str,
        query: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> RetrievalResult:
        """Search within a specific source document."""
        filters = {"source": source_filename}
        return self.retrieve(
            query=query,
            strategy=RetrievalStrategy.SEMANTIC,
            top_k=top_k,
            namespace=namespace,
            filters=filters
        )
    
    def search_by_file_type(
        self,
        file_type: str,
        query: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> RetrievalResult:
        """Search within documents of a specific file type."""
        filters = {"file_type": file_type}
        return self.retrieve(
            query=query,
            strategy=RetrievalStrategy.SEMANTIC,
            top_k=top_k,
            namespace=namespace,
            filters=filters
        )
    
    def get_similar_chunks(
        self,
        reference_text: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> RetrievalResult:
        """Find chunks similar to a reference text."""
        return self.retrieve(
            query=reference_text,
            strategy=RetrievalStrategy.SEMANTIC,
            top_k=top_k,
            namespace=namespace
        )
    
    def conversation_retrieval(
        self,
        query: str,
        conversation_history: List[str],
        top_k: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> RetrievalResult:
        """Retrieval optimized for conversational context."""
        return self.retrieve(
            query=query,
            strategy=RetrievalStrategy.CONTEXTUAL,
            top_k=top_k,
            namespace=namespace,
            conversation_history=conversation_history
        )


# Global retriever instance
retriever = DocumentRetriever()


# Convenience functions for common use cases
def semantic_search(
    query: str,
    top_k: int = PINECONE_SETTINGS.TOP_K,
    namespace: str = "default"
) -> RetrievalResult:
    """Simple semantic search function."""
    return retriever.retrieve(query, RetrievalStrategy.SEMANTIC, top_k, namespace)


def hybrid_search(
    query: str,
    top_k: int = PINECONE_SETTINGS.TOP_K,
    namespace: str = "default",
    keyword_boost: float = 0.1
) -> RetrievalResult:
    """Hybrid search with keyword boosting."""
    return retriever.retrieve(
        query, RetrievalStrategy.HYBRID, top_k, namespace,
        keyword_boost=keyword_boost
    )


def contextual_search(
    query: str,
    conversation_history: List[str],
    top_k: int = PINECONE_SETTINGS.TOP_K,
    namespace: str = "default"
) -> RetrievalResult:
    """Context-aware search for conversations."""
    return retriever.conversation_retrieval(query, conversation_history, top_k, namespace)


def search_document(
    query: str,
    source_filename: str,
    top_k: int = PINECONE_SETTINGS.TOP_K,
    namespace: str = "default"
) -> RetrievalResult:
    """Search within a specific document."""
    return retriever.search_by_source(source_filename, query, top_k, namespace)


def rerank_search(
    query: str,
    strategy: str = "diversity",
    top_k: int = PINECONE_SETTINGS.TOP_K,
    namespace: str = "default",
    rerank_factor: int = RETRIEVER_SETTINGS.RERANK_FACTOR
) -> RetrievalResult:
    """
    Advanced search with intelligent re-ranking.
    
    Strategies:
    - 'diversity': Maximize content diversity
    - 'length_balanced': Balance chunk lengths 
    - 'source_diversity': Prioritize diverse sources
    - 'query_coverage': Maximize query term coverage
    """
    return retriever.retrieve(
        query=query,
        strategy=RetrievalStrategy.RERANK,
        top_k=top_k,
        namespace=namespace,
        rerank_strategy=strategy,
        rerank_factor=rerank_factor
    )
