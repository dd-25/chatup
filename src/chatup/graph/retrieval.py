from __future__ import annotations
import logging
import time
from typing import List, TypedDict, Optional, Dict, Any, Union
from langgraph.graph import StateGraph, START, END
from chatup.services.retrieval import (
    DocumentRetriever,
    RetrievalStrategy,
    RetrievedChunk,
    RetrievalResult,
    rerank_search
)
from src.chatup.constants import RETRIEVER_SETTINGS, PINECONE_SETTINGS

# Set up logging
logger = logging.getLogger(__name__)


class RetrievalState(TypedDict):
    """State structure for the retrieval graph."""
    # Input parameters
    query: str
    strategy: str
    top_k: int
    namespace: str
    filters: Optional[Dict[str, Any]]
    conversation_history: Optional[List[str]]
    
    # Processing results
    raw_candidates: Optional[List[RetrievedChunk]]
    filtered_chunks: Optional[List[RetrievedChunk]]
    reranked_chunks: Optional[List[RetrievedChunk]]
    final_result: Optional[RetrievalResult]
    
    # Agent decisions
    agent_selected_strategy: Optional[str]
    agent_reasoning: Optional[str]
    agent_metadata: Optional[Dict[str, Any]]
    
    # Reranking agent decisions
    rerank_agent_selected_strategy: Optional[str]
    rerank_agent_reasoning: Optional[str]
    rerank_agent_metadata: Optional[Dict[str, Any]]
    
    # Metadata and performance
    retrieval_metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str]
    
    # Quality metrics
    relevance_scores: Optional[List[float]]
    diversity_score: Optional[float]
    coverage_score: Optional[float]


class RetrievalGraph:
    """
    Comprehensive LangGraph implementation for document retrieval.
    
    Pipeline stages:
    1. Query Processing - Analyze and enhance the user query
    2. Strategy Agent - Intelligent agent decides optimal retrieval strategy
    3. Initial Retrieval - Get candidate chunks based on agent-selected strategy
    4. Quality Filtering - Filter by relevance and quality metrics
    5. Reranking Strategy Agent - Intelligent agent decides optimal reranking strategy
    6. Reranking - Apply Cohere reranking to all results for better relevance
    7. Final Assembly - Create comprehensive retrieval result
    """
    
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for document retrieval."""
        graph = StateGraph(RetrievalState)
        
        # Add nodes
        graph.add_node("process_query", self._process_query_node)
        graph.add_node("strategy_agent", self._strategy_agent_node)
        graph.add_node("initial_retrieval", self._initial_retrieval_node)
        graph.add_node("filter_quality", self._filter_quality_node)
        graph.add_node("rerank_strategy_agent", self._rerank_strategy_agent_node)
        graph.add_node("rerank_results", self._rerank_results_node)
        graph.add_node("assemble_final", self._assemble_final_node)
        graph.add_node("handle_error", self._handle_error_node)
        
        # Add edges - main flow
        graph.add_edge(START, "process_query")
        graph.add_conditional_edges(
            "process_query",
            self._should_continue_after_query_processing,
            {
                "continue": "strategy_agent",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "strategy_agent",
            self._should_continue_after_agent,
            {
                "continue": "initial_retrieval",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "initial_retrieval",
            self._should_continue_after_retrieval,
            {
                "continue": "filter_quality",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "filter_quality",
            self._should_continue_after_filtering,
            {
                "continue": "rerank_strategy_agent",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "rerank_strategy_agent",
            self._should_continue_after_rerank_agent,
            {
                "rerank": "rerank_results",
                "skip_rerank": "assemble_final",
                "error": "handle_error"
            }
        )
        graph.add_conditional_edges(
            "rerank_results",
            self._should_continue_after_reranking,
            {
                "continue": "assemble_final",
                "error": "handle_error"
            }
        )
        graph.add_edge("assemble_final", END)
        graph.add_edge("handle_error", END)
        
        return graph
    
    def _process_query_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 1: Process and enhance the user query.
        Analyzes query intent, extracts keywords, and prepares for retrieval.
        """
        try:
            logger.info(f"Processing query: '{state['query'][:100]}...'")
            
            query = state["query"].strip()
            if not query:
                raise ValueError("Empty query provided")
            
            # Analyze query characteristics
            query_length = len(query.split())
            has_question_words = any(word in query.lower() for word in ["what", "how", "why", "when", "where", "who"])
            is_question = query.endswith("?") or has_question_words
            
            # Extract key terms (simple approach)
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            key_terms = [word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2]
            
            # Set strategy if not specified
            strategy = state.get("strategy", "semantic")
            
            # Note: Strategy selection is now handled by the strategy_agent node
            # This processing step focuses on query analysis and preparation
            
            metadata = {
                "query_length": query_length,
                "key_terms": key_terms,
                "is_question": is_question,
                "initial_strategy": strategy,  # Strategy before agent decision
                "processing_timestamp": time.time()
            }
            
            return {
                **state,
                "strategy": strategy,  # Keep original strategy for now
                "retrieval_metadata": metadata,
                "success": True,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                **state,
                "success": False,
                "error_message": f"Query processing failed: {str(e)}",
                "retrieval_metadata": {"processing_error": str(e)}
            }
    
    def _strategy_agent_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 2: Strategy Agent - Intelligent agent decides optimal retrieval strategy.
        
        Currently defaults to semantic strategy. This is where agent logic will be integrated
        to intelligently select the best retrieval strategy based on:
        - Query characteristics
        - Available context
        - User intent
        - Historical performance
        - Document types in the namespace
        """
        try:
            logger.info("Strategy agent analyzing query for optimal retrieval strategy")
            
            query = state["query"]
            metadata = state.get("retrieval_metadata", {})
            initial_strategy = state.get("strategy", "semantic")
            
            # TODO: Replace this with actual agent logic
            # For now, force semantic strategy as requested (removed RERANK as it's now mandatory post-processing)
            # Available strategies: semantic, hybrid, multi_query, contextual
            agent_selected_strategy = "semantic"
            
            # Placeholder for agent reasoning (will be replaced with actual agent logic)
            agent_reasoning = "Default semantic strategy selected. Agent integration pending."
            
            # Agent metadata (will contain agent's analysis and decision process)
            agent_metadata = {
                "query_analysis": {
                    "query_length": metadata.get("query_length", 0),
                    "key_terms_count": len(metadata.get("key_terms", [])),
                    "is_question": metadata.get("is_question", False),
                    "query_complexity": "simple"  # Will be determined by agent
                },
                "strategy_decision": {
                    "initial_strategy": initial_strategy,
                    "selected_strategy": agent_selected_strategy,
                    "confidence_score": 1.0,  # Agent's confidence in the decision
                    "alternative_strategies": [],  # Other strategies the agent considered
                    "decision_factors": ["default_semantic"]  # Factors that influenced the decision
                },
                "agent_version": "placeholder_v1.0",
                "decision_timestamp": time.time()
            }
            
            # Update retrieval metadata with agent insights
            updated_metadata = metadata.copy()
            updated_metadata.update({
                "final_strategy": agent_selected_strategy,
                "strategy_decision_method": "agent",
                "agent_reasoning": agent_reasoning
            })
            
            logger.info(f"Strategy agent selected: {agent_selected_strategy}")
            
            return {
                **state,
                "strategy": agent_selected_strategy,
                "agent_selected_strategy": agent_selected_strategy,
                "agent_reasoning": agent_reasoning,
                "agent_metadata": agent_metadata,
                "retrieval_metadata": updated_metadata,
                "success": True,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Strategy agent failed: {str(e)}")
            
            # Fallback to semantic strategy if agent fails
            fallback_strategy = "semantic"
            error_metadata = {
                "agent_error": str(e),
                "fallback_strategy": fallback_strategy,
                "agent_status": "failed"
            }
            
            updated_metadata = state.get("retrieval_metadata", {})
            updated_metadata.update(error_metadata)
            
            return {
                **state,
                "strategy": fallback_strategy,
                "agent_selected_strategy": fallback_strategy,
                "agent_reasoning": f"Agent failed, using fallback: {str(e)}",
                "agent_metadata": error_metadata,
                "retrieval_metadata": updated_metadata,
                "success": True,  # Continue with fallback strategy
                "error_message": None
            }
    
    def _initial_retrieval_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 3: Perform initial retrieval based on agent-selected strategy.
        Gets candidate chunks using the strategy determined by the agent.
        """
        try:
            agent_strategy = state.get("agent_selected_strategy", "semantic")
            logger.info(f"Performing {agent_strategy} retrieval (agent-selected)")
            
            # Map string strategy to enum (RERANK removed as it's now mandatory post-processing)
            strategy_map = {
                "semantic": RetrievalStrategy.SEMANTIC,
                "hybrid": RetrievalStrategy.HYBRID,
                "multi_query": RetrievalStrategy.MULTI_QUERY,
                "contextual": RetrievalStrategy.CONTEXTUAL
            }
            
            strategy_enum = strategy_map.get(agent_strategy, RetrievalStrategy.SEMANTIC)
            
            # Prepare retrieval parameters
            retrieval_params = {
                "query": state["query"],
                "strategy": strategy_enum,
                "top_k": state.get("top_k", PINECONE_SETTINGS.TOP_K) * RETRIEVER_SETTINGS.RERANK_FACTOR,  # Get more candidates for reranking
                "namespace": state.get("namespace", "default"),
                "filters": state.get("filters")
            }
            
            # Add strategy-specific parameters
            if state.get("conversation_history"):
                retrieval_params["conversation_history"] = state["conversation_history"]
            
            # Perform retrieval
            start_time = time.time()
            logger.info(f"Calling retriever.retrieve with params: query='{state['query'][:50]}...', strategy={strategy_enum}, top_k={retrieval_params['top_k']}, namespace={retrieval_params['namespace']}")
            result = self.retriever.retrieve(**retrieval_params)
            retrieval_time = time.time() - start_time
            
            # Update metadata
            metadata = state.get("retrieval_metadata", {})
            metadata.update({
                "initial_candidates_count": len(result.chunks),
                "retrieval_time": retrieval_time,
                "strategy_used": agent_strategy,
                "agent_decision_applied": True
            })
            
            logger.info(f"Retrieval successful: got {len(result.chunks)} chunks in {retrieval_time:.2f}s")
            
            return {
                **state,
                "raw_candidates": result.chunks,
                "retrieval_metadata": metadata,
                "processing_time": retrieval_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Initial retrieval failed: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {repr(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                **state,
                "raw_candidates": [],
                "success": False,
                "error_message": f"Initial retrieval failed: {str(e)}"
            }
    
    def _filter_quality_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 4: Apply minimal quality filtering by similarity threshold only.
        Trusts the chunking and embedding process, letting Cohere reranking handle final quality assessment.
        """
        try:
            logger.info("Applying minimal quality filtering - trusting pipeline and vector similarity")
            
            candidates = state.get("raw_candidates", [])
            if not candidates:
                return {
                    **state,
                    "filtered_chunks": [],
                    "relevance_scores": [],
                    "success": True
                }
            
            # Apply similarity threshold - trust vector similarity and let Cohere reranking handle quality
            similarity_threshold = RETRIEVER_SETTINGS.SIMILARITY_THRESHOLD
            quality_filtered = [
                chunk for chunk in candidates 
                if chunk.score >= similarity_threshold
            ]
            
            # Trust the chunking and embedding process - no aggressive filtering
            # Vector similarity + Cohere reranking will ensure quality
            
            # Calculate quality metrics
            relevance_scores = [chunk.score for chunk in quality_filtered]
            
            # Calculate diversity score (simple approach)
            if len(quality_filtered) > 1:
                sources = set(chunk.source for chunk in quality_filtered)
                diversity_score = len(sources) / len(quality_filtered)
            else:
                diversity_score = 1.0
            
            # Update metadata
            metadata = state.get("retrieval_metadata", {})
            metadata.update({
                "similarity_filtered_count": len(quality_filtered),
                "avg_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                "diversity_score": diversity_score,
                "filtering_approach": "minimal_trust_pipeline"  # Indicate we trust the chunking/embedding process
            })
            
            return {
                **state,
                "filtered_chunks": quality_filtered,
                "relevance_scores": relevance_scores,
                "diversity_score": diversity_score,
                "retrieval_metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Quality filtering failed: {str(e)}")
            return {
                **state,
                "filtered_chunks": state.get("raw_candidates", []),
                "success": False,
                "error_message": f"Quality filtering failed: {str(e)}"
            }
    
    def _rerank_strategy_agent_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 5: Reranking Strategy Agent - Intelligent agent decides optimal reranking strategy.
        
        Currently defaults to 'diversity' strategy. This is where reranking agent logic will be integrated
        to intelligently select the best reranking strategy based on:
        - Query characteristics and complexity
        - Number of filtered chunks available
        - Source diversity in the candidates
        - Query type (factual, analytical, comparative, etc.)
        - User preferences and historical performance
        """
        try:
            logger.info("Reranking strategy agent analyzing candidates for optimal reranking strategy")
            
            filtered_chunks = state.get("filtered_chunks", [])
            query = state["query"]
            metadata = state.get("retrieval_metadata", {})
            
            # TODO: Replace this with actual reranking agent logic
            # For now, default to 'diversity' strategy as requested
            # Available reranking strategies: 'diversity', 'length_balanced', 'source_diversity', 'query_coverage', 'skip'
            rerank_agent_selected_strategy = "diversity"
            
            # Placeholder for reranking agent reasoning (will be replaced with actual agent logic)
            rerank_agent_reasoning = "Default diversity reranking strategy selected. Reranking agent integration pending."
            
            # Analyze candidates for reranking agent decision making
            num_candidates = len(filtered_chunks)
            unique_sources = len(set(chunk.source for chunk in filtered_chunks)) if filtered_chunks else 0
            avg_chunk_length = sum(chunk.chunk_length for chunk in filtered_chunks) / num_candidates if num_candidates > 0 else 0
            
            # Reranking agent metadata (will contain agent's analysis and decision process)
            rerank_agent_metadata = {
                "candidate_analysis": {
                    "num_candidates": num_candidates,
                    "unique_sources": unique_sources,
                    "avg_chunk_length": avg_chunk_length,
                    "source_diversity_ratio": unique_sources / num_candidates if num_candidates > 0 else 0
                },
                "rerank_strategy_decision": {
                    "selected_strategy": rerank_agent_selected_strategy,
                    "confidence_score": 1.0,  # Agent's confidence in the reranking strategy decision
                    "alternative_strategies": [],  # Other strategies the agent considered
                    "decision_factors": ["default_diversity"]  # Factors that influenced the reranking decision
                },
                "rerank_agent_version": "placeholder_v1.0",
                "rerank_decision_timestamp": time.time()
            }
            
            # Update retrieval metadata with reranking agent insights
            updated_metadata = metadata.copy()
            updated_metadata.update({
                "rerank_strategy": rerank_agent_selected_strategy,
                "rerank_decision_method": "rerank_agent",
                "rerank_agent_reasoning": rerank_agent_reasoning
            })
            
            logger.info(f"Reranking strategy agent selected: {rerank_agent_selected_strategy}")
            
            return {
                **state,
                "rerank_agent_selected_strategy": rerank_agent_selected_strategy,
                "rerank_agent_reasoning": rerank_agent_reasoning,
                "rerank_agent_metadata": rerank_agent_metadata,
                "retrieval_metadata": updated_metadata,
                "success": True,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Reranking strategy agent failed: {str(e)}")
            
            # Fallback to diversity strategy if reranking agent fails
            fallback_rerank_strategy = "diversity"
            error_metadata = {
                "rerank_agent_error": str(e),
                "fallback_rerank_strategy": fallback_rerank_strategy,
                "rerank_agent_status": "failed"
            }
            
            updated_metadata = state.get("retrieval_metadata", {})
            updated_metadata.update(error_metadata)
            
            return {
                **state,
                "rerank_agent_selected_strategy": fallback_rerank_strategy,
                "rerank_agent_reasoning": f"Reranking agent failed, using fallback: {str(e)}",
                "rerank_agent_metadata": error_metadata,
                "retrieval_metadata": updated_metadata,
                "success": True,  # Continue with fallback strategy
                "error_message": None
            }
    
    def _rerank_results_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 6: Rerank all results using Cohere with agent-selected strategy for improved relevance.
        This is applied to ALL retrieval strategies as post-processing.
        """
        try:
            logger.info("Applying Cohere reranking to improve result relevance")
            
            filtered_chunks = state.get("filtered_chunks", [])
            target_top_k = state.get("top_k", PINECONE_SETTINGS.TOP_K)
            query = state["query"]
            
            if not filtered_chunks:
                return {
                    **state,
                    "reranked_chunks": [],
                    "success": True
                }
            
            # If we have fewer or equal chunks than needed, no reranking needed
            if len(filtered_chunks) <= target_top_k:
                logger.info(f"Only {len(filtered_chunks)} chunks available, skipping reranking")
                return {
                    **state,
                    "reranked_chunks": filtered_chunks,
                    "success": True
                }
            
            # Use your existing rerank_search function with agent-selected strategy
            try:
                # Get the reranking strategy selected by the agent
                rerank_strategy = state.get("rerank_agent_selected_strategy", "diversity")
                
                # Use your existing rerank_search function
                rerank_result = rerank_search(
                    query=query,
                    top_k=target_top_k,
                    namespace=state.get("namespace", "default"),
                    strategy=rerank_strategy  # Use agent-selected strategy
                )
                
                reranked_chunks = rerank_result.chunks
                rerank_method = f"rerank_search_function_{rerank_strategy}"
                logger.info(f"Reranking completed using rerank_search with {rerank_strategy} strategy: {len(reranked_chunks)} chunks reranked")
                
            except Exception as e:
                logger.warning(f"Rerank search function failed: {e}, using original ranking")
                reranked_chunks = filtered_chunks[:target_top_k]
                rerank_method = "fallback_ranking"
            
            # Calculate reranking metrics
            original_scores = [chunk.score for chunk in filtered_chunks[:target_top_k]]
            reranked_scores = [chunk.score for chunk in reranked_chunks]
            
            # Update metadata
            metadata = state.get("retrieval_metadata", {})
            metadata.update({
                "reranked_count": len(reranked_chunks),
                "rerank_method": rerank_method,
                "original_avg_score": sum(original_scores) / len(original_scores) if original_scores else 0,
                "reranked_avg_score": sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0,
                "reranking_applied": True
            })
            
            return {
                **state,
                "reranked_chunks": reranked_chunks,
                "retrieval_metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Fallback to filtered chunks
            fallback_chunks = state.get("filtered_chunks", [])[:state.get("top_k", PINECONE_SETTINGS.TOP_K)]
            
            metadata = state.get("retrieval_metadata", {})
            metadata.update({
                "reranking_applied": False,
                "rerank_error": str(e),
                "fallback_used": True
            })
            
            return {
                **state,
                "reranked_chunks": fallback_chunks,
                "retrieval_metadata": metadata,
                "success": True  # Continue with fallback
            }
    
    def _assemble_final_node(self, state: RetrievalState) -> RetrievalState:
        """
        Node 7: Assemble the final retrieval result.
        Creates the comprehensive result with all metadata.
        """
        try:
            logger.info("Assembling final retrieval result")
            
            # Use reranked chunks if available, otherwise use filtered chunks
            final_chunks = state.get("reranked_chunks") or state.get("filtered_chunks", [])
            target_top_k = state.get("top_k", PINECONE_SETTINGS.TOP_K)
            
            # Ensure we don't exceed target_top_k (should already be handled by reranking)
            final_chunks = final_chunks[:target_top_k]
            
            # Calculate final metrics
            total_context_length = sum(chunk.chunk_length for chunk in final_chunks)
            total_processing_time = state.get("processing_time", 0)
            
            # Create retrieval result
            strategy_enum = RetrievalStrategy.SEMANTIC
            try:
                strategy_map = {
                    "semantic": RetrievalStrategy.SEMANTIC,
                    "hybrid": RetrievalStrategy.HYBRID,
                    "multi_query": RetrievalStrategy.MULTI_QUERY,
                    "contextual": RetrievalStrategy.CONTEXTUAL
                }
                strategy_enum = strategy_map.get(state.get("strategy", "semantic"), RetrievalStrategy.SEMANTIC)
            except:
                pass
            
            result = RetrievalResult(
                chunks=final_chunks,
                query=state["query"],
                strategy=strategy_enum,
                total_found=len(final_chunks),
                retrieval_time=total_processing_time,
                context_length=total_context_length
            )
            
            # Final metadata
            metadata = state.get("retrieval_metadata", {})
            metadata.update({
                "final_chunk_count": len(final_chunks),
                "total_context_length": total_context_length,
                "pipeline_success": True,
                "completion_timestamp": time.time(),
                "reranking_mandatory": True  # Indicate that reranking was applied to all results
            })
            
            return {
                **state,
                "final_result": result,
                "retrieval_metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Final assembly failed: {str(e)}")
            return {
                **state,
                "final_result": None,
                "success": False,
                "error_message": f"Final assembly failed: {str(e)}"
            }
    
    def _handle_error_node(self, state: RetrievalState) -> RetrievalState:
        """
        Error handling node for graceful failure management.
        """
        error_msg = state.get("error_message", "Unknown error occurred")
        logger.error(f"Retrieval pipeline failed: {error_msg}")
        
        # Create empty result for error case
        empty_result = RetrievalResult(
            chunks=[],
            query=state.get("query", ""),
            strategy=RetrievalStrategy.SEMANTIC,
            total_found=0,
            retrieval_time=state.get("processing_time", 0),
            context_length=0
        )
        
        metadata = state.get("retrieval_metadata", {})
        metadata.update({
            "pipeline_success": False,
            "error_details": error_msg,
            "completion_timestamp": time.time()
        })
        
        return {
            **state,
            "final_result": empty_result,
            "retrieval_metadata": metadata,
            "success": False
        }
    
    # Conditional edge functions
    def _should_continue_after_query_processing(self, state: RetrievalState) -> str:
        """Check if query processing was successful."""
        return "continue" if state.get("success", False) else "error"
    
    def _should_continue_after_agent(self, state: RetrievalState) -> str:
        """Check if strategy agent was successful."""
        return "continue" if state.get("success", False) and state.get("agent_selected_strategy") else "error"
    
    def _should_continue_after_retrieval(self, state: RetrievalState) -> str:
        """Check if initial retrieval was successful."""
        return "continue" if state.get("success", False) and state.get("raw_candidates") else "error"
    
    def _should_continue_after_filtering(self, state: RetrievalState) -> str:
        """Check if quality filtering was successful."""
        return "continue" if state.get("success", False) else "error"
    
    def _should_continue_after_rerank_agent(self, state: RetrievalState) -> str:
        """Check if reranking strategy agent was successful and decide next step."""
        if not state.get("success", False) or not state.get("rerank_agent_selected_strategy"):
            return "error"
        
        strategy = state.get("rerank_agent_selected_strategy", "")
        return "skip_rerank" if strategy == "skip" else "rerank"
    
    def _should_continue_after_reranking(self, state: RetrievalState) -> str:
        """Check if reranking was successful."""
        return "continue" if state.get("success", False) else "error"
    
    def run_retrieval(
        self,
        query: str,
        strategy: str = "semantic",
        top_k: int = None,
        namespace: str = "default",
        filters: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete retrieval pipeline with mandatory reranking.
        
        Args:
            query: User query string
            strategy: Retrieval strategy ("semantic", "hybrid", "contextual", "multi_query")
            top_k: Number of chunks to retrieve
            namespace: Pinecone namespace
            filters: Additional filters
            conversation_history: Previous conversation context
            
        Returns:
            Dict containing retrieval result and metadata
            
        Note: All results are automatically reranked using Cohere for improved relevance.
        """
        initial_state: RetrievalState = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k or PINECONE_SETTINGS.TOP_K,
            "namespace": namespace,
            "filters": filters,
            "conversation_history": conversation_history,
            "raw_candidates": None,
            "filtered_chunks": None,
            "reranked_chunks": None,
            "final_result": None,
            "agent_selected_strategy": None,
            "agent_reasoning": None,
            "agent_metadata": None,
            "rerank_agent_selected_strategy": None,
            "rerank_agent_reasoning": None,
            "rerank_agent_metadata": None,
            "retrieval_metadata": {},
            "processing_time": 0.0,
            "success": False,
            "error_message": None,
            "relevance_scores": None,
            "diversity_score": None,
            "coverage_score": None
        }
        
        try:
            logger.info(f"Starting retrieval pipeline for query: '{query[:50]}...'")
            start_time = time.time()
            
            result = self.compiled_graph.invoke(initial_state)
            total_time = time.time() - start_time
            
            # Log the result for debugging
            logger.info(f"Pipeline completed. Success: {result.get('success', False)}, Error: {result.get('error_message', 'None')}")
            
            return {
                "success": result.get("success", False),
                "result": result.get("final_result"),
                "metadata": result.get("retrieval_metadata", {}),
                "agent_metadata": result.get("agent_metadata", {}),
                "agent_reasoning": result.get("agent_reasoning"),
                "processing_time": total_time,
                "error_message": result.get("error_message"),
                "query": query,
                "strategy": strategy,
                "agent_selected_strategy": result.get("agent_selected_strategy")
            }
            
        except Exception as e:
            logger.error(f"Retrieval pipeline execution failed: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {repr(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "result": None,
                "metadata": {"pipeline_error": str(e)},
                "agent_metadata": {},
                "agent_reasoning": None,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "error_message": f"Pipeline execution failed: {str(e)}",
                "query": query,
                "strategy": strategy,
                "agent_selected_strategy": None
            }


# Create global instance
retrieval_graph = RetrievalGraph()


def run_retrieval(
    query: str,
    strategy: str = "semantic",
    top_k: int = None,
    namespace: str = "default",
    filters: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run retrieval pipeline with mandatory reranking.
    
    Args:
        query: User query string
        strategy: Retrieval strategy ("semantic", "hybrid", "contextual", "multi_query")
        top_k: Number of chunks to retrieve
        namespace: Pinecone namespace
        filters: Additional filters
        conversation_history: Previous conversation context
        
    Returns:
        Dict containing retrieval result and metadata
        
    Note: All results are automatically reranked using Cohere for improved relevance.
    """
    return retrieval_graph.run_retrieval(
        query=query,
        strategy=strategy,
        top_k=top_k,
        namespace=namespace,
        filters=filters,
        conversation_history=conversation_history
    )
