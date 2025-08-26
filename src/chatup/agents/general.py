from __future__ import annotations
import logging
import time
from typing import List, Optional, Dict, Any
from openai import OpenAI
from chatup.services.retrieval import RetrievedChunk
from chatup.config import settings

# Set up logging
logger = logging.getLogger(__name__)


class GeneralAgent:
    """
    General purpose agent that takes retrieved chunks and generates responses.
    
    This agent:
    1. Takes retrieved chunks as context
    2. Formats them into a comprehensive prompt
    3. Uses OpenAI to generate intelligent responses
    4. Handles conversation history and context
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"  # Cost-effective model for general responses
        self.max_tokens = 1000
        self.temperature = 0.7
    
    def generate_response(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        conversation_history: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved chunks.
        
        Args:
            query: User's question/query
            chunks: Retrieved chunks providing context
            conversation_history: Previous conversation messages
            system_prompt: Custom system prompt (optional)
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            Dict containing response and metadata
        """
        try:
            logger.info(f"Generating response for query: '{query[:50]}...'")
            start_time = time.time()
            
            # Prepare the context from chunks
            context = self._format_chunks_as_context(chunks)
            
            # Build the prompt
            messages = self._build_prompt(
                query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            # Set up API parameters
            api_params = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": False
            }
            
            # Generate response
            response = self.client.chat.completions.create(**api_params)
            
            generation_time = time.time() - start_time
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Prepare metadata
            metadata = {
                "model_used": api_params["model"],
                "generation_time": generation_time,
                "chunks_used": len(chunks),
                "context_length": len(context),
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "temperature": api_params["temperature"],
                "sources": list(set(chunk.source for chunk in chunks)),
                "timestamp": time.time()
            }
            
            logger.info(f"Response generated successfully in {generation_time:.2f}s")
            
            return {
                "success": True,
                "response": response_content,
                "metadata": metadata,
                "chunks_used": chunks,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return {
                "success": False,
                "response": f"I apologize, but I encountered an error while generating a response: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "generation_time": time.time() - start_time if 'start_time' in locals() else 0,
                    "chunks_used": len(chunks) if chunks else 0
                },
                "chunks_used": chunks or [],
                "query": query
            }
    
    def _format_chunks_as_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a coherent context string.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with source information
            chunk_text = f"[Source {i}: {chunk.source}]\n{chunk.text}\n"
            if hasattr(chunk, 'metadata') and chunk.metadata:
                # Add relevant metadata if available
                if 'page' in chunk.metadata:
                    chunk_text = f"[Source {i}: {chunk.source}, Page {chunk.metadata['page']}]\n{chunk.text}\n"
            
            context_parts.append(chunk_text)
        
        return "\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build the prompt messages for the OpenAI API.
        
        Args:
            query: User's question
            context: Formatted context from chunks
            conversation_history: Previous conversation
            system_prompt: Custom system prompt
            
        Returns:
            List of message dictionaries
        """
        # Default system prompt
        default_system_prompt = """You are a Customer Care AI Assistant that answers questions based on the provided context with utmost accuracy. Your replies should be like you are replying to that person on Whatsapp and more friendly. Your tone should be friendly, empathetic, caring and professional.
Guidelines:
1. Use the provided context to answer the user's question accurately
2. If the context doesn't contain enough information, say so clearly
3. Cite sources when possible using the source information provided
4. Be concise but comprehensive in your responses
5. If the question cannot be answered from the context, explain what information is missing
6. Maintain a helpful and professional tone

Context:
{context}

Please answer the user's question based on this context."""
        
        # Use custom system prompt if provided
        final_system_prompt = system_prompt or default_system_prompt
        final_system_prompt = final_system_prompt.format(context=context)
        
        messages = [
            {"role": "system", "content": final_system_prompt}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for i, msg in enumerate(conversation_history[-5:]):  # Limit to last 5 messages
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})
        
        # Add the current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def stream_response(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        conversation_history: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate a streaming response for real-time output.
        
        Args:
            query: User's question/query
            chunks: Retrieved chunks providing context
            conversation_history: Previous conversation messages
            system_prompt: Custom system prompt (optional)
            **kwargs: Additional parameters for OpenAI API
            
        Yields:
            Response chunks as they are generated
        """
        try:
            logger.info(f"Starting streaming response for query: '{query[:50]}...'")
            
            # Prepare the context from chunks
            context = self._format_chunks_as_context(chunks)
            
            # Build the prompt
            messages = self._build_prompt(
                query=query,
                context=context,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            # Set up API parameters for streaming
            api_params = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": True
            }
            
            # Generate streaming response
            stream = self.client.chat.completions.create(**api_params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}")
            yield f"Error: {str(e)}"


# Create global instance
general_agent = GeneralAgent()


def ask_agent(
    query: str,
    chunks: List[RetrievedChunk],
    conversation_history: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to ask the general agent a question.
    
    Args:
        query: User's question
        chunks: Retrieved chunks for context
        conversation_history: Previous conversation
        system_prompt: Custom system prompt
        stream: Whether to return streaming response
        **kwargs: Additional parameters
        
    Returns:
        Response dictionary or generator for streaming
    """
    if stream:
        return general_agent.stream_response(
            query=query,
            chunks=chunks,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            **kwargs
        )
    else:
        return general_agent.generate_response(
            query=query,
            chunks=chunks,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            **kwargs
        )


# Specialized prompt templates for different use cases
class PromptTemplates:
    """Collection of specialized prompt templates for different scenarios."""
    
    @staticmethod
    def qa_template() -> str:
        """Template for question-answering tasks."""
        return """You are a precise question-answering assistant. Answer the user's question based on the provided context.

Guidelines:
- Provide direct, accurate answers
- Cite specific sources when possible
- If unsure, state your confidence level
- Keep responses focused and relevant

Context:
{context}

Answer the question precisely based on this context."""
    
    @staticmethod
    def summarization_template() -> str:
        """Template for summarization tasks."""
        return """You are a summarization expert. Create a comprehensive summary based on the provided context.

Guidelines:
- Extract key points and main ideas
- Organize information logically
- Maintain important details
- Use clear, concise language

Context:
{context}

Provide a well-structured summary of the key information."""
    
    @staticmethod
    def analysis_template() -> str:
        """Template for analytical tasks."""
        return """You are an analytical assistant. Analyze the provided context to answer the user's question.

Guidelines:
- Break down complex information
- Identify patterns and relationships
- Provide insights and implications
- Support conclusions with evidence from the context

Context:
{context}

Analyze the information and provide a thoughtful response."""
    
    @staticmethod
    def comparison_template() -> str:
        """Template for comparison tasks."""
        return """You are a comparison specialist. Compare and contrast information from the provided context.

Guidelines:
- Identify similarities and differences
- Organize comparisons clearly
- Highlight key distinguishing factors
- Provide balanced analysis

Context:
{context}

Compare the relevant information to answer the user's question."""
