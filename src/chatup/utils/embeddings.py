from langchain_openai import OpenAIEmbeddings
from functools import lru_cache
from typing import Union, List
from src.chatup.constants import EMBEDDING_SETTINGS
from src.chatup.config import settings


@lru_cache
def get_embeddings():
    return OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        model=EMBEDDING_SETTINGS.EMBED_MODEL,
    )
    

def embed_texts(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Embed text(s) using OpenAI embeddings.
    
    Args:
        texts: Single text string or list of text strings
        
    Returns:
        Single embedding vector for string input, list of vectors for list input
    """
    embeddings = get_embeddings()
    
    # Handle single string input
    if isinstance(texts, str):
        return embeddings.embed_query(texts)
    
    # Handle list input
    return [embeddings.embed_query(text) for text in texts]