import re
from typing import List
import json
from src.chatup.utils.token import count_tokens
from src.chatup.constants import CHUNKING_SETTINGS


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts if parts else [text.strip()]

def split_json_chunk(chunk_text: str) -> List[str]:
    """
    Split a JSON chunk that's too large into smaller, meaningful pieces.
    Preserves JSON structure and context better than sentence splitting.
    """
    # chunk_text is already parsed and formatted text, not raw JSON
    lines = chunk_text.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try adding this line to current chunk
        test_chunk = current_chunk + "\n" + line if current_chunk else line
        
        if count_tokens(test_chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If we still have chunks that are too large, split by individual key-value pairs
    final_chunks = []
    for chunk in chunks:
        if count_tokens(chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
            final_chunks.append(chunk)
        else:
            # Split by individual key-value pairs (lines that contain ":")
            pairs = [line.strip() for line in chunk.split('\n') if ':' in line]
            current_pair_chunk = ""
            
            for pair in pairs:
                test_chunk = current_pair_chunk + "\n" + pair if current_pair_chunk else pair
                if count_tokens(test_chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
                    current_pair_chunk = test_chunk
                else:
                    if current_pair_chunk:
                        final_chunks.append(current_pair_chunk.strip())
                    current_pair_chunk = pair
            
            if current_pair_chunk:
                final_chunks.append(current_pair_chunk.strip())
    
    return final_chunks