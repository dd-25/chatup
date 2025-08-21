from typing import List, TypedDict
from langgraph.graph import StateGraph, START
from src.chatup.services.data_ingestion import extract_text, semantic_split_text, embed_and_upload


class IngestionState(TypedDict):
    file_bytes: bytes
    filename: str
    namespace: str
    extract_text_node: str
    chunk_embed_node: List[str]


ingestion_graph = StateGraph(IngestionState)


def extract_text_node(state: IngestionState) -> IngestionState:
    # Extract text and return updated state, always preserve all keys
    text = extract_text(state["file_bytes"], state["filename"])
    return {**state, "extract_text_node": text}


def chunk_embed_node(state: IngestionState) -> IngestionState:
    chunks = semantic_split_text(state["extract_text_node"])
    embed_and_upload(chunks, namespace=state["namespace"], source=state["filename"])
    return {**state, "chunk_embed_node": chunks}


ingestion_graph.add_node("extract_text_node", extract_text_node)
ingestion_graph.add_node("chunk_embed_node", chunk_embed_node)

ingestion_graph.add_edge(START, "extract_text_node")
ingestion_graph.add_edge("extract_text_node", "chunk_embed_node")


compiled_graph = ingestion_graph.compile()


def run_ingestion(file_bytes: bytes, filename: str, namespace: str = "default") -> List[str]:
    initial_state = {
        "file_bytes": file_bytes,
        "filename": filename,
        "namespace": namespace,
        "extract_text_node": None,
        "chunk_embed_node": None,
    }
    result = compiled_graph.invoke(input=initial_state)
    return result["chunk_embed_node"]
