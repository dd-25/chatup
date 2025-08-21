from typing import List, Any
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from src.chatup.config import settings

# Initialize Pinecone client
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
INDEX_NAME = settings.PINECONE_INDEX_NAME


# def create_index_if_not_exists(dim: int = 1536, metric: str = "cosine"):
#     """
#     Create the Pinecone index with the specified dimension if it does not exist.
#     """
#     indexes = [idx.name for idx in pc.list_indexes()]
#     if INDEX_NAME not in indexes:
#         spec = ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1)
#         return pc.create_index(
#             name=INDEX_NAME,
#             dimension=dim,
#             metric=metric,
#             spec=spec
#         )
# create_index_if_not_exists()


def get_index():
    """Get the Pinecone Index client for the configured index (assumes it already exists)."""
    idx_config = pc.describe_index(INDEX_NAME)
    return pc.Index(host=idx_config.host)

def upsert_embeddings(vectors: List[Any], namespace: str = "default"):
    """
    Upsert embeddings into Pinecone. Vectors should be a list of tuples: (id, values, metadata)
    """
    idx = get_index()
    return idx.upsert(vectors=vectors, namespace=namespace)

def query_embeddings(vector: List[float], top_k: int = 5, namespace: str = "default"):
    """
    Query top-K similar vectors from a namespace.
    """
    idx = get_index()
    return idx.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=True)

def delete_namespace(namespace: str):
    """
    Delete all vectors in a specific namespace.
    """
    idx = get_index()
    idx.delete(delete_all=True, namespace=namespace)
