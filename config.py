"""
Configuration and client initialization for the Agentic RAG System
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from clip_utils import get_clip, encode_text
from groq import Groq

load_dotenv()


def env(name: str, cast=str, default=None):
    """Get environment variable with type casting"""
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return cast(value)


# LLM Configuration
GROQ_API_KEY = env("GROQ_API_KEY")
LLM_MODEL = env("LLM_MODEL")
TEMPERATURE = env("TEMPERATURE", float)
TOP_P = env("TOP_P", float)
MAX_TOKENS = env("MAX_TOKENS", int)

# Embedding Configuration
EMBEDDING_MODEL_NAME = env("EMBEDDING_MODEL_NAME")
VECTOR_SIZE = env("VECTOR_SIZE", int)

# RAG Configuration
TOP_K = env("TOP_K", int)
CHUNK_SIZE = env("CHUNK_SIZE", int)
CHUNK_OVERLAP = env("CHUNK_OVERLAP", int)

# Qdrant Configuration
QDRANT_HOST = env("QDRANT_HOST")
QDRANT_PORT = env("QDRANT_PORT", int)
COLLECTION_NAME = env("QDRANT_COLLECTION_NAME")
QDRANT_DISTANCE = env("QDRANT_DISTANCE")

DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
}


# Initialize global clients
groq_client = Groq(api_key=GROQ_API_KEY)
clip_model, clip_processor, CLIP_DEVICE = get_clip()


class CLIPEmbedder:
    def encode(self, texts):
        # returns numpy array (B, D) or (D,) depending on input
        emb = encode_text(texts)
        return emb


# expose `embedder` compatibility shim used elsewhere in the codebase
embedder = CLIPEmbedder()

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
