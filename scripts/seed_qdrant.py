"""Seed Qdrant with a few synthetic text and image vectors for tests.

Run: python scripts/seed_qdrant.py
"""
import os
import sys
import uuid
from pathlib import Path
import numpy as np
from qdrant_client.models import PointStruct, VectorParams

# ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import qdrant, COLLECTION_NAME, VECTOR_SIZE, QDRANT_DISTANCE, DISTANCE_MAP
from qdrant_client.models import VectorParams


def ensure_collection():
    # Attempt to create the collection explicitly and verify
    params = VectorParams(size=int(VECTOR_SIZE), distance=DISTANCE_MAP.get(QDRANT_DISTANCE, None))
    try:
        existing = [c.name for c in qdrant.get_collections().collections]
    except Exception as e:
        print("Failed to list collections:", e)
        existing = []

    if COLLECTION_NAME in existing:
        try:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
        except Exception:
            pass

    # Fallback to HTTP API to create collection to avoid client signature mismatch
    try:
        import requests
        url = f"http://{os.getenv('QDRANT_HOST','localhost')}:{os.getenv('QDRANT_PORT','6333')}/collections/{COLLECTION_NAME}"
        payload = {"vectors": {"size": int(VECTOR_SIZE), "distance": os.getenv('QDRANT_DISTANCE', 'cosine')}}
        r = requests.put(url, json=payload)
        if r.status_code not in (200, 201):
            print('HTTP create collection failed:', r.status_code, r.text)
            raise RuntimeError('Failed to create collection via HTTP')
        print(f"Created collection {COLLECTION_NAME} via HTTP API")
    except Exception as e:
        print("Failed to create collection via HTTP:", e)
        raise


def seed():
    ensure_collection()

    # create three normalized vectors
    a = np.zeros(VECTOR_SIZE, dtype=np.float32)
    b = np.zeros(VECTOR_SIZE, dtype=np.float32)
    a[0] = 1.0
    b[1] = 1.0
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    points = []
    points.append(PointStruct(id=str(uuid.uuid4()), vector=a.tolist(), payload={"modality":"text","content":"text:alpha","caption":"alpha caption"}))
    points.append(PointStruct(id=str(uuid.uuid4()), vector=b.tolist(), payload={"modality":"text","content":"text:beta","caption":"beta caption"}))
    points.append(PointStruct(id=str(uuid.uuid4()), vector=a.tolist(), payload={"modality":"image","content":"image:alpha","caption":"alpha image caption"}))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Seeded Qdrant with synthetic points.")


if __name__ == '__main__':
    seed()
