import numpy as np
import torch

from image_generation import ExampleDiffusionModel, generate_image_from_embedding
from clip_utils import summarize_vector
try:
    import pytest
except Exception:
    pytest = None
from config import qdrant, COLLECTION_NAME


def test_generate_image_from_random_clip_vector():
    dim = 512
    vec = np.random.randn(dim).astype(np.float32)
    model = ExampleDiffusionModel(clip_dim=dim, device='cpu', out_shape=(3, 32, 32))
    x = generate_image_from_embedding(vec, model, device='cpu', save_dir='.', w=0)
    assert isinstance(x, torch.Tensor)
    assert x.shape[1:] == (3, 32, 32)


def test_summarize_vector():
    v = np.array([1.0, 0.0, -1.0, 0.5, 0.2, -0.3, 0.7, 0.8])
    s = summarize_vector(v, n=4)
    assert 'first_4' in s
    assert 'mean=' in s and 'std=' in s


def test_adapter_forward():
    """Ensure EmbedToTokenAdapter (soft prompt) forward pass produces expected shape."""
    import torch
    from embed_to_token_adapter import EmbedToTokenAdapter

    clip_dim = 512
    prefix_length = 4
    token_emb_dim = 128
    adapter = EmbedToTokenAdapter(clip_dim=clip_dim, prefix_length=prefix_length, token_emb_dim=token_emb_dim)
    x = torch.randn(2, clip_dim)
    out = adapter(x)
    assert out.shape == (2, prefix_length, token_emb_dim)


def test_multimodal_tool_returns_vectors():
    from tools import MultimodalSearchTool
    tool = MultimodalSearchTool()
    res = tool.execute('diagram', with_vectors=True)
    assert res.success is True
    data = res.data
    assert 'image_results' in data
    imgs = data['image_results']
    assert isinstance(imgs, list)
    if imgs:
        assert imgs[0].get('vector') is not None
        # payload should include caption after ingestion
        payload = imgs[0].get('payload', {})
        assert isinstance(payload, dict)
        assert 'caption' in payload and payload['caption']


def test_visual_conditioned_generation_runs():
    """Smoke test: retrieve image vectors and run generate_answer() to ensure
    image-conditioned prompt path executes without error and returns text."""
    from tools import MultimodalSearchTool
    import rag

    tool = MultimodalSearchTool()
    res = tool.execute('diagram', with_vectors=True)
    assert res.success is True
    image_refs = res.data.get('image_results', [])

    answer = rag.generate_answer('Describe the main subject of the top image.', [], image_refs=image_refs)
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


def test_qdrant_vectors_are_normalized():
    """Verify that returned vectors from retrieval are approximately unit norm."""
    import rag
    res = rag.retrieve_with_vectors('diagram')
    assert isinstance(res, list)
    if res:
        # check any vector present is normalized to ~1
        import numpy as np
        found = False
        for content, score, vec in res:
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                assert abs(norm - 1.0) < 1e-3
                found = True
        assert found


    def test_all_image_points_have_captions():
        """Assert every stored image point in Qdrant has a non-empty caption payload."""
        # Use qdrant.scroll to iterate all points; skip if API unavailable.
        try:
            points = []
            limit = 128
            offset = 0
            while True:
                res = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True, limit=limit, offset=offset)
                batch = getattr(res, 'points', None)
                if not batch:
                    break
                points.extend(batch)
                offset += len(batch)
                if len(batch) < limit:
                    break
        except AttributeError:
            if pytest is not None:
                pytest.skip('qdrant.scroll not available in this client version')
            else:
                print('SKIP: qdrant.scroll not available in this client version')
                return

        # If there are no points, skip the test to avoid false failure.
        if not points:
            if pytest is not None:
                pytest.skip('No points in collection to verify')
            else:
                print('SKIP: No points in collection to verify')
                return

        found_image = False
        for p in points:
            payload = getattr(p, 'payload', {}) or {}
            if payload.get('modality') == 'image':
                found_image = True
                caption = payload.get('caption')
                assert caption is not None and str(caption).strip() != ''

        assert found_image, 'No image modality points found in collection'
