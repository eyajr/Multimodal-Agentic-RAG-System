import os
import torch
import numpy as np


def test_cross_attention_adapter_forward():
    from scripts.cross_attention_adapter import CrossAttentionAdapter

    clip_dim = 512
    prefix_length = 4
    token_emb_dim = 128
    num_heads = 8

    adapter = CrossAttentionAdapter(clip_dim=clip_dim, prefix_length=prefix_length, token_emb_dim=token_emb_dim, num_heads=num_heads)
    x = torch.randn(3, clip_dim)
    out = adapter(x)
    assert out.shape == (3, prefix_length, token_emb_dim)


def test_adapter_training_smoke(tmp_path):
    """Run a tiny training loop for EmbedToTokenAdapter and verify save/load."""
    import torch
    from embed_to_token_adapter import EmbedToTokenAdapter

    clip_dim = 64
    prefix_length = 3
    token_emb_dim = 32

    adapter = EmbedToTokenAdapter(clip_dim=clip_dim, prefix_length=prefix_length, token_emb_dim=token_emb_dim)
    adapter.train()

    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    # synthetic data
    B = 2
    for step in range(2):
        x = torch.randn(B, clip_dim)
        target = torch.randn(B, prefix_length, token_emb_dim)
        out = adapter(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # save and load
    out_path = tmp_path / "adapter_test.pt"
    adapter.save(str(out_path))
    assert out_path.exists()

    loaded = EmbedToTokenAdapter.load(str(out_path), clip_dim=clip_dim, prefix_length=prefix_length, token_emb_dim=token_emb_dim, device='cpu')
    assert loaded is not None
    # forward with loaded model
    loaded_out = loaded(torch.randn(1, clip_dim))
    assert loaded_out.shape == (1, prefix_length, token_emb_dim)


def test_retrieval_precision_smoke(monkeypatch):
    """Index a few synthetic items into a fake Qdrant and verify retrieval matches top-k."""
    import types

    import rag

    # create synthetic storage
    storage = []

    def add_point(content, vector, modality='text'):
        storage.append({'vector': np.array(vector, dtype=np.float32), 'payload': {'modality': modality, 'content': content}})

    # two text and one image vectors (normalized)
    v_text1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_text1 = v_text1 / np.linalg.norm(v_text1)
    v_text2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_text2 = v_text2 / np.linalg.norm(v_text2)
    v_image = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_image = v_image / np.linalg.norm(v_image)

    add_point('text:alpha', v_text1.tolist(), modality='text')
    add_point('text:beta', v_text2.tolist(), modality='text')
    add_point('image:alpha', v_image.tolist(), modality='image')

    class FakeHit:
        def __init__(self, payload, score):
            self.payload = payload
            self.score = float(score)

    class FakeResults:
        def __init__(self, points):
            self.points = points

    class FakeQdrant:
        def __init__(self, storage):
            self._storage = storage

        def query_points(self, collection_name, query, limit=5):
            q = np.array(query, dtype=np.float32)
            # normalize query
            if q.ndim == 1:
                qn = q / (np.linalg.norm(q) + 1e-12)
            else:
                qn = q
            hits = []
            for p in self._storage:
                vec = p['vector']
                score = float(np.dot(qn, vec))
                hits.append(FakeHit(p['payload'], score))
            # sort desc
            hits.sort(key=lambda h: h.score, reverse=True)
            return FakeResults(hits[:limit])

    fake = FakeQdrant(storage)
    monkeypatch.setattr(rag, 'qdrant', fake)

    # monkeypatch clip_model and clip_processor to return query that equals v_text1
    class DummyProcessor:
        def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=None):
            return {'input_ids': torch.tensor([[1]])}

    class DummyClip:
        def get_text_features(self, **kwargs):
            import torch
            # return vector matching v_text1
            return torch.tensor([v_text1], dtype=torch.float32)

    monkeypatch.setattr(rag, 'clip_processor', DummyProcessor())
    monkeypatch.setattr(rag, 'clip_model', DummyClip())

    # call the retrieval function; should return items with content where top hit is text:alpha or image:alpha
    res = rag.retrieve_with_scores('alpha')
    assert isinstance(res, list)
    assert len(res) > 0
    top_content, top_score = res[0]
    assert top_content in ('text:alpha', 'image:alpha')
