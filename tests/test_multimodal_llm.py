def test_multimodal_llm_fallback():
    """Test the local fallback of MultimodalLLM (no external LLM call)."""
    from multimodal_llm import MultimodalLLM

    mm = MultimodalLLM()
    query = "What is in the image?"
    text_ctx = ["This is some surrounding text."]
    captions = ["a red car on a street", "people walking"]
    vectors = [[0.1, -0.2, 0.3], None]

    # Ensure env disables external call
    import os
    os.environ.pop('ENABLE_MULTIMODAL_LLM', None)

    out = mm.generate_with_images(query, text_ctx, image_paths=['/tmp/a.png', '/tmp/b.png'], image_vectors=vectors, image_captions=captions)
    assert isinstance(out, str)
    assert 'Image Captions' in out or 'Vector Summaries' in out


def test_multimodal_llm_calls_client_with_attachments(monkeypatch, tmp_path):
    """Verify multimodal path encodes attachments and calls groq client when enabled."""
    from multimodal_llm import MultimodalLLM
    import os

    # create small dummy image files
    p1 = tmp_path / 'a.png'
    p1.write_bytes(b'PNGDATA')

    # fake groq client
    class FakeResp:
        def __init__(self, text):
            class Choice:
                class Msg:
                    def __init__(self, c):
                        self.content = c
                def __init__(self):
                    self.message = FakeResp.Msg('ok')
            self.choices = [Choice()]

    class FakeCompletions:
        def create(self, **kwargs):
            # assert attachments passed
            assert 'attachments' in kwargs or any(isinstance(m.get('image'), dict) for m in kwargs.get('messages', []))
            return FakeResp('ok')

    class FakeChat:
        completions = FakeCompletions()

    fake = type('F', (), {})()
    fake.chat = FakeChat()

    # monkeypatch module groq_client to fake
    import multimodal_llm as mm_mod
    monkeypatch.setenv('ENABLE_MULTIMODAL_LLM', '1')
    monkeypatch.setattr(mm_mod, 'groq_client', fake)

    mm = MultimodalLLM()
    out = mm.generate_with_images('q', [], image_paths=[str(p1)], image_captions=['c'])
    assert isinstance(out, str)
