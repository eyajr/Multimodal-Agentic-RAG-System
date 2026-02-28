import os
import base64
import torch
from typing import Optional
from soft_prompt_trainer import SoftPromptAdapter


class EmbedToTokenAdapter(SoftPromptAdapter):
    """Thin wrapper around SoftPromptAdapter with save/load helpers.

    Usage: create adapter with matching `clip_dim`, `prefix_length`, and
    `token_emb_dim`. After training, save via `save(path)`. At inference,
    set env `SOFT_PROMPT_ADAPTER_PATH` to the saved file and `rag.generate_answer`
    will attempt to load and use it.
    """

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, clip_dim: int = 512, prefix_length: int = 8, token_emb_dim: int = 768, device: Optional[str] = None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = cls(clip_dim=clip_dim, prefix_length=prefix_length, token_emb_dim=token_emb_dim)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model


def embeddings_to_base64(tensor: torch.Tensor) -> str:
    """Convert a float32 tensor to a compact base64 string for inclusion in prompts.

    The consumer must know how to decode: base64 -> bytes -> float32 -> reshape.
    """
    arr = tensor.detach().cpu().numpy().astype('float32')
    b = arr.tobytes()
    return base64.b64encode(b).decode('ascii')


def base64_to_embeddings(s: str, shape: tuple):
    import numpy as _np
    b = base64.b64decode(s.encode('ascii'))
    arr = _np.frombuffer(b, dtype=_np.float32).reshape(shape)
    return arr
