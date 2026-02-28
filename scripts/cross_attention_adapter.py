import torch
import torch.nn as nn


class CrossAttentionAdapter(nn.Module):
    """Produce a prefix by attending learned prefix queries to image-derived memory.

    Forward input: image_embeddings (B, clip_dim)
    Output: prefix embeddings (B, prefix_length, token_emb_dim)
    """

    def __init__(self, clip_dim: int, prefix_length: int, token_emb_dim: int, num_heads: int = 8):
        super().__init__()
        self.clip_dim = clip_dim
        self.prefix_length = prefix_length
        self.token_emb_dim = token_emb_dim
        self.num_heads = num_heads

        # Map clip embedding to a memory vector (could be expanded to multiple slots)
        self.memory_proj = nn.Linear(clip_dim, token_emb_dim)

        # Learned prefix queries (prefix_length, token_emb_dim)
        self.prefix_queries = nn.Parameter(torch.randn(prefix_length, token_emb_dim) * 0.02)

        # Cross-attention module (batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=token_emb_dim, num_heads=num_heads, batch_first=True)

        self.out_proj = nn.Linear(token_emb_dim, token_emb_dim)

    def forward(self, image_embs: torch.Tensor) -> torch.Tensor:
        # image_embs: (B, clip_dim)
        B = image_embs.shape[0]
        mem = self.memory_proj(image_embs).unsqueeze(1)  # (B, 1, token_emb_dim)

        # queries: expand learned queries to batch
        queries = self.prefix_queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, prefix_length, token_emb_dim)

        # MultiheadAttention expects (B, L, E) when batch_first=True
        attn_out, _ = self.cross_attn(query=queries, key=mem, value=mem)
        out = self.out_proj(attn_out)
        return out
