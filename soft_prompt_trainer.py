import torch
import torch.nn as nn
from typing import Optional


class SoftPromptAdapter(nn.Module):
    """Maps CLIP-style vectors to an LLM token embedding prefix.

    This is a lightweight, trainable adapter that projects CLIP vectors into
    a sequence of token embeddings (a soft prompt). It is a scaffold — training
    requires a chosen LLM/token-embedding space and supervised pairs.
    """

    def __init__(self, clip_dim: int = 512, prefix_length: int = 8, token_emb_dim: int = 768):
        super().__init__()
        self.clip_dim = clip_dim
        self.prefix_length = prefix_length
        self.token_emb_dim = token_emb_dim
        # map clip vector -> (prefix_length * token_emb_dim)
        self.mapper = nn.Linear(clip_dim, prefix_length * token_emb_dim)

    def forward(self, clip_vec: torch.Tensor) -> torch.Tensor:
        """Return shape: (B, prefix_length, token_emb_dim)"""
        x = self.mapper(clip_vec)
        B = x.shape[0]
        x = x.view(B, self.prefix_length, self.token_emb_dim)
        return x


def train_soft_prompt(adapter: SoftPromptAdapter, dataloader, lr: float = 1e-4, epochs: int = 1, device: Optional[str] = None):
    """Simple trainer loop placeholder.

    Args:
      adapter: SoftPromptAdapter instance
      dataloader: iterable yielding (clip_vec, target_embedding_seq) pairs
      lr: learning rate
      epochs: number of epochs
      device: cpu/cuda

    Note: `target_embedding_seq` should be a tensor shaped like the adapter output
    (prefix_length, token_emb_dim) or a supervision signal; in practice you need
    a suitable objective (e.g., LM loss with injected soft prompt) and access to
    the LLM token embeddings.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    adapter.to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    adapter.train()
    for epoch in range(epochs):
        total = 0.0
        for clip_vec, target in dataloader:
            clip_vec = clip_vec.to(device)
            target = target.to(device)
            pred = adapter(clip_vec)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{epochs} avg loss: {total:.4f}")

    return adapter


if __name__ == '__main__':
    print('soft_prompt_trainer scaffold loaded. Provide a dataloader and LLM token embeddings to train.')
