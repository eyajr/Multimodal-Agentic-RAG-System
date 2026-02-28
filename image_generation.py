import torch
import torch.nn as nn
import numpy as np
from labs.diffusion_utilities_context import plot_grid


# Lightweight projector that maps CLIP vectors to an image-shaped latent
class CLIPToImageProjector(nn.Module):
    def __init__(self, clip_dim: int, out_shape=(3, 32, 32)):
        super().__init__()
        self.clip_dim = clip_dim
        self.out_shape = out_shape
        out_size = out_shape[0] * out_shape[1] * out_shape[2]
        self.project = nn.Sequential(
            nn.Linear(clip_dim, out_size),
            nn.Tanh()
        )

    def forward(self, clip_vec: torch.Tensor) -> torch.Tensor:
        # clip_vec: (B, clip_dim)
        x = self.project(clip_vec)
        B = x.shape[0]
        return x.view(B, *self.out_shape)


# Example diffusion model wrapper that conditions on projected CLIP vectors.
class ExampleDiffusionModel(nn.Module):
    def __init__(self, clip_dim: int = 512, device='cpu', out_shape=(3, 32, 32)):
        super().__init__()
        self.device = torch.device(device)
        self.out_shape = out_shape
        self.projector = CLIPToImageProjector(clip_dim, out_shape).to(self.device)

    def sample(self, embedding_tensor: torch.Tensor) -> torch.Tensor:
        """
        A toy sampler that uses the projected CLIP embedding as a conditioning map
        added to random noise to produce a deterministic-ish output for demos.
        embedding_tensor: (B, clip_dim)
        returns: (B, C, H, W) tensor in [-1, 1]
        """
        B = embedding_tensor.shape[0]
        C, H, W = self.out_shape

        cond = self.projector(embedding_tensor.to(self.device))
        noise = torch.randn(B, C, H, W, device=self.device)

        # simple conditioning: add scaled conditioner to noise and tanh
        x = noise + 0.7 * cond
        x = torch.tanh(x)
        return x


def generate_image_from_embedding(embedding, diffusion_model: ExampleDiffusionModel = None):
    """Utility wrapper used by the UI: accepts a stored embedding (list/np.ndarray/torch.Tensor)
    and returns a CPU tensor image (C,H,W) in range [-1,1]. If no diffusion_model is provided,
    instantiate a local `ExampleDiffusionModel`.
    """
    import numpy as _np
    if diffusion_model is None:
        diffusion_model = ExampleDiffusionModel()

    if isinstance(embedding, list):
        emb = _np.asarray(embedding, dtype='float32')
        emb = torch.from_numpy(emb).unsqueeze(0)
    elif isinstance(embedding, _np.ndarray):
        emb = torch.from_numpy(embedding.astype('float32')).unsqueeze(0)
    elif isinstance(embedding, torch.Tensor):
        emb = embedding.unsqueeze(0) if embedding.dim() == 1 else embedding
    else:
        raise TypeError('Unsupported embedding type')

    with torch.no_grad():
        out = diffusion_model.sample(emb)
    # out: (B, C, H, W)
    return out.squeeze(0)


def generate_image_from_embedding(embedding, diffusion_model, device='cpu', save_dir='./', w=0):
    """
    Generate an image from a given CLIP embedding using the diffusion model.
    embedding: array-like or torch tensor with shape (dim,) or (B, dim)
    diffusion_model: instance of ExampleDiffusionModel (or similar) that accepts
                     CLIP embedding tensors as input to `sample()`.
    """
    if isinstance(embedding, (list, tuple)) or isinstance(embedding, np.ndarray):
        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
    else:
        emb_tensor = embedding

    # Ensure shape (B, dim)
    if emb_tensor.dim() == 1:
        emb_tensor = emb_tensor.unsqueeze(0)

    emb_tensor = emb_tensor.to(device)
    x_gen = diffusion_model.sample(emb_tensor)

    # Ensure tensor is detached before converting to numpy in plotting utilities
    if isinstance(x_gen, torch.Tensor) and x_gen.requires_grad:
        x_gen = x_gen.detach()

    plot_grid(x_gen, n_sample=x_gen.shape[0], n_rows=1, save_dir=save_dir, w=w)
    return x_gen

# Example usage:
# embedding = ... # Retrieve from Qdrant
# diffusion_model = ExampleDiffusionModel(device='cpu')
# generate_image_from_embedding(embedding, diffusion_model)
