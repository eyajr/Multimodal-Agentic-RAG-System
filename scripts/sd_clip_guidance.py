"""Stable Diffusion + CLIP guidance refinement script.

Workflow:
- Load Stable Diffusion pipeline (diffusers) and CLIP model (from `clip_utils`).
- Generate initial image(s) from a `prompt` (or use provided `init_image`).
- Compute target CLIP image embedding from `target_image` (or from a provided image path).
- Refine generated latents by gradient-ascent on latents to maximize cosine similarity
  between generated images and the target CLIP embedding.

Notes:
- This script is a practical integration scaffold; depending on diffusers/transformers versions
  small API adjustments may be required.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so top-level imports like `clip_utils`
# work when running this file directly from the `scripts/` folder.
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

from clip_utils import get_clip, encode_image


def pil_from_tensor(tensor: torch.Tensor):
    # tensor expected in [-1, 1], shape (B, C, H, W)
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).round().to(torch.uint8)
    pil = Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())
    return pil


def refine_with_clip(sd_pipeline, clip_model, clip_processor, target_emb, init_prompt, device, steps=100, lr=1e-1):
    # Generate initial image
    with torch.no_grad():
        out = sd_pipeline(init_prompt, num_inference_steps=20)
        init_img = out.images[0]

    # convert PIL -> tensor normalized to [-1,1]
    init_tensor = sd_pipeline.feature_extractor(init_img, return_tensors='pt').pixel_values
    init_tensor = init_tensor.to(device)

    # Encode to latents via VAE encoder
    with torch.no_grad():
        enc = sd_pipeline.vae.encode(init_tensor).latent_dist.mean
    latents = enc.clone().detach().requires_grad_(True)

    optim = torch.optim.Adam([latents], lr=lr)

    target = torch.from_numpy(target_emb).to(device)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    for i in range(steps):
        optim.zero_grad()

        # decode latents to image pixels
        dec = sd_pipeline.vae.decode(latents).sample
        # scale to [-1,1]
        img = dec

        # convert to PIL for CLIP encode
        # assume batch size 1
        pil = pil_from_tensor(img[0])
        emb = encode_image([pil])
        emb = torch.from_numpy(emb).to(device)

        # cosine similarity
        cos = torch.nn.functional.cosine_similarity(emb, target)
        loss = -cos.mean()
        loss.backward()
        optim.step()

        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{steps} loss={loss.item():.4f} cos={cos.item():.4f}")

    # final decode
    with torch.no_grad():
        final_dec = sd_pipeline.vae.decode(latents).sample
        final_pil = pil_from_tensor(final_dec[0])
    return final_pil


def main(args):
    if StableDiffusionPipeline is None:
        raise RuntimeError('diffusers StableDiffusionPipeline not available; install diffusers')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sd = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    sd = sd.to(device)
    sd.safety_checker = None

    clip_model, clip_processor, clip_device = get_clip()

    # determine target embedding
    if args.target_image:
        img = Image.open(args.target_image).convert('RGB')
        target_emb = encode_image([img])[0]
    elif args.target_text:
        # use CLIP text encoder to get a target image-text aligned embedding
        import numpy as np
        from transformers import CLIPProcessor
        # reuse get_clip's processor for text
        inputs = clip_processor(text=[args.target_text], return_tensors='pt')
        with torch.no_grad():
            out = clip_model.get_text_features(**{k: v.to(clip_device) for k, v in inputs.items()})
        if isinstance(out, torch.Tensor):
            vec = out[0].cpu().numpy()
        else:
            vec = getattr(out, 'pooler_output', None)
            if vec is None and hasattr(out, 'last_hidden_state'):
                vec = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        target_emb = vec
    else:
        raise RuntimeError('Either --target_image or --target_text is required')

    final = refine_with_clip(sd, clip_model, clip_processor, target_emb, args.prompt, device, steps=args.steps, lr=args.lr)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    final.save(outp)
    print(f"Saved refined image to {outp}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.getenv('SD_MODEL', 'runwayml/stable-diffusion-v1-5'))
    p.add_argument('--prompt', required=True)
    p.add_argument('--target_image', help='Path to image to guide towards')
    p.add_argument('--target_text', help='Text to convert to a CLIP target embedding')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-1)
    p.add_argument('--output', default='outputs/refined.png')
    args = p.parse_args()
    main(args)
