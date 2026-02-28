"""Train `EmbedToTokenAdapter` to map CLIP image embeddings -> LLM prefix tokens.

This is a skeleton/training scaffold. The exact adapter forward signature may
require small edits depending on the `EmbedToTokenAdapter` implementation.
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

from scripts.dataset import SimpleImageCaptionDataset
from clip_utils import get_clip, encode_image
from embed_to_token_adapter import EmbedToTokenAdapter
from scripts.cross_attention_adapter import CrossAttentionAdapter


def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    ds = SimpleImageCaptionDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # CLIP
    clip_model, clip_processor, clip_device = get_clip()

    # LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model).to(device)
    model.eval()  # keep LM frozen by default; adapter will be trained

    # Adapter or fusion
    adapter = EmbedToTokenAdapter(clip_dim=512, prefix_length=args.prefix_length, token_emb_dim=model.config.hidden_size)
    adapter = adapter.to(device)

    fusion = None
    if args.use_fusion:
        fusion = CrossAttentionAdapter(clip_dim=512, prefix_length=args.prefix_length, token_emb_dim=model.config.hidden_size, num_heads=args.num_heads).to(device)

    params = list(adapter.parameters())
    if fusion is not None:
        params = list(fusion.parameters())

    optim = AdamW(params, lr=args.lr)

    for epoch in range(args.epochs):
        for images, captions in dl:
            # compute CLIP embeddings for images
            # encode_image expects a list of PIL images
            img_embs = encode_image(images)  # numpy (B, D)
            img_embs = torch.from_numpy(img_embs).to(device)

            # produce prefix embeddings
            if fusion is not None:
                prefix_embeds = fusion(img_embs)
            else:
                prefix_embeds = adapter(img_embs)

            # tokenize captions
            tok = tokenizer(list(captions), return_tensors='pt', padding=True)
            input_ids = tok['input_ids'].to(device)
            labels = input_ids.clone()

            # get token embeddings and prepend prefix
            token_emb = model.get_input_embeddings()(input_ids)
            # concatenate along sequence dim: prefix first
            inputs_embeds = torch.cat([prefix_embeds, token_emb], dim=1)

            # shift labels to account for prefix (we don't have token labels for prefix positions)
            prefix_len = prefix_embeds.shape[1]
            pad = torch.full((labels.shape[0], prefix_len), -100, dtype=labels.dtype, device=device)
            labels_with_prefix = torch.cat([pad, labels], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, labels=labels_with_prefix)
            loss = outputs.loss

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1}/{args.epochs} finished — loss {loss.item():.4f}")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    # prefer adapter.save if available
    try:
        adapter.save(args.out)
    except Exception:
        torch.save(adapter.state_dict(), args.out)
    print(f"Saved adapter to {args.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='CSV with image_path,caption')
    p.add_argument('--out', required=True, help='Path to save trained adapter')
    p.add_argument('--llm_model', default=os.getenv('LLM_MODEL', 'distilgpt2'))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--prefix_length', type=int, default=8)
    args = p.parse_args()
    main(args)
