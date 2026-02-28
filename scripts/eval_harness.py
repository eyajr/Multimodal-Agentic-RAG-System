"""Evaluation harness that measures CLIP-alignment between images and captions.

For each (image, caption) pair in a CSV, the harness:
- computes CLIP image embedding
- computes CLIP text embedding for the gold caption
- (optionally) generates a caption via `MultimodalLLM` or a baseline
- computes and reports cosine similarities

This provides a lightweight, reproducible metric to show multimodal gains
without heavy metric dependencies.
"""
import argparse
import csv
import sys
import numpy as np
from pathlib import Path

# Ensure repository root is on sys.path so top-level imports like `clip_utils`
# work when running this file directly from the `scripts/` folder.
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from clip_utils import get_clip, encode_image, encode_text


def main(args):
    clip_model, clip_processor, device = get_clip()
    rows = []
    with open(args.data, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        for r in reader:
            if not r:
                continue
            if r[0].lower().startswith('image'):
                continue
            rows.append((r[0], r[1] if len(r) > 1 else ''))

    gold_sims = []
    gen_sims = []

    # optional multimodal generator
    try:
        from multimodal_llm import MultimodalLLM
        mm = MultimodalLLM()
    except Exception:
        mm = None

    for img_path, caption in rows:
        try:
            img_emb = encode_image([__import__('PIL').Image.open(img_path).convert('RGB')])[0]
        except Exception as e:
            print('Failed to load image', img_path, e)
            continue

        text_emb = encode_text(caption)[0]
        # cosine
        cos = lambda a,b: np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        gold = cos(img_emb, text_emb)
        gold_sims.append(gold)

        gen_caption = None
        if mm is not None:
            try:
                gen_caption = mm.generate_with_images('Describe the image.', [], image_paths=[img_path])
            except Exception:
                gen_caption = None

        if gen_caption:
            gen_emb = encode_text(gen_caption)[0]
            gen_sims.append(cos(img_emb, gen_emb))
        else:
            gen_sims.append(0.0)

        print(f'Image: {img_path} | gold_sim={gold:.4f} | gen_sim={gen_sims[-1]:.4f}')

    import statistics
    print('--- Summary ---')
    print('Avg gold sim:', statistics.mean(gold_sims) if gold_sims else 0.0)
    print('Avg gen sim :', statistics.mean(gen_sims) if gen_sims else 0.0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='CSV image_path,caption')
    args = p.parse_args()
    main(args)
