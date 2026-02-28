"""Simple demo pipeline that exercises ingestion, retrieval and SD guidance.

This script is intended to be runnable from the UI or CLI to demonstrate the
end-to-end components. It prints progress to stdout so the UI can stream logs.
"""
import argparse
import os
from pathlib import Path

def main(args):
    print('Demo pipeline started')

    # Ingest images if provided
    if args.image_dir and os.path.exists(args.image_dir):
        try:
            import rag
            print('Ingesting images from', args.image_dir)
            rag.ingest_images(args.image_dir)
            print('Image ingestion complete')
        except Exception as e:
            print('Image ingestion skipped or failed:', e)

    # Run a sample retrieval
    try:
        from rag import retrieve_with_scores
        print('Running sample retrieval for query:', args.query)
        res = retrieve_with_scores(args.query)
        print('Retrieval results:')
        for c, s in res[:5]:
            print('-', c[:120].replace('\n',' ') , f'(score={s:.4f})')
    except Exception as e:
        print('Retrieval failed:', e)

    # Optionally run SD guidance to refine a generated image toward the top image
    try:
        from scripts.sd_clip_guidance import main as sd_main
        if args.refine and res:
            top = res[0][0]
            print('Attempting SD+CLIP refine (may be slow)...')
            # call sd_clip_guidance with a simple prompt and target_text
            sd_args = ["--prompt", args.prompt or "Generate scene", "--target_text", args.query, "--steps", "20", "--output", args.output]
            sd_main(argparse.Namespace(prompt=args.prompt or 'Generate scene', target_image=None, target_text=args.query, steps=20, lr=0.1, output=args.output))
            print('SD guidance finished. Output:', args.output)
    except Exception as e:
        print('SD guidance skipped or failed:', e)

    print('Demo pipeline finished')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', help='Optional directory of images to ingest')
    p.add_argument('--query', default='A photo of a cat')
    p.add_argument('--refine', action='store_true')
    p.add_argument('--prompt', help='SD prompt for refinement')
    p.add_argument('--output', default='outputs/demo_refined.png')
    args = p.parse_args()
    main(args)
