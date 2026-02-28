"""Direct caption check using Qdrant client without importing project `config`.

This avoids importing `config.py` (which loads large HF models) so it can run
quickly in CI or lightweight environments.
"""
import os
import sys
import traceback
from qdrant_client import QdrantClient


def main():
    try:
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', '6333'))
        collection = os.getenv('QDRANT_COLLECTION_NAME')
        if not collection:
            print('ENV MISSING: QDRANT_COLLECTION_NAME')
            sys.exit(2)

        client = QdrantClient(host=host, port=port)

        points = []
        limit = 128
        offset = 0
        while True:
            res = client.scroll(collection_name=collection, with_payload=True, limit=limit, offset=offset)
            batch = getattr(res, 'points', None)
            if not batch:
                break
            points.extend(batch)
            offset += len(batch)
            if len(batch) < limit:
                break

        if not points:
            print('NO_POINTS')
            sys.exit(2)

        found_image = False
        for p in points:
            payload = getattr(p, 'payload', {}) or {}
            if payload.get('modality') == 'image':
                found_image = True
                caption = payload.get('caption')
                if caption is None or str(caption).strip() == '':
                    print('MISSING_CAPTION', getattr(p, 'id', None))
                    sys.exit(3)

        if not found_image:
            print('NO_IMAGE_POINTS')
            sys.exit(2)

        print('ALL_CAPTIONS_PRESENT')
        sys.exit(0)

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
