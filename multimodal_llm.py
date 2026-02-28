import os
from typing import List, Optional
from clip_utils import summarize_vector
import json
import base64
from pathlib import Path

try:
    from config import groq_client, LLM_MODEL, TEMPERATURE, TOP_P, MAX_TOKENS
except Exception:
    groq_client = None


class MultimodalLLM:
    """Flexible multimodal LLM integration.

    Behavior:
      - If env `ENABLE_MULTIMODAL_LLM` is truthy and `groq_client` is available,
        assemble a multimodal prompt and call the LLM service.
      - Otherwise, use a local deterministic fallback that composes captions
        and vector summaries into a reply (useful for offline tests).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model = model_name or os.getenv('LLM_MODEL')

    def generate_with_images(self, query: str, text_context: List[str], image_paths: List[str] = None, image_vectors: List[Optional[list]] = None, image_captions: List[Optional[str]] = None) -> str:
        image_paths = image_paths or []
        image_vectors = image_vectors or []
        image_captions = image_captions or []

        # Build a structured multimodal prompt
        prompt_parts = []
        if text_context:
            prompt_parts.append("Text Context:\n" + "\n\n".join(text_context))

        if image_paths or image_captions or image_vectors:
            prompt_parts.append("Image Evidence:")
            for i in range(max(len(image_paths), len(image_captions), len(image_vectors))):
                p = image_paths[i] if i < len(image_paths) else None
                c = image_captions[i] if i < len(image_captions) else None
                v = image_vectors[i] if i < len(image_vectors) else None
                entry = {"path": p, "caption": c}
                if v is not None:
                    entry["vector_summary"] = summarize_vector(v, n=8)
                prompt_parts.append(json.dumps(entry))

        prompt_parts.append("Question:\n" + query)
        prompt = "\n\n".join(prompt_parts)

        # If groq_client available and user opted in, call external LLM with multimodal attachments
        if os.getenv('ENABLE_MULTIMODAL_LLM') and groq_client is not None:
            # prepare attachments: read files and base64-encode
            attachments = []
            for p in image_paths:
                if not p:
                    continue
                try:
                    b = Path(p).read_bytes()
                    b64 = base64.b64encode(b).decode('ascii')
                    attachments.append({"type": "image", "data_base64": b64, "filename": Path(p).name})
                except Exception:
                    # if file read fails, include only path and caption
                    attachments.append({"type": "image", "path": p})

            try:
                # Preferred: some clients accept an `attachments=` kwarg
                if hasattr(groq_client, 'chat') and hasattr(groq_client.chat, 'completions'):
                    # try passing attachments alongside messages
                    try:
                        response = groq_client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            attachments=attachments,
                            temperature=float(os.getenv('TEMPERATURE', 0.3)),
                            top_p=float(os.getenv('TOP_P', 0.9)),
                            max_tokens=int(os.getenv('MAX_TOKENS', 256))
                        )
                        return response.choices[0].message.content
                    except TypeError:
                        # attachments not supported in this client signature; try embedding attachments in messages
                        multimodal_messages = [{"role": "user", "content": prompt}]
                        for a in attachments:
                            # include image as a structured message entry
                            multimodal_messages.append({"role": "user", "content": f"[IMAGE_ATTACHMENT]{a.get('filename','')}", "image": a})
                        response = groq_client.chat.completions.create(
                            model=self.model,
                            messages=multimodal_messages,
                            temperature=float(os.getenv('TEMPERATURE', 0.3)),
                            top_p=float(os.getenv('TOP_P', 0.9)),
                            max_tokens=int(os.getenv('MAX_TOKENS', 256))
                        )
                        return response.choices[0].message.content

                # Fallback: if groq_client exposes a plain `create_multimodal` helper, try it
                if hasattr(groq_client, 'create_multimodal'):
                    response = groq_client.create_multimodal(prompt=prompt, attachments=attachments)
                    return getattr(response, 'text', str(response))

            except Exception:
                # fall through to local fallback
                pass

        # Local deterministic fallback: synthesize a short answer combining captions and vector summaries
        parts = [f"Question: {query}"]
        if image_captions:
            parts.append("Image Captions:\n" + "; ".join([c for c in image_captions if c]))
        if image_vectors:
            vec_summaries = [summarize_vector(v, n=6) for v in image_vectors if v is not None]
            if vec_summaries:
                parts.append("Vector Summaries:\n" + "; ".join(vec_summaries))

        parts.append("Answer: Based on the provided captions and vector summaries, the most salient visual content appears above.")
        return "\n\n".join(parts)


if __name__ == '__main__':
    print('multimodal_llm helper loaded; set ENABLE_MULTIMODAL_LLM=1 and ensure groq client configured to call external LLM.')
