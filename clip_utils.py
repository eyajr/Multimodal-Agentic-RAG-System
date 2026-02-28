import torch
import numpy as np
from typing import List, Union

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None


_CLIP_CACHE = {}


def get_clip(device: str = None):
    """Load and cache a CLIP model + processor. Returns (model, processor, device)."""
    global _CLIP_CACHE
    if 'clip' in _CLIP_CACHE:
        return _CLIP_CACHE['clip']

    if CLIPModel is None:
        raise RuntimeError('transformers.CLIPModel not available; install transformers')

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = model.to(device)
    model.eval()
    _CLIP_CACHE['clip'] = (model, processor, device)
    return model, processor, device


def encode_text(texts: Union[str, List[str]]):
    """Return L2-normalized CLIP text embeddings as numpy arrays.

    Args:
      texts: string or list of strings
    Returns:
      np.ndarray shape (B, D)
    """
    model, processor, device = get_clip()
    if isinstance(texts, str):
        texts = [texts]
    inputs = processor(text=texts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb.cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    return emb


def encode_image(pil_images: List):
    """Return L2-normalized CLIP image embeddings for a list of PIL images."""
    model, processor, device = get_clip()
    inputs = processor(images=pil_images, return_tensors='pt').to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb.cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    return emb


def summarize_vector(vec: Union[List[float], np.ndarray, torch.Tensor], n: int = 8) -> str:
    """Return a short textual summary of a vector (first n dims, mean, std)."""
    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().numpy()
    vec = np.asarray(vec).reshape(-1)
    first = ','.join([f'{float(x):.4f}' for x in vec[:n]])
    summary = f'first_{n}=[{first}], mean={float(vec.mean()):.4f}, std={float(vec.std()):.4f}'
    return summary


_STOPWORDS = set([
    'the','and','is','in','to','of','a','an','for','with','on','by','that','this','these','those','it','as','are','be','or','from','at','which'
])


def extract_keywords_from_texts(texts: List[str], top_k: int = 8) -> List[str]:
    """Simple keyword extractor: token frequency minus stopwords, returns top_k tokens.

    This is a lightweight, deterministic soft-prompt generator that can be
    used to synthesize a small textual prompt describing image-associated
    text context for LLM conditioning.
    """
    import re
    toks = {}
    for t in texts:
        words = re.findall(r"[A-Za-z0-9_'-]+", t.lower())
        for w in words:
            if w in _STOPWORDS or len(w) < 3 or w.isdigit():
                continue
            toks[w] = toks.get(w, 0) + 1

    items = sorted(toks.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[:top_k]]


_BLIP_CACHE = {}


def get_blip(device: str = None):
    """Load and cache BLIP captioning model and processor."""
    global _BLIP_CACHE
    if 'blip' in _BLIP_CACHE:
        return _BLIP_CACHE['blip']

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except Exception:
        raise RuntimeError('transformers.Blip* not available; install transformers>=4.30')

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model.eval()
    _BLIP_CACHE['blip'] = (model, processor, device)
    return model, processor, device


def caption_images(pil_images: List, max_length: int = 32) -> List[str]:
    """Generate captions for a list of PIL images using BLIP.

    Returns a list of strings (one per image).
    """
    if not pil_images:
        return []
    model, processor, device = get_blip()
    inputs = processor(images=pil_images, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=max_length)
    captions = [processor.decode(g, skip_special_tokens=True) for g in generated_ids]
    return captions


def caption_images(pil_images: List, model_name: str = "nlpconnect/vit-gpt2-image-captioning") -> List[str]:
    """Generate captions for a list of PIL images using a vision->text model.

    Returns a list of short captions (strings) in the same order as `pil_images`.
    This uses Hugging Face's VisionEncoderDecoderModel pipeline; model will be
    downloaded on first run.
    """
    try:
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        import torch
    except Exception:
        raise RuntimeError("transformers is required to run caption_images()")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    images = pil_images
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(pixel_values, max_length=32, num_beams=4)
        captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # simple cleanup
    captions = [c.strip() for c in captions]
    return captions
