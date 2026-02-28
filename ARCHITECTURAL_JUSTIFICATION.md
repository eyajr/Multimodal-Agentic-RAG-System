## Architectural justification — Multimodal RAG Enhancements

This document explains why the multimodal additions were designed the way they are and how they connect to core concepts from class: alignment, fusion strategies, latent representations, and conditioning.

1) Alignment: shared embedding space
- Rationale: We use a CLIP family encoder so text and image inputs map into a common latent space. This enables cross-modal retrieval directly via nearest-neighbor search, avoiding brittle hand-designed bridges.
- Code: see CLIP loading and encoders in [clip_utils.py](clip_utils.py#L15-L60) and the use of CLIP for both image and text embedding in [rag.py](rag.py#L61-L135) and [rag.py](rag.py#L189-L227).
- Effect: Text queries can retrieve image vectors and vice-versa; similarity scores are meaningful because vectors are L2-normalized before storage.

2) Latent representations: normalized CLIP vectors and DB storage
- Rationale: Storing normalized fixed-size vectors simplifies similarity computation and lets the vector DB (Qdrant) handle nearest-neighbor retrieval efficiently.
- Code: ingestion writes vectors and payloads in `ingest_images` and `ingest_pdf` ([rag.py](rag.py#L61-L135), [rag.py](rag.py#L189-L227)).
- Effect: Downstream components (retrieval, prompt construction) can treat vectors uniformly regardless of modality and request nearby text/image chunks from the same collection.

3) Fusion strategies: soft-prompting vs. cross-attention
- Soft-prompt adapter (conditioning): We implemented `EmbedToTokenAdapter` / `SoftPromptAdapter` that projects CLIP vectors into an LLM token-embedding prefix. This is a light-weight, low-risk conditioning method allowing the LLM to receive a continuous visual hint without fine-tuning the LLM weights. See `soft_prompt_trainer.py` and `embed_to_token_adapter.py` ([soft_prompt_trainer.py](soft_prompt_trainer.py#L6-L28), [embed_to_token_adapter.py](embed_to_token_adapter.py#L8-L40)).
- Cross-attention fusion (mid-level): The repo contains scaffolding for cross-attention fusion (see `scripts/cross_attention_adapter.py`) which, if enabled, inserts learned cross-attention layers that allow deeper integration between visual and language representations inside the model's transformer layers. This is more expressive but also more invasive and computationally costly.
- Tradeoffs: Soft-prompts are simpler to train and deploy (just map a vector → token prefix), preserve LLM weights, and are robust for many conditioning tasks. Cross-attention fusion tends to yield stronger grounding and alignment when you can fine-tune internal layers but requires more engineering and compute and careful regularization.

4) Conditioning and prompt construction
- Prompt-level conditioning: When image vectors are present the pipeline constructs an augmented prompt containing:
  - Image reference list and vector summaries (short human-readable summaries derived from vectors).
  - Nearby text chunks retrieved using the image vector (image-associated text) to provide concrete context.
  - Optional soft-prompt base64 section produced by the adapter when `SOFT_PROMPT_ADAPTER_PATH` is set ([rag.py](rag.py#L317-L437), [rag.py](rag.py#L402-L425)).
- LLM-level conditioning: The `MultimodalLLM` abstraction supports sending images as attachments to an external multimodal LLM (if available) or falling back to a deterministic composition of captions + vector summaries locally ([multimodal_llm.py](multimodal_llm.py#L14-L114)).

5) Coherence: not an isolated add-on
- Design principle: All multimodal artifacts (text vectors, image vectors, captions) share the same collection and retrieval APIs. The retrieval stage can therefore produce mixed-modality context for any query. The generation stage consumes that unified context via prompt augmentation and, if available, adapter-generated soft prompts. See the end-to-end flow in [rag.py](rag.py#L317-L437).

6) Practical considerations and reproducibility
- Repro steps for graders:
  1. Start Qdrant and ingest text (`ingest_pdf`) and images (`ingest_images`).
  2. Run queries via `retrieve_with_scores` to validate cross-modal retrieval ([rag.py](rag.py#L229-L255)).
  3. Optionally set `SOFT_PROMPT_ADAPTER_PATH` to a trained adapter and run the generation path to see LLM conditioning via embedded soft prompt.
- Files to inspect: [rag.py](rag.py), [clip_utils.py](clip_utils.py), [soft_prompt_trainer.py](soft_prompt_trainer.py), [embed_to_token_adapter.py](embed_to_token_adapter.py), [multimodal_llm.py](multimodal_llm.py), [scripts/cross_attention_adapter.py](scripts/cross_attention_adapter.py).

Conclusion
- The repo implements a coherent multimodal RAG extension: CLIP-based alignment, unified vector storage, retrieval-augmented prompt construction and two plausible fusion/conditioning strategies (soft prompts and cross-attention).
