<<<<<<< HEAD
# Multimodal-Agentic-RAG-System
=======
# Agentic RAG System

Multi-agent RAG system using Qdrant, Groq LLM, and multiple specialized AI agents.

## Prerequisites

**IMPORTANT: Setup API Key First**
1. Get your Groq API key from [https://console.groq.com](https://console.groq.com)
2. Add it to [.env](.env) file:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Setup & Running

### Step 1: Start Docker Desktop
- Open Docker Desktop application
- Wait for it to fully start

### Step 2: Run Qdrant Vector Database
In VS Code terminal, run:
```powershell
docker run -p 6333:6333 qdrant/qdrant
```

#### Run Qdrant & Streamlit demo
After starting the Qdrant container, verify the service and then run the Streamlit demo (`app.py`):

- Verify Qdrant is reachable (API available at port 6333):

```powershell
# quick health check
curl http://localhost:6333/collections
# or open in your browser: http://localhost:6333
```

- (Optional) Seed the database if you need demo data:

```powershell
& ".venv\Scripts\Activate.ps1"
pip install -r requirements.txt
python scripts/seed_qdrant.py
```

- Prepare environment variables (copy `.env.example` and set Qdrant host/port if not present):

```powershell
copy .env.example .env
notepad .env   # edit and ensure QDRANT_HOST=localhost and QDRANT_PORT=6333
```

- Run the Streamlit demo (`app.py`):

```powershell
& ".venv\Scripts\Activate.ps1"   # if not already active
streamlit run app.py
```

Open the Streamlit UI at http://localhost:8501 in your browser.


### Step 3: Download PDF Document
Download the file: `a-practical-guide-to-building-agents.pdf`

Place it in: `C:\Users\eajroud\Downloads\`

Or update the path in [rag.py](rag.py) (line 224):
```python
pdf_path = r"YOUR_PATH_HERE\your-file.pdf"
```

### Step 4: Run the System
Open a new terminal and run:
```powershell
python rag.py
```

### Step 5: Use the System
1. Choose an agent mode by typing a number (1-5):
   - `1` - Auto-route (system decides)
   - `2` - ReAct Agent
   - `3` - Reflection Agent  
   - `4` - Planner Agent
   - `5` - Collaborative Mode

2. Ask your question, e.g.:
   ```
   what are agents
   ```

## File Structure

- **[config.py](config.py)** - Configuration, environment variables, and client initialization (Groq, Qdrant, embeddings)
- **[tools.py](tools.py)** - Tool implementations (RAG search, calculator, validator, semantic search, database query)
- **[memory.py](memory.py)** - Conversation memory management for storing messages and agent thoughts
- **[agents.py](agents.py)** - Agent implementations (ReAct, Reflection, Planner) with reasoning capabilities
- **[orchestrator.py](orchestrator.py)** - Routes tasks to appropriate agents and coordinates multi-agent collaboration
- **[rag.py](rag.py)** - Main entry point with PDF ingestion, vector search, and interactive CLI
- **[.env](.env)** - Environment variables (API keys, model settings, RAG configuration)

## Features

- Multiple specialized agents (ReAct, Reflection, Planner)
- RAG-based document retrieval
- Conversational memory
- Extensible tool system
- Vector similarity search with Qdrant

## Coherent Multimodal Enhancement (what we added)

### Base system (what we started with)

The project originally provided a classic Retrieval-Augmented Generation (RAG) base: multiple specialized agents (ReAct, Reflection, Planner) coordinated by an `orchestrator`, document ingestion and chunking pipelines, text embedding + vector storage in Qdrant, and LLM-based generation downstream. The core flow was: ingest text → embed text → index in the vector DB → retrieve relevant text chunks for a query → condition an LLM to produce answers.

### What we added (coherent enhancement, not an isolated add-on)

- Integrated image modality end-to-end by adding CLIP-based encoders and image ingestion utilities. Images are embedded via CLIP and stored alongside text vectors in the same Qdrant collection (`rag.py` and `clip_utils.py`).
- Unified embedding space: text is embedded with the CLIP text encoder and images with the CLIP image encoder so that text↔image retrieval is possible using the same similarity search (`rag.py`).
- Retrieval augmentation: for image-conditioned queries the system fetches image-associated text chunks from the vector DB to create richer, image-aware context for generation (`rag.py`).
- Generation conditioning: we add vector summaries and an optional soft-prompt adapter that maps CLIP vectors into LLM token-embedding prefixes (`soft_prompt_trainer.py`, `embed_to_token_adapter.py`) so the LLM can be directly conditioned on visual features rather than only textual context.
- Multimodal LLM hook: a flexible `MultimodalLLM` integration supports sending images as attachments to an external multimodal LLM or falling back to a local deterministic synthesis combining captions and vector summaries (`multimodal_llm.py`).

These components are wired into the existing ingest→index→retrieve→generate pipeline so multimodality improves retrieval and conditioning rather than being a separate utility. See `rag.py`, `clip_utils.py`, `multimodal_llm.py`, `soft_prompt_trainer.py`, and `embed_to_token_adapter.py` for the core code paths.

## Current project structure (detailed file descriptions)

- **`app.py`** - Top-level launcher / convenience entrypoint for running the system or demos.
- **`agents.py`** - Implementations of the specialized agents (ReAct, Reflection, Planner) and their reasoning loops.
- **`orchestrator.py`** - High-level coordinator that routes tasks to agents and manages multi-agent workflows.
- **`rag.py`** - Retrieval-Augmented-Generation pipeline: PDF/text ingestion, embedding, indexing (Qdrant), retrieval, and interactive CLI loop.
- **`tools.py`** - Utility tools and actions used by agents (semantic search wrapper, calculators, validators, DB helpers).
- **`memory.py`** - Conversation and agent-memory utilities for storing messages, thoughts, and context across turns.
- **`config.py`** - Configuration values, environment variable loading, and client initialization (Groq, Qdrant, embedding clients).
- **`clip_utils.py`** - CLIP-related helpers for encoding images and text and image preprocessing utilities.
- **`image_encoder.py`** - Image encoder wrapper(s) and model integration logic used for producing image embeddings.
- **`image_generation.py`** - Helpers for image generation pipelines (diffusion sampling/training utilities and wrappers).
- **`multimodal_llm.py`** - Adapter layer for talking to an external or local multimodal LLM and composing multimodal prompts.
- **`embed_to_token_adapter.py`** - Implementation that maps embedding vectors into token-prefix embeddings (soft-prompt adapter).
- **`soft_prompt_trainer.py`** - Training utilities and scripts for learning soft prompts / adapters that condition LLMs on vectors.
- **`rag.py`** - (see above) main RAG flow and interactive CLI.
- **`agents.py`**, **`orchestrator.py`** - (see above) core orchestration and agent definitions.
- **`requirements.txt`** - Pin and list Python package dependencies for the project.
- **`ARCHITECTURAL_JUSTIFICATION.md`** - Rationale and design notes explaining the system architecture and tradeoffs.

- **`scripts/`** - One-off and runnable scripts (examples: `scripts/seed_qdrant.py`, `scripts/demo_pipeline.py`, `scripts/train_adapter.py`, `scripts/eval_harness.py`, `scripts/sd_clip_guidance.py`).
- **`tests/`** - Unit and integration tests for adapters, multimodal flows, tools, and the RAG pipeline (examples: `test_adapter_and_retrieval.py`, `test_multimodal_integration.py`).
- **`docs/`** - Project documentation and how-tos (e.g., `adapter_training.md`).
- **`labs/`** - Experimental notebooks and training utilities (diffusion utility scripts and experiments).
- **`images/`** - Example images and assets used by demos and tests.
- **`uploads/`** - Runtime folder for uploaded documents/images used by demos or ingestion pipelines.

- **`.env` / `.env.example`** - Environment variable examples and runtime secret storage (API keys, endpoints).
- **`.gitignore`** - Files and directories excluded from source control.

If you'd like, I can now:

- Update these descriptions to be more specific for any file you pick, or
- Generate a machine-readable `STRUCTURE.md` or `docs/structure.md` with the same content.
