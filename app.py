import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image

# Backend imports (best-effort)
try:
    from rag import ingest_pdf, ingest_images, retrieve_with_scores
except Exception:
    ingest_pdf = ingest_images = retrieve_with_scores = None

try:
    from multimodal_llm import MultimodalLLM
except Exception:
    MultimodalLLM = None

try:
    from image_generation import generate_image_from_embedding
except Exception:
    generate_image_from_embedding = None

# Page config
st.set_page_config(page_title="Multimodal Agentic RAG", page_icon="🧠", layout="wide")
st.title("🧠 Multimodal Agentic RAG System")
st.markdown("Enhancing Retrieval-Augmented Generation with Image & Text Understanding")
st.divider()


def run_command(cmd: list) -> str:
    """Run a subprocess command and return combined stdout/stderr. (synchronous)"""
    try:
        result = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, check=False)
        out = result.stdout or ""
        err = result.stderr or ""
        return out + ("\nSTDERR:\n" + err if err else "")
    except Exception as e:
        return f"Failed to run command: {e}"

# background runner (for non-blocking long tasks)
try:
    from scripts.background_runner import runner as bg_runner
except Exception:
    bg_runner = None


def save_uploaded_file(uploaded, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, uploaded.name)
    with open(path, 'wb') as f:
        f.write(uploaded.getbuffer())
    return path


def show_pipeline_description(title: str, steps: list):
    st.markdown(f"**{title} — Pipeline**")
    for i, s in enumerate(steps, 1):
        st.markdown(f"- **Step {i}:** {s}")


# initialize simple system placeholders if real functions missing
@st.cache_resource(show_spinner=False)
def get_system():
    # Attempt to initialize a full system. Prefer `rag.initialize_system`
    # (which wires up tools, memory and the orchestrator). If that
    # isn't available, fall back to `orchestrator.initialize_system`.
    try:
        from rag import initialize_system as initialize_system_from_rag
        return initialize_system_from_rag()
    except Exception:
        try:
            from orchestrator import initialize_system as initialize_system_from_orch
            return initialize_system_from_orch()
        except Exception:
            # fallback simple stub
            class StubMemory:
                def add_message(self, role, text):
                    pass
                def get_context(self):
                    return ""
                def get_thoughts_summary(self):
                    return ""

            class StubOrch:
                def route_task(self, q, preferred_agent=None):
                    return {"answer": "Stub response: orchestrator not available."}
                def collaborative_process(self, q):
                    return {"answer": "Stub collaborative response."}

            return StubOrch(), StubMemory()


orchestrator, memory = get_system()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Document Ingestion", "🖼 Image Ingestion", "💬 Ask a Question"])

# ---------------------
# TAB 1: DOCUMENT INGESTION
# ---------------------
with tab1:
    st.subheader("Upload PDF Documents")
    show_pipeline_description("Document Ingestion", [
        "User uploads PDF.",
        "PDF is parsed into text chunks and embedded via text encoder.",
        "Chunks and vectors are stored in the vector DB (Qdrant).",
        "Retrieval uses embedding similarity during queries."
    ])

    uploaded_files = st.file_uploader("Drag & drop PDF files here", type=["pdf"], accept_multiple_files=True, key='pdf_uploads')
    if uploaded_files:
        doc_dir = os.path.join(os.path.dirname(__file__), "docs")
        os.makedirs(doc_dir, exist_ok=True)
        for f in uploaded_imgs:
                upload_paths.append(save_uploaded_file(f, up_dir))

        with st.spinner("Processing..."):
            # Multimodal direct path
            if enable_mm and upload_paths and MultimodalLLM is not None:
                try:
                    results = []
                    if retrieve_with_scores is not None:
                        results = retrieve_with_scores(query)
                        text_chunks = [t for t, _ in results]
                    else:
                        text_chunks = []

                    mm = MultimodalLLM()
                    answer_text = mm.generate_with_images(query, text_chunks, image_paths=upload_paths)
                    result = {'final_answer': answer_text, 'image_results': [{'content': p, 'score': 0.0} for p in upload_paths]}
                except Exception as e:
                    result = {'final_answer': f'Multimodal call failed: {e}'}
            else:
                # route to orchestrator
                if current_mode == "collaborative":
                    result = orchestrator.collaborative_process(query)
                elif current_mode == "auto":
                    result = orchestrator.route_task(query)
                else:
                    result = orchestrator.route_task(query, preferred_agent=current_mode)

        answer = result.get('final_answer', result.get('answer', 'No answer generated'))
        st.markdown("### 📌 Final Answer")
        st.write(answer)

        # show image_results
        image_results = None
        def find_image_results(obj):
            if isinstance(obj, dict):
                if "image_results" in obj and isinstance(obj["image_results"], list):
                    return obj["image_results"]
                for v in obj.values():
                    found = find_image_results(v)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = find_image_results(item)
                    if found:
                        return found
            return None

        image_results = find_image_results(result)
        if image_results:
            st.markdown("### 🖼 Retrieved Images")
            cols = st.columns(3)
            for idx, img in enumerate(image_results):
                with cols[idx % 3]:
                    st.image(img["content"]) if os.path.exists(img["content"]) else st.image(img.get("content"))
                    st.caption(f"Score: {img.get('score', 0.0):.4f}")

            if len(image_results) > 0 and st.button("Generate Image Variation from top image"):
                emb = image_results[0].get('vector')
                if emb is not None and generate_image_from_embedding is not None:
                    x_gen = generate_image_from_embedding(emb)
                    # x_gen: torch tensor (C,H,W)
                    import numpy as _np
                    arr = x_gen.permute(1,2,0).cpu().numpy()
                    st.image(arr, caption="Generated Variation")
                else:
                    st.warning("Embedding or generator not available.")

        with st.expander("🔍 Full Agent Output"):
            st.json(result)

# ---------------------
# TAB 4: ADMIN & DEMOS
# ---------------------
with tab4:
    st.subheader("Admin / Demos — Run training and generation scripts from the UI")

    # Adapter training UI removed
    st.markdown("---")
    st.markdown("**Stable Diffusion + CLIP Guidance (Refinement)**")
    sd_prompt = st.text_input("SD Prompt", "A photorealistic portrait of a fox")
    target_text = st.text_input("Target text (optional)")
    target_image = st.file_uploader("Target image (optional)", type=["png","jpg","jpeg"], key='target_img')
    steps = st.number_input("Refinement steps", min_value=1, max_value=1000, value=80)
    sdlr = st.number_input("Refinement LR", value=0.1)
    out_img = st.text_input("Output image path", "outputs/refined.png")

    if st.button("Run SD+CLIP Guidance"):
        cmd = ["scripts/sd_clip_guidance.py", "--prompt", sd_prompt, "--steps", str(int(steps)), "--lr", str(sdlr), "--output", out_img]
        if target_text:
            cmd += ["--target_text", target_text]
        if target_image:
            tmpd = tempfile.mkdtemp()
            tpath = save_uploaded_file(target_image, tmpd)
            cmd += ["--target_image", tpath]

        if bg_runner is not None:
            task_id = bg_runner.run_async(cmd)
            log_area = st.empty()
            log_area.text(f"SD+CLIP task {task_id} started — streaming logs below")
            done = False
            while not done:
                logs, done = bg_runner.get_logs(task_id)
                log_area.text(logs)
                if not done:
                    import time
                    time.sleep(0.5)
            st.success("SD+CLIP guidance finished")
            if os.path.exists(out_img):
                st.image(out_img, caption="Refined image")
            else:
                st.warning("Output image not produced; check log.")
        else:
            st.text("Running synchronously: " + " ".join(cmd))
            out = run_command(cmd)
            st.text_area("SD Guidance log", value=out, height=200)
            if os.path.exists(out_img):
                st.image(out_img, caption="Refined image")
            else:
                st.warning("Output image not produced; check log.")

    st.markdown("---")
    st.markdown("**Pipelines & Rationale**")
    st.markdown("This dashboard shows how backend building blocks map to UI actions.\n- Ingestion uses CLIP/text encoders to produce vectors stored in Qdrant.\n- Retrieval queries Qdrant using CLIP text/image vectors.\n- Adapters map CLIP vectors into LLM token-space for conditioning.\n- Fusion adapters provide cross-attention-based mid-level fusion.\n- SD+CLIP guidance refines generated images towards a CLIP target embedding.")

    st.markdown("---")
    st.markdown("**System Info**")
    st.text(memory.get_context())

 






