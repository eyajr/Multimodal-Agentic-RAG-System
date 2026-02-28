"""
Tool system: Base classes and concrete tool implementations
"""
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
import math
from config import groq_client, embedder, qdrant, LLM_MODEL, COLLECTION_NAME, VECTOR_SIZE, QDRANT_DISTANCE, DISTANCE_MAP
from qdrant_client.models import VectorParams


class ToolResult:
    """Encapsulates the result of a tool execution"""
    def __init__(self, success: bool, data: Any, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now()

    def __str__(self):
        if self.success:
            return f"Success: {self.data}"
        return f"Error: {self.error}"


class Tool(ABC):
    """Base class for all tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's parameter schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Define tool parameters"""
        pass


class ToolRegistry:
    """Registry for managing available tools"""
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools"""
        return [tool.get_schema() for tool in self.tools.values()]

    def get_tools_description(self) -> str:
        """Get formatted description of all tools"""
        descriptions = []
        for tool in self.tools.values():
            params = tool._get_parameters()
            param_str = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in params.items()])
            descriptions.append(f"- {tool.name}({param_str}): {tool.description}")
        return "\n".join(descriptions)


# ============================================================================
# CONCRETE TOOLS
# ============================================================================

class RAGTool(Tool):
    """Tool for retrieving relevant documents and generating answers"""
    def __init__(self, retrieve_func, generate_func):
        super().__init__(
            name="rag_search",
            description="Search the knowledge base and generate an answer based on relevant documents"
        )
        self.retrieve_func = retrieve_func
        self.generate_func = generate_func

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query or question"
            }
        }

    def execute(self, query: str) -> ToolResult:
        try:
            results = self.retrieve_func(query)
            chunks = [text for text, _ in results]
            answer = self.generate_func(query, chunks)
            
            return ToolResult(
                success=True,
                data={
                    "answer": answer,
                    "sources": len(chunks),
                    "top_score": results[0][1] if results else 0.0
                }
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and evaluate expressions"
        )

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '10 * 5')"
            }
        }

    def execute(self, expression: str) -> ToolResult:
        try:
            # Safe evaluation of mathematical expressions
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow,
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                'tan': math.tan, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e
            }
            
            # Clean and evaluate
            clean_expr = expression.replace('^', '**')
            result = eval(clean_expr, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(
                success=True,
                data={"result": result, "expression": expression}
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Calculation error: {str(e)}")


class ValidatorTool(Tool):
    """Tool for validating and fact-checking information"""
    def __init__(self, retrieve_func):
        super().__init__(
            name="validator",
            description="Validate claims or facts by checking against the knowledge base"
        )
        self.retrieve_func = retrieve_func

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "claim": {
                "type": "string",
                "description": "The claim or fact to validate"
            }
        }

    def execute(self, claim: str) -> ToolResult:
        try:
            # Retrieve relevant context
            results = self.retrieve_func(claim)
            chunks = [text for text, _ in results[:3]]
            
            # Use LLM to validate
            context = "\n\n".join(chunks)
            prompt = f"""Based on the following context, validate this claim:

Claim: {claim}

Context:
{context}

Respond with:
1. VERDICT: [SUPPORTED/CONTRADICTED/UNCERTAIN]
2. CONFIDENCE: [0-100]%
3. EXPLANATION: Brief explanation

Be concise and objective."""

            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            validation = response.choices[0].message.content
            
            return ToolResult(
                success=True,
                data={
                    "claim": claim,
                    "validation": validation,
                    "sources_checked": len(chunks)
                }
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SearchTool(Tool):
    """Tool for advanced semantic search in the knowledge base"""
    def __init__(self):
        super().__init__(
            name="semantic_search",
            description="Perform advanced semantic search to find specific information"
        )

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)"
            }
        }

    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        try:
            q_emb = embedder.encode(query)
            # embedder.encode may return (D,) or (1,D)
            if hasattr(q_emb, 'tolist'):
                query_vector = q_emb.tolist() if q_emb.ndim == 1 else q_emb[0].tolist()
            else:
                query_vector = list(q_emb)
            results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k
            )
            
            findings = [
                {
                    "text": hit.payload["text"][:200] + "...",
                    "score": float(hit.score),
                    "full_text": hit.payload["text"]
                }
                for hit in results.points
            ]
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": findings,
                    "count": len(findings)
                }
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DatabaseQueryTool(Tool):
    """Tool for querying structured information from the knowledge base"""
    def __init__(self, retrieve_func):
        super().__init__(
            name="database_query",
            description="Query structured information and extract specific data points"
        )
        self.retrieve_func = retrieve_func

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query_type": {
                "type": "string",
                "description": "Type of query: 'count', 'extract', 'summarize'"
            },
            "target": {
                "type": "string",
                "description": "What to query for"
            }
        }

    def execute(self, query_type: str, target: str) -> ToolResult:
        try:
            results = self.retrieve_func(target)
            chunks = [text for text, _ in results[:5]]
            context = "\n\n".join(chunks)
            
            if query_type == "count":
                prompt = f"Count or quantify: {target}\nContext:\n{context}\nProvide just the number and unit."
            elif query_type == "extract":
                prompt = f"Extract specific information: {target}\nContext:\n{context}\nProvide just the extracted data."
            else:  # summarize
                prompt = f"Summarize information about: {target}\nContext:\n{context}\nBe concise."
            
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            return ToolResult(
                success=True,
                data={
                    "query_type": query_type,
                    "target": target,
                    "result": response.choices[0].message.content
                }
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

# ============================================================================
# MULTIMODAL TOOL
# ============================================================================

from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
from clip_utils import caption_images
import os

class MultimodalSearchTool(Tool):
    """Multimodal search using CLIP (shared image-text embedding space)"""

    def __init__(self):
        super().__init__(
            name="multimodal_search",
            description="Retrieve text and image results using CLIP cross-modal embeddings."
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Ensure the Qdrant collection exists (idempotent)
        try:
            existing = [c.name for c in qdrant.get_collections().collections]
            if COLLECTION_NAME not in existing:
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=int(VECTOR_SIZE), distance=DISTANCE_MAP[QDRANT_DISTANCE])
                )
        except Exception:
            # ignore creation errors here; upsert/query will surface issues
            pass

    def _get_parameters(self):
        return {
            "query": {"type": "string", "description": "Text query"},
            "with_vectors": {"type": "boolean", "description": "Whether to return stored vectors"}
        }
    def execute(self, query: str, with_vectors: bool = False) -> ToolResult:
        try:
            inputs = self.processor(text=[query], return_tensors="pt").to(self.device)

            with torch.no_grad():
                text_out = self.model.get_text_features(**inputs)
                if isinstance(text_out, torch.Tensor):
                    text_features = text_out
                else:
                    text_features = getattr(text_out, 'pooler_output', None)
                    if text_features is None and hasattr(text_out, 'last_hidden_state'):
                        text_features = text_out.last_hidden_state.mean(dim=1)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=text_features[0].cpu().numpy().tolist(),
                limit=10,
                with_vectors=with_vectors
            )

            text_results = []
            image_results = []

            for hit in results.points:
                payload = hit.payload
                score = float(hit.score)

                if payload.get("modality") == "text":
                    text_results.append({
                        "content": payload.get("content"),
                        "score": score,
                        "payload": payload
                    })

                elif payload.get("modality") == "image":
                    vec = (hit.vector if with_vectors and hasattr(hit, 'vector') else None)
                    # normalize returned vector to unit length if present
                    if vec is not None:
                        try:
                            arr = np.asarray(vec, dtype=np.float32)
                            n = np.linalg.norm(arr)
                            if n > 0:
                                arr = arr / n
                            vec = arr.tolist()
                        except Exception:
                            pass

                    image_results.append({
                        "content": payload.get("content"),
                        "score": score,
                        "vector": vec,
                        "payload": payload
                    })

                    # If caption missing, try to generate one on-the-fly (best-effort)
                    try:
                        if isinstance(payload, dict) and not payload.get('caption'):
                            img_path = payload.get('content')
                            if img_path and os.path.exists(img_path):
                                pil = Image.open(img_path).convert('RGB')
                                cap = caption_images([pil])[0]
                                if cap:
                                    image_results[-1]['payload']['caption'] = cap
                    except Exception:
                        pass

            return ToolResult(
                success=True,
                data={
                    "text_results": text_results,
                    "image_results": image_results
                }
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ReturnImageTool(Tool):
    """Return an image path or the top image result for a query.

    Parameters:
      - path: explicit filesystem path to an image to return
      - query: a text query to find a top image via MultimodalSearchTool
    """

    def __init__(self):
        super().__init__(
            name="return_image",
            description="Return an image path either directly or by searching for a query"
        )

    def _get_parameters(self):
        return {
            "path": {"type": "string", "description": "Filesystem path to image (optional)"},
            "query": {"type": "string", "description": "Text query to find an image (optional)"}
        }

    def execute(self, path: Optional[str] = None, query: Optional[str] = None) -> ToolResult:
        try:
            # If explicit path provided, validate existence
            if path:
                try:
                    with open(path, 'rb'):
                        pass
                except Exception:
                    return ToolResult(success=False, data=None, error=f"Image path not found: {path}")
                return ToolResult(success=True, data={"type": "image", "path": path})

            # If query provided, use MultimodalSearchTool to find top image
            if query:
                mm = MultimodalSearchTool()
                res = mm.execute(query, with_vectors=False)
                if not res.success:
                    return ToolResult(success=False, data=None, error=res.error)
                imgs = res.data.get('image_results', [])
                if not imgs:
                    return ToolResult(success=False, data=None, error="No images found for query")
                top = imgs[0]
                return ToolResult(success=True, data={"type": "image", "path": top.get('content')})

            return ToolResult(success=False, data=None, error="Either 'path' or 'query' must be provided")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))