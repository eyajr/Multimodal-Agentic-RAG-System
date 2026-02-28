"""
Agent implementations: Base Agent, ReAct, Reflection, and Planner

This module defines:
- Agent (base abstract class)
- ReActAgent (Thought -> Action -> Observation loop)
- ReflectionAgent (Generate -> Critique -> Improve loop)
- PlannerAgent (Plan -> Execute subtasks -> Synthesize)

Multimodal note (NO OCR):
- If a tool returns {"image_results": [...]}, agents propagate them in their returned dict
  so the Streamlit UI can display the retrieved images.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from config import groq_client, LLM_MODEL
from tools import ToolRegistry, ToolResult
from memory import Memory


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    DONE = "done"


# -------------------------
# Base Agent
# -------------------------

class Agent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, role: str, tool_registry: ToolRegistry, memory: Memory):
        self.name = name
        self.role = role
        self.tool_registry = tool_registry
        self.memory = memory
        self.state = AgentState.IDLE
        self.max_iterations = 5

    @abstractmethod
    def process(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """Process a task and return results"""
        raise NotImplementedError

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Call the LLM with a prompt"""
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        if "Final Answer:" not in text:
            return None
        return text.split("Final Answer:", 1)[-1].strip()

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from LLM output.

        Supported formats:
          - Action: tool_name(param1="value1", param2="value2")
          - Action: tool_name()
        """
        pattern = r"Action:\s*(\w+)\((.*?)\)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1).strip()
        params_str = (match.group(2) or "").strip()

        params: Dict[str, Any] = {}
        if params_str:
            # naive param parsing: key="value", key='value', key=value (no nested commas)
            for param in params_str.split(","):
                if "=" not in param:
                    continue
                key, value = param.split("=", 1)
                key = key.strip()
                raw = value.strip()
                # strip surrounding quotes if present
                if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                    val = raw[1:-1]
                else:
                    val = raw

                # try to coerce booleans, ints, floats
                low = val.lower() if isinstance(val, str) else val
                if isinstance(val, str) and low in ("true", "false"):
                    coerced = True if low == "true" else False
                else:
                    coerced = None
                    # int
                    try:
                        coerced = int(val)
                    except Exception:
                        try:
                            coerced = float(val)
                        except Exception:
                            coerced = val

                params[key] = coerced

        return {"tool": tool_name, "params": params}

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name"""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool '{tool_name}' not found")

        try:
            return tool.execute(**params)
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Tool execution failed: {str(e)}")


# ============================================================================
# REACT AGENT
# ============================================================================

class ReActAgent(Agent):
    """ReAct agent: Reasoning + Acting in an interleaved manner"""

    def __init__(self, tool_registry: ToolRegistry, memory: Memory):
        super().__init__(
            name="ReActAgent",
            role="Reasoning and acting agent that thinks step-by-step",
            tool_registry=tool_registry,
            memory=memory,
        )

    def process(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """Process task using ReAct loop: Thought -> Action -> Observation"""
        print(f"\n{'=' * 70}")
        print(f" {self.name} processing: {task}")
        print(f"{'=' * 70}\n")

        tools_desc = self.tool_registry.get_tools_description()
        context = self.memory.get_context()

        trajectory: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        collected_images: List[Dict[str, Any]] = []

        # If the task is clearly visual, bypass the LLM and run multimodal search directly.
        visual_keywords = ["show", "diagram", "architecture", "image", "display", "figure"]
        if any(k in task.lower() for k in visual_keywords):
            tool = self.tool_registry.get_tool("multimodal_search")
            if tool:
                # Request vectors so UI can generate variations if needed
                result = tool.execute(query=task, with_vectors=True)
                if result.success and isinstance(result.data, dict):
                    image_results = result.data.get("image_results", [])
                    if image_results:
                        return {
                            "agent": self.name,
                            "task": task,
                            "answer": "Retrieved images from the knowledge base:",
                            "image_results": image_results,
                            "iterations": 1,
                        }

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # THOUGHT PHASE
            self.state = AgentState.THINKING
            thought_prompt = f"""You are a reasoning agent solving tasks step by step.

Previous context:
{context}

Current task: {task}

Trajectory so far:
{self._format_trajectory(trajectory)}

Available tools:
{tools_desc}

Important:
- If you need an image/diagram stored in the knowledge base, use the multimodal search tool.
- If a tool returns image results, I will show them in the UI automatically.

If you have enough information to answer, start with "Final Answer:"
Otherwise, specify the next step as:
Action: tool_name(param1="value1", param2="value2")

Your response:"""

            thought_response = self._call_llm(thought_prompt, temperature=0.7)
            print(f"\n Thought:\n{thought_response}\n")
            self.memory.add_thought(self.name, thought_response)

            # Check for final answer
            extracted = self._extract_final_answer(thought_response)
            if extracted is not None:
                final_answer = extracted
                trajectory.append({"type": "answer", "content": final_answer})
                break

            # ACTION PHASE
            self.state = AgentState.ACTING
            tool_call = self._parse_tool_call(thought_response)

            # Diagnostic: show detected tool call structure
            if tool_call:
                print("TOOL CALL DETECTED:", tool_call)

            if not tool_call:
                # No valid action detected - return the model response as final answer
                final_answer = thought_response.strip()
                trajectory.append({"type": "answer", "content": final_answer})
                break

            print(f" Action: {tool_call['tool']}({tool_call['params']})")
            result = self._execute_tool(tool_call["tool"], tool_call["params"])

            # OBSERVATION PHASE
            print(f" Observation: {result}\n")

            trajectory.append(
                {
                    "type": "action",
                    "thought": thought_response,
                    "action": tool_call,
                    "observation": str(result),
                }
            )
            context += f"\nAction: {tool_call}\nResult: {result}"

            # =========================
            # HANDLE MULTIMODAL OUTPUT
            # =========================
            if result.success and isinstance(result.data, dict):

                # Case 1: Image retrieval
                image_results = result.data.get("image_results")
                if isinstance(image_results, list) and image_results:
                    return {
                        "agent": self.name,
                        "task": task,
                        "answer": "Retrieved relevant images:",
                        "image_results": image_results,
                        "trajectory": trajectory,
                        "iterations": iteration + 1,
                    }

                # Case 2: Text answer from tool
                if result.data.get("answer") and isinstance(result.data.get("answer"), str):
                    final_answer = result.data.get("answer").strip()
                    trajectory.append({"type": "answer", "content": final_answer})
                    break

            # Otherwise continue loop

        self.state = AgentState.DONE

        # If images were collected but no final answer, still return images
        if collected_images and not final_answer:
            return {
                "agent": self.name,
                "task": task,
                "answer": "Retrieved images from the knowledge base:",
                "image_results": collected_images,
                "trajectory": trajectory,
                "iterations": len(trajectory),
            }

        if not final_answer:
            last_obs = trajectory[-1].get("observation", "") if trajectory else ""
            if "answer" in last_obs.lower():
                final_answer = last_obs
            else:
                final_answer = (
                    "I gathered some information, but I couldn't produce a clear final answer with the available tools."
                )

        return {
            "agent": self.name,
            "task": task,
            "answer": final_answer,
            "trajectory": trajectory,
            "iterations": len(trajectory),
        }

    def _format_trajectory(self, trajectory: List[Dict[str, Any]]) -> str:
        """Format trajectory for prompt"""
        if not trajectory:
            return "No actions taken yet."

        formatted: List[str] = []
        for i, step in enumerate(trajectory, 1):
            if step.get("type") == "action":
                formatted.append(f"Step {i}:")
                formatted.append(f"  Thought: {str(step.get('thought', ''))[:100]}...")
                formatted.append(f"  Action: {step.get('action')}")
                formatted.append(f"  Result: {str(step.get('observation', ''))[:150]}...")

        return "\n".join(formatted) if formatted else "No actions yet."


# ============================================================================
# REFLECTION AGENT
# ============================================================================

class ReflectionAgent(Agent):
    """Agent that reflects on and improves responses"""

    def __init__(self, tool_registry: ToolRegistry, memory: Memory):
        super().__init__(
            name="ReflectionAgent",
            role="Self-critique and improvement agent",
            tool_registry=tool_registry,
            memory=memory,
        )
        self.max_reflections = 2

    def process(self, task: str, initial_answer: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Process with reflection: Generate -> Critique -> Improve"""
        print(f"\n{'=' * 70}")
        print(f" {self.name} processing: {task}")
        print(f"{'=' * 70}\n")

        # If no initial answer, generate one using RAG
        if not initial_answer:
            rag_tool = self.tool_registry.get_tool("rag_search")
            if rag_tool:
                result = rag_tool.execute(query=task)
                if result.success and isinstance(result.data, dict):
                    initial_answer = result.data.get("answer", "") or ""
                else:
                    initial_answer = "No answer generated."
            else:
                initial_answer = "Unable to generate initial answer."

        current_answer = initial_answer
        reflections: List[Dict[str, Any]] = []

        for reflection_round in range(self.max_reflections):
            print(f"\n--- Reflection Round {reflection_round + 1}/{self.max_reflections} ---")

            # CRITIQUE PHASE
            self.state = AgentState.REFLECTING
            critique_prompt = f"""You are a critical reviewer. Analyze this answer for accuracy, completeness, and clarity.

Task: {task}

Current Answer:
{current_answer}

Provide constructive critique:
1. What is correct?
2. What is missing or unclear?
3. What could be improved?
4. Rate quality (1-10):

Your critique:"""

            critique = self._call_llm(critique_prompt, temperature=0.3)
            print(f"\n Critique:\n{critique}\n")

            reflections.append({"round": reflection_round + 1, "critique": critique})

            # Stop early if quality is high
            if any(x in critique.lower() for x in ["9/10", "10/10", "excellent", "outstanding"]) or (
                re.search(r"\b(9|10)\b", critique) is not None
            ):
                print("✓ Answer quality sufficient, stopping reflection.")
                break

            # IMPROVEMENT PHASE
            improve_prompt = f"""Based on the critique, improve the answer.

Task: {task}

Previous Answer:
{current_answer}

Critique:
{critique}

Generate an improved answer that addresses the critique:"""

            improved_answer = self._call_llm(improve_prompt, temperature=0.5)
            print(f"\n Improved Answer:\n{improved_answer}\n")

            reflections[-1]["improved_answer"] = improved_answer
            current_answer = improved_answer

            self.memory.add_thought(self.name, f"Reflection {reflection_round + 1}: {critique[:100]}...")

        self.state = AgentState.DONE

        return {
            "agent": self.name,
            "task": task,
            "initial_answer": initial_answer,
            "final_answer": current_answer,
            "reflections": reflections,
            "improvement_rounds": len(reflections),
        }


# ============================================================================
# PLANNER AGENT
# ============================================================================

class PlannerAgent(Agent):
    """Agent that decomposes complex tasks into subtasks"""

    def __init__(self, tool_registry: ToolRegistry, memory: Memory):
        super().__init__(
            name="PlannerAgent",
            role="Task decomposition and planning agent",
            tool_registry=tool_registry,
            memory=memory,
        )

    def process(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """Process by planning and executing subtasks"""
        print(f"\n{'=' * 70}")
        print(f" {self.name} processing: {task}")
        print(f"{'=' * 70}\n")

        tools_desc = self.tool_registry.get_tools_description()

        # PLANNING PHASE
        self.state = AgentState.THINKING
        plan_prompt = f"""You are a task planning agent. Break down complex tasks into subtasks.

Main Task: {task}

Available tools:
{tools_desc}

Rules:
- Use Tool: <tool_name> for steps that need a tool.
- Use Tool: None when no tool is needed.

Format:
Plan:
- Step 1: [description] -> Tool: [tool_name or None]
- Step 2: [description] -> Tool: [tool_name or None]
...

Your plan:"""

        plan = self._call_llm(plan_prompt, temperature=0.5)
        print(f"\n Plan:\n{plan}\n")

        self.memory.add_thought(self.name, f"Created plan: {plan[:150]}...")

        steps = self._parse_plan(plan)

        # EXECUTION PHASE
        self.state = AgentState.ACTING
        results: List[Dict[str, Any]] = []
        collected_images: List[Dict[str, Any]] = []

        for i, step in enumerate(steps, 1):
            print(f"\n--- Executing Step {i}/{len(steps)} ---")
            print(f"Task: {step['description']}")

            tool_name = step.get("tool")
            if tool_name:
                params = self._prepare_tool_params(tool_name, step["description"], task)
                print(f"Using tool: {tool_name}")
                result = self._execute_tool(tool_name, params)
                print(f"Result: {result}\n")

                success = bool(result.success)
                result_data = result.data

                if success and isinstance(result_data, dict):
                    imgs = result_data.get("image_results")
                    if isinstance(imgs, list) and imgs:
                        collected_images.extend(imgs)

                results.append(
                    {
                        "step": i,
                        "description": step["description"],
                        "tool": tool_name,
                        "result": str(result),
                        "success": success,
                    }
                )
            else:
                results.append(
                    {
                        "step": i,
                        "description": step["description"],
                        "tool": None,
                        "result": "No tool execution needed",
                        "success": True,
                    }
                )

        # SYNTHESIS PHASE (grounded)
        evidence_lines: List[str] = []
        for r in results:
            if r["success"]:
                evidence_lines.append(f"Step {r['step']} ({r.get('tool') or 'no_tool'}): {r['result']}")

        evidence_text = "\n\n".join(evidence_lines)[:6000]

        synthesis_prompt = f"""You are a grounded reasoning agent.

Answer using ONLY the retrieved evidence below.
If information is missing, say so clearly.

Original Task:
{task}

Retrieved Evidence:
{evidence_text}

Now provide the final grounded answer (no fabrication):"""

        final_answer = self._call_llm(synthesis_prompt, temperature=0.2, max_tokens=700)

        self.state = AgentState.DONE

        response: Dict[str, Any] = {
            "agent": self.name,
            "task": task,
            "plan": plan,
            "steps": steps,
            "results": results,
            "final_answer": final_answer,
        }
        if collected_images:
            response["image_results"] = collected_images
        return response

    def _parse_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        """Parse plan into structured steps"""
        steps: List[Dict[str, Any]] = []
        lines = plan_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("-") or line.lower().startswith("step"):
                tool_match = re.search(r"Tool:\s*([A-Za-z_]\w*)", line, re.IGNORECASE)
                tool = tool_match.group(1) if tool_match else None

                if tool and tool.lower() in {"none", "null"}:
                    tool = None

                description = re.sub(r"^-\s*", "", line)
                description = re.sub(r"^Step\s*\d+\s*:\s*", "", description, flags=re.IGNORECASE)
                description = re.sub(r"->\s*Tool:.*$", "", description, flags=re.IGNORECASE).strip()

                if description:
                    steps.append({"description": description, "tool": tool})

        return steps if steps else [{"description": plan_text.strip(), "tool": None}]

    def _prepare_tool_params(self, tool_name: str, description: str, original_task: str) -> Dict[str, Any]:
        """Prepare parameters for tool execution"""
        if tool_name in {"rag_search", "semantic_search", "multimodal_search"}:
            return {"query": description}
        if tool_name == "calculator":
            return {"expression": description}
        if tool_name == "validator":
            return {"claim": description}
        if tool_name == "database_query":
            return {"query_type": "extract", "target": description}
        return {"query": description}