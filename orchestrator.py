"""
Agent Orchestrator: Routes tasks and coordinates multiple agents
"""
from typing import Dict, Any, Optional

from agents import ReActAgent, ReflectionAgent, PlannerAgent
from tools import ToolRegistry
from memory import Memory


class AgentOrchestrator:
    """Orchestrates multiple agents and routes tasks"""
    def __init__(self, tool_registry: ToolRegistry, memory: Memory):
        self.tool_registry = tool_registry
        self.memory = memory
        
        # Initialize agents
        self.react_agent = ReActAgent(tool_registry, memory)
        self.reflection_agent = ReflectionAgent(tool_registry, memory)
        self.planner_agent = PlannerAgent(tool_registry, memory)
        
        self.agents = {
            "react": self.react_agent,
            "reflection": self.reflection_agent,
            "planner": self.planner_agent
        }

    def route_task(self, task: str, preferred_agent: Optional[str] = None) -> Dict[str, Any]:
        """Route task to the most appropriate agent"""

        task_lower = task.lower()

        # ------------------------------------------------------------------
        # 1- HARD ROUTING FOR VISUAL / MULTIMODAL TASKS
        # ------------------------------------------------------------------
        visual_keywords = [
            "diagram",
            "image",
            "architecture",
            "visual",
            "show",
            "picture",
            "figure"
        ]

        if any(word in task_lower for word in visual_keywords):
            print("\n Visual query detected → Using MultimodalSearchTool directly")

            multimodal_tool = self.tool_registry.get_tool("multimodal_search")

            if multimodal_tool:
                # Request vectors so UI can optionally generate variations
                result = multimodal_tool.execute(query=task, with_vectors=True)

                return {
                    "mode": "multimodal_direct",
                    "task": task,
                    "image_results": result.data.get("image_results"),
                    "text_results": result.data.get("text_results"),
                    "final_answer": "Retrieved relevant visual results."
                }

        # ------------------------------------------------------------------
        # 2- PREFERRED AGENT
        # ------------------------------------------------------------------
        if preferred_agent and preferred_agent in self.agents:
            agent = self.agents[preferred_agent]
            print(f"\n Using preferred agent: {agent.name}")
            return agent.process(task)

        # ------------------------------------------------------------------
        # 3- AUTO ROUTING
        # ------------------------------------------------------------------
        if any(word in task_lower for word in ["plan", "steps", "first", "then", "multiple", "complex"]):
            print("\n Auto-routing to PlannerAgent")
            return self.planner_agent.process(task)

        elif any(word in task_lower for word in ["validate", "check", "verify", "correct", "review", "improve"]):
            print("\n Auto-routing to ReflectionAgent")
            return self.reflection_agent.process(task)

        else:
            print("\n Auto-routing to ReActAgent")
            return self.react_agent.process(task)

    def collaborative_process(self, task: str) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(" COLLABORATIVE MODE")
        print(f"{'='*70}\n")

        task_lower = task.lower()

        visual_keywords = [
            "diagram",
            "image",
            "architecture",
            "visual",
            "show",
            "picture",
            "figure"
        ]

        # -------------------------------------------------------
        # 1️- VISUAL SHORT-CIRCUIT
        # -------------------------------------------------------
        if any(word in task_lower for word in visual_keywords):
            print(" Visual query detected → Direct multimodal retrieval")

            multimodal_tool = self.tool_registry.get_tool("multimodal_search")

            if multimodal_tool:
                result = multimodal_tool.execute(query=task, with_vectors=True)

                return {
                    "mode": "collaborative_multimodal_direct",
                    "task": task,
                    "image_results": result.data.get("image_results"),
                    "text_results": result.data.get("text_results"),
                    "final_answer": "Here are the retrieved architecture diagram(s)."
                }

        # -------------------------------------------------------
        # 2️- OTHERWISE FULL COLLABORATION
        # -------------------------------------------------------

        print("Step 1: Planning...")
        plan_result = self.planner_agent.process(task)

        print("\nStep 2: Executing with reasoning...")
        react_result = self.react_agent.process(task)

        print("\nStep 3: Reflecting...")
        reflection_result = self.reflection_agent.process(
            task,
            initial_answer=react_result['answer']
        )

        return {
            "mode": "collaborative",
            "task": task,
            "plan": plan_result,
            "execution": react_result,
            "reflection": reflection_result,
            "final_answer": reflection_result['final_answer']
        }
