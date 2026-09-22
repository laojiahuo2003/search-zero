"""
Agent state definition for the ReAct loop.
"""
from dataclasses import dataclass, field
from app.tools.base import Document


@dataclass
class AgentStep:
    """A single ReAct step: Thought -> Action -> Observation."""
    step_num: int
    thought: str = ""
    action: str = ""
    action_query: str = ""
    observation: str = ""
    reflection_decision: str = ""
    documents: list[Document] = field(default_factory=list)


@dataclass
class AgentState:
    """Full agent state carried through the ReAct graph."""
    question: str = ""
    sub_queries: list[str] = field(default_factory=list)
    current_query: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    all_documents: list[Document] = field(default_factory=list)
    final_answer: str = ""
    step_count: int = 0
    error: str = ""

    def add_step(self, step: AgentStep):
        self.steps.append(step)
        self.step_count = len(self.steps)

    def to_trace(self) -> list[dict]:
        """Serialize steps for frontend visualization."""
        trace = []
        for s in self.steps:
            trace.append({
                "step": s.step_num,
                "thought": s.thought,
                "action": s.action,
                "action_query": s.action_query,
                "observation": s.observation[:500],
                "documents": [d.to_dict() for d in s.documents],
            })
        return trace
