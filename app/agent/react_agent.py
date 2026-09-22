"""
ReAct Agent — iterative search with multi-step reasoning, built on LangGraph.
"""
import re
from typing import Literal
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState, AgentStep
from app.agent.prompts import REACT_SYSTEM_PROMPT, FINAL_ANSWER_PROMPT, SHOULD_CONTINUE_PROMPT
from app.tools.search import SearchTool
from app.tools.base import Document
from app.planner.query_rewriter import QueryRewriter
from app.utils.llm import get_llm
from app.utils.config import get_config


class ReactAgent:
    """LangGraph-based ReAct agent with iterative web search and reranking.

    Set use_retrieval=False to skip vector store and reranker (e.g. for SFT
    data generation where local embedding models are not available).
    """

    def __init__(self, use_retrieval: bool = True):
        cfg = get_config()
        self.max_steps = cfg.agent_max_steps
        self.top_k = cfg.agent_top_k_retrieval
        self.llm = get_llm()
        self.search_tool = SearchTool()
        self.rewriter = QueryRewriter()
        self.vector_store = None
        self.reranker = None
        if use_retrieval:
            from app.retrieval.vector_store import VectorStore
            from app.reranker.reranker import Reranker
            self.vector_store = VectorStore()
            self.reranker = Reranker()
        self.graph = self._build_graph()

    # ─── Graph Builder ────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Construct the ReAct StateGraph."""
        workflow = StateGraph(AgentState)

        workflow.add_node("init", self._node_init)
        workflow.add_node("think", self._node_think)
        workflow.add_node("search", self._node_search)
        workflow.add_node("reflect", self._node_reflect)
        workflow.add_node("answer", self._node_answer)

        workflow.set_entry_point("init")
        workflow.add_edge("init", "think")
        workflow.add_conditional_edges(
            "think",
            self._route_after_think,
            {"search": "search", "answer": "answer", "__end__": END},
        )
        workflow.add_edge("search", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {"think": "think", "search": "search", "answer": "answer", "__end__": END},
        )
        workflow.add_edge("answer", END)

        return workflow.compile()

    # ─── Nodes ────────────────────────────────────────────────────

    def _node_init(self, state: AgentState) -> AgentState:
        """Decompose the question into sub-queries."""
        sub_queries = self.rewriter.rewrite(state.question)
        state.sub_queries = sub_queries
        state.current_query = sub_queries[0]
        return state

    def _node_think(self, state: AgentState) -> AgentState:
        """LLM thinks about what to do next."""
        step_num = state.step_count + 1

        # Build conversation with previous steps
        messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT.format(
            step_num=step_num, max_steps=self.max_steps
        )}]
        messages.append({"role": "user", "content": f"Question: {state.question}"})

        # Include history of previous steps
        for s in state.steps:
            messages.append({"role": "assistant", "content": f"THOUGHT: {s.thought}\nACTION: {s.action}"})
            messages.append({"role": "user", "content": f"OBSERVATION: {s.observation[:800]}"})

        response = self.llm.generate(messages, max_tokens=2048)

        # Parse thought and action from response
        thought, action, action_query = self._parse_react_output(response)

        step = AgentStep(step_num=step_num, thought=thought, action=action, action_query=action_query)
        state.add_step(step)

        return state

    def _node_search(self, state: AgentState) -> AgentState:
        """Execute the search action and retrieve results."""
        step = state.steps[-1]
        query = step.action_query

        if not query:
            # No query to search — use current query from planner
            query = state.current_query

        # Execute search
        result = self.search_tool.search(query)

        if result.success:
            step.documents = result.documents
            step.observation = self.search_tool.format_observation(result)
            state.all_documents.extend(result.documents)

            # Index into vector store for later retrieval
            if self.vector_store:
                self.vector_store.add(result.documents)
        else:
            step.observation = f"Search error: {result.error}"

        return state

    def _node_reflect(self, state: AgentState) -> AgentState:
        """Decide whether to continue searching or to answer."""
        # If we've hit max steps, force answer
        if state.step_count >= self.max_steps:
            return state

        # Collect observations so far
        observations = "\n".join([
            f"Step {s.step_num}: {s.observation[:300]}" for s in state.steps
        ])

        prompt = SHOULD_CONTINUE_PROMPT.format(
            question=state.question,
            step_num=state.step_count,
            max_steps=self.max_steps,
            observations=observations,
        )
        messages = [{"role": "user", "content": prompt}]
        decision = self.llm.generate(messages, max_tokens=512).strip().upper()

        # Store decision separately (not in thought — would leak into SFT data)
        state.steps[-1].reflection_decision = decision

        # If continuing, refine the query
        if "CONTINUE" in decision and state.step_count < self.max_steps:
            prev_obs = [s.observation for s in state.steps]
            refined_query = self.rewriter.rewrite_iterative(state.question, prev_obs)
            state.current_query = refined_query

        return state

    def _node_answer(self, state: AgentState) -> AgentState:
        """Synthesize the final answer with citations."""
        # Collect and rerank all documents
        all_docs = state.all_documents
        if all_docs and self.reranker:
            try:
                all_docs = self.reranker.rerank(state.question, all_docs, top_k=min(10, len(all_docs)))
            except Exception:
                pass  # Fall back to unranked

        # Build research summary
        summary_parts = []
        seen = set()
        doc_idx = 1
        for doc in all_docs:
            key = doc.url or doc.title
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            summary_parts.append(f"[{doc_idx}] {doc.title}\n{doc.content[:300]}")
            doc_idx += 1

        research_summary = "\n\n".join(summary_parts) if summary_parts else "No research results found."

        prompt = FINAL_ANSWER_PROMPT.format(
            question=state.question,
            research_summary=research_summary,
        )
        messages = [{"role": "user", "content": prompt}]
        state.final_answer = self.llm.generate(messages, max_tokens=2048)

        return state

    # ─── Routing ──────────────────────────────────────────────────

    def _route_after_think(self, state: AgentState) -> Literal["search", "answer", "__end__"]:
        """Route based on the last step's action."""
        step = state.steps[-1]
        if step.action.upper().startswith("SEARCH"):
            return "search"
        elif step.action.upper().startswith("ANSWER"):
            return "answer"
        return "__end__"

    def _route_after_reflect(self, state: AgentState) -> Literal["think", "search", "answer", "__end__"]:
        """Route based on reflection decision."""
        decision = state.steps[-1].reflection_decision
        if "ANSWER" in decision:
            return "answer"
        if state.step_count >= self.max_steps:
            return "answer"
        return "think"

    # ─── Parsing ──────────────────────────────────────────────────

    def _parse_react_output(self, text: str) -> tuple[str, str, str]:
        """Parse LLM output into (thought, action, action_query)."""
        thought = ""
        action = ""
        action_query = ""

        # Extract THOUGHT
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=\nACTION:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract ACTION
        action_match = re.search(r'ACTION:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
        if action_match:
            action_raw = action_match.group(1).strip()
            action = action_raw

            # Parse SEARCH: <query> or ANSWER: <answer>
            if action_raw.upper().startswith("SEARCH"):
                colon_idx = action_raw.find(":")
                if colon_idx >= 0:
                    action_query = action_raw[colon_idx + 1:].strip()
                    action = "SEARCH"
                else:
                    action_query = action_raw[len("SEARCH"):].strip()
                    action = "SEARCH"
            elif action_raw.upper().startswith("ANSWER"):
                action = "ANSWER"
                colon_idx = action_raw.find(":")
                if colon_idx >= 0:
                    action_query = action_raw[colon_idx + 1:].strip()

        # Fallback: if parsing failed, treat entire response as thought + search
        if not thought and not action:
            thought = text[:500]
            action = "SEARCH"
            # Use last meaningful sentence as search query
            sentences = text.strip().split("\n")
            action_query = sentences[-1].strip().lstrip("#-* ")

        return thought, action, action_query

    # ─── Public API ───────────────────────────────────────────────

    def run(self, question: str) -> AgentState:
        """Run the full ReAct loop and return the final state."""
        state = AgentState(question=question)
        if self.vector_store:
            self.vector_store.reset()
        final_state = self.graph.invoke(state)
        return final_state

    def run_stream(self, question: str):
        """Run the graph with streaming, yielding intermediate states."""
        state = AgentState(question=question)
        if self.vector_store:
            self.vector_store.reset()
        for event in self.graph.stream(state):
            yield event
