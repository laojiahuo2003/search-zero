"""
Streamlit frontend — interactive demo with reasoning trace visualization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Search-R1 Mini",
    page_icon="🔍",
    layout="wide",
)

# ─── Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
.thought-box { background: #f0f4ff; border-left: 4px solid #4a90d9; padding: 12px; margin: 8px 0; border-radius: 4px; }
.action-box { background: #fff8e1; border-left: 4px solid #f5a623; padding: 12px; margin: 8px 0; border-radius: 4px; }
.observation-box { background: #f5f5f5; border-left: 4px solid #7b8c9d; padding: 12px; margin: 8px 0; border-radius: 4px; }
.answer-box { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 20px; margin: 12px 0; border-radius: 8px; }
.step-header { font-weight: bold; color: #555; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Search-R1 Mini")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    max_steps = st.slider("Max Search Steps", 1, 10, 5, help="Maximum iterations before forcing an answer")
    use_api = st.checkbox("Use API server", value=False, help="Send requests to the FastAPI server instead of running locally")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Search-R1 Mini** is a deep search agent that uses **ReAct reasoning** to iteratively search the web and synthesize answers.

    **Capabilities:**
    - Multi-step reasoning
    - Iterative web search
    - Query decomposition
    - Retrieval + reranking
    - Source citation

    **Architecture:**
    ```
    Question → Decompose → Think → Search → Rerank → Reflect → Answer
                  ↑___________________________________________↓
                  (iterative loop until sufficient info)
    ```
    """)

# ─── Main Layout ──────────────────────────────────────────────────
st.title("Search-R1 Mini")
st.markdown("*Deep Search Agent with ReAct Reasoning*")

# Input
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Why did NVIDIA stock rise after earnings in May 2024?",
        key="question_input",
    )
with col2:
    search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

# Example questions
with st.expander("💡 Example Questions"):
    examples = [
        "What government position was held by the woman who portrayed Jane Roe in the 1997 film 'Roe vs. Wade'?",
        "Are both the director of Inception and the director of Interstellar from the same country?",
        "Which programming language was created first: Python or JavaScript?",
        "What is the capital of the country that produced the inventor of the telephone?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            question = ex
            st.session_state.question_input = ex

# ─── Execute Search ───────────────────────────────────────────────
if search_btn and question:
    with st.spinner("🤔 Agent is researching..."):
        t0 = time.time()

        if use_api:
            resp = requests.post(
                f"{API_URL}/search",
                json={"question": question, "max_steps": max_steps},
                timeout=300,
            )
            if resp.status_code == 200:
                data = resp.json()
            else:
                st.error(f"API error: {resp.text}")
                st.stop()
        else:
            from app.agent.react_agent import ReactAgent
            agent = ReactAgent()
            agent.max_steps = max_steps
            state = agent.run(question)
            data = {
                "question": state.question,
                "answer": state.final_answer,
                "trace": state.to_trace(),
                "total_steps": state.step_count,
                "sub_queries": state.sub_queries,
            }

        elapsed = time.time() - t0

    # ─── Display Results ──────────────────────────────────────────
    st.markdown("---")

    # Sub-queries
    if data.get("sub_queries") and len(data["sub_queries"]) > 1:
        st.markdown("### 🔀 Query Decomposition")
        cols = st.columns(len(data["sub_queries"]))
        for i, sq in enumerate(data["sub_queries"]):
            with cols[i]:
                st.info(f"**Q{i+1}:** {sq}")

    # Final Answer
    st.markdown("### 📝 Final Answer")
    st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)
    st.caption(f"Completed in {elapsed:.1f}s · {data['total_steps']} search steps")

    # Reasoning Trace
    st.markdown("### 🧠 Reasoning Trace")
    for s in data.get("trace", []):
        with st.expander(f"Step {s['step']}: {s.get('action_query', s.get('action', ''))[:80]}...", expanded=(s['step'] <= 2)):
            st.markdown(f'<div class="step-header">💭 Thought</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="thought-box">{s["thought"]}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="step-header">⚡ Action: {s["action"]}</div>', unsafe_allow_html=True)
            if s.get("action_query"):
                st.markdown(f'<div class="action-box">🔎 <code>{s["action_query"]}</code></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="step-header">📋 Observation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="observation-box">{s["observation"][:600]}</div>', unsafe_allow_html=True)

            if s.get("documents"):
                st.markdown("**Sources found:**")
                for d in s["documents"][:3]:
                    st.caption(f"📄 [{d.get('title', 'N/A')}]({d.get('url', '#')})")

elif search_btn and not question:
    st.warning("Please enter a question first.")
