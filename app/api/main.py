"""
FastAPI server for the Search-R1 Mini agent.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.schemas import SearchRequest, SearchResponse, TraceStep, ErrorResponse
from app.agent.react_agent import ReactAgent

app = FastAPI(
    title="Search-R1 Mini",
    description="Deep Search Agent with ReAct reasoning, iterative search, and reranking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: ReactAgent | None = None


def get_agent() -> ReactAgent:
    global _agent
    if _agent is None:
        _agent = ReactAgent()
    return _agent


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse, responses={500: {"model": ErrorResponse}})
async def deep_search(request: SearchRequest):
    """Execute a deep search with the ReAct agent."""
    try:
        agent = get_agent()
        if request.max_steps is not None:
            agent.max_steps = request.max_steps

        state = agent.run(request.question)

        trace = [
            TraceStep(
                step=s.step_num,
                thought=s.thought,
                action=s.action,
                action_query=s.action_query,
                observation=s.observation,
                documents=[d.to_dict() for d in s.documents],
            )
            for s in state.steps
        ]

        return SearchResponse(
            question=request.question,
            answer=state.final_answer,
            trace=trace,
            total_steps=len(state.steps),
            sub_queries=state.sub_queries,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/stream")
async def deep_search_stream(request: SearchRequest):
    """Execute a deep search with streaming trace events (SSE)."""
    from fastapi.responses import StreamingResponse
    import json

    async def event_generator():
        try:
            agent = get_agent()
            if request.max_steps is not None:
                agent.max_steps = request.max_steps

            for event in agent.run_stream(request.question):
                yield f"data: {json.dumps(event, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    from app.utils.config import get_config
    cfg = get_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
