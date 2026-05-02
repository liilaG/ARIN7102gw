from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..artifacts import ArtifactWriter
from ..chatbot import (
    DeepSeekClient,
    apply_live_data_env,
    build_chatbot_response,
    load_chatbot_config,
    render_index_html,
)
from ..contracts import (
    AnalyzeRequest,
    ArtifactRequest,
    ArtifactResponse,
    MAX_DIALOG_CONTEXT_ITEMS,
    MAX_QUERY_LENGTH,
    MAX_RETRIEVAL_TOP_K,
    MAX_USER_PROFILE_FIELDS,
    MIN_RETRIEVAL_TOP_K,
    PipelineRequest,
    PipelineResponse,
    RetrievalRequest,
)
from ..service import QueryIntelligenceService, build_default_service


def _elapsed_seconds(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.1f}s"


def _short_query(query: str, *, limit: int = 80) -> str:
    return query if len(query) <= limit else f"{query[:limit - 3]}..."


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    user_profile: dict[str, Any] = Field(default_factory=dict, max_length=MAX_USER_PROFILE_FIELDS)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_DIALOG_CONTEXT_ITEMS)
    top_k: int = Field(default=20, ge=MIN_RETRIEVAL_TOP_K, le=MAX_RETRIEVAL_TOP_K)
    debug: bool = False


def create_app(
    service: QueryIntelligenceService | None = None,
    artifact_output_dir: str | Path | None = None,
    app_config: dict[str, Any] | None = None,
    app_config_path: str | Path | None = None,
    deepseek_client: DeepSeekClient | None = None,
) -> FastAPI:
    chatbot_config = app_config or load_chatbot_config(app_config_path, load_env_file=False)
    if service is None:
        apply_live_data_env(chatbot_config)

    app = FastAPI(title="Query Intelligence Service", version="0.1.0")
    if service is None:
        service_started_at = time.perf_counter()
        print("[startup] Loading default Query Intelligence service...", flush=True)
        runtime = build_default_service()
        print(
            f"[startup] Query Intelligence service loaded in {_elapsed_seconds(service_started_at)}.",
            flush=True,
        )
    else:
        runtime = service
    artifact_writer = ArtifactWriter(artifact_output_dir or os.getenv("QI_API_OUTPUT_DIR", "outputs/query_intelligence"))
    print("[startup] Preparing DeepSeek response client...", flush=True)
    response_client = deepseek_client or DeepSeekClient(chatbot_config)
    print("[startup] FastAPI routes are ready.", flush=True)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index_html(chatbot_config)

    @app.post("/chat")
    def chat(payload: ChatRequest) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        request_started_at = time.perf_counter()
        print(f"[chat] Received query: {_short_query(query)}", flush=True)
        print("[chat] Step 1/3: running NLU, retrieval, and live data providers...", flush=True)
        pipeline_started_at = time.perf_counter()
        result = runtime.run_pipeline(
            query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            top_k=payload.top_k,
            debug=payload.debug,
        )
        retrieval = result.get("retrieval_result") or {}
        print(
            "[chat] Step 1/3 complete "
            f"({_elapsed_seconds(pipeline_started_at)}): "
            f"documents={len(retrieval.get('documents') or [])}, "
            f"structured_data={len(retrieval.get('structured_data') or [])}, "
            f"warnings={len(retrieval.get('warnings') or [])}",
            flush=True,
        )
        print("[chat] Step 2/3: calling DeepSeek or fallback answer generator...", flush=True)
        answer_started_at = time.perf_counter()
        response = build_chatbot_response(
            query=query,
            pipeline_result=result,
            deepseek_client=response_client,
            progress=lambda message: print(f"[chat] {message}", flush=True),
        )
        llm_status = response.get("llm") or {}
        print(
            "[chat] Step 2/3 complete "
            f"({_elapsed_seconds(answer_started_at)}): "
            f"llm_status={llm_status.get('status', 'unknown')}",
            flush=True,
        )
        print(f"[chat] Completed request in {_elapsed_seconds(request_started_at)}", flush=True)
        return response

    @app.post("/nlu/analyze")
    def analyze(payload: AnalyzeRequest) -> dict:
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        return runtime.analyze_query(
            payload.query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            debug=payload.debug,
        )

    @app.post("/retrieval/search")
    def retrieval(payload: RetrievalRequest) -> dict:
        return runtime.retrieve_evidence(payload.nlu_result.model_dump(mode="json"), top_k=payload.top_k, debug=payload.debug)

    @app.post("/query/intelligence", response_model=PipelineResponse)
    def pipeline(payload: PipelineRequest) -> dict:
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        return runtime.run_pipeline(
            payload.query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            top_k=payload.top_k,
            debug=payload.debug,
        )

    @app.post("/query/intelligence/artifacts", response_model=ArtifactResponse)
    def pipeline_artifacts(payload: ArtifactRequest) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        result = runtime.run_pipeline(
            query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            top_k=payload.top_k,
            debug=payload.debug,
        )
        written = artifact_writer.write(
            query=query,
            nlu_result=result["nlu_result"],
            retrieval_result=result["retrieval_result"],
            session_id=payload.session_id,
            message_id=payload.message_id,
        )
        return {
            "query_id": result["nlu_result"]["query_id"],
            "run_id": written["run_id"],
            "status": "completed",
            "artifact_dir": written["artifact_dir"],
            "artifacts": written["artifacts"],
            "nlu_result": result["nlu_result"],
            "retrieval_result": result["retrieval_result"],
        }

    return app
