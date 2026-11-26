"""FastAPI application exposing AgentShield scan orchestration and reporting."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agentshield.api.schemas import (
    CreateScanRequest,
    ReportResponse,
    ScanFindingsResponse,
    ScanStatusResponse,
    TestSuiteDescriptor,
    TestSuitesResponse,
)
from agentshield.api.security import TargetValidationError, validate_target_url
from agentshield.api.store import ScanStore
from agentshield.api.worker import ScanWorker
from agentshield.evaluation.models import TAXONOMY_BY_CATEGORY
from agentshield.providers.base import BaseLLMProvider
from agentshield.providers.router import FreeProviderRouter
from agentshield.tracing.store import TraceStore

_SUITE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "unauthorized_tool_execution": (
        "Detects tools executed without server-side authorization or mutating sandbox state.",
        "Agents with mutating/transactional tools",
    ),
    "sensitive_information_disclosure": (
        "Detects synthetic canary secrets or credentials disclosed in responses or tool outputs.",
        "All agents and read/document assistants",
    ),
    "excessive_agency": (
        "Detects tool invocations exceeding declared allowed_tools or hitting forbidden_tools.",
        "Agents with declared tool permission policies",
    ),
    "data_isolation_failure": (
        "Verifies synthetic tenant boundaries across customer/order identifiers.",
        "Multi-tenant and data-lookup agents",
    ),
    "resource_consumption": (
        "Verifies token, LLM step, and tool-call execution budgets.",
        "All agents with configured resource budgets",
    ),
    "indirect_prompt_injection": (
        "Verifies whether untrusted retrieved content triggers unauthorized tool behavior.",
        "Agents retrieving external/FAQ/document text",
    ),
}


def create_app(
    *,
    db_path: str | Path = ":memory:",
    allow_localhost_targets: bool = False,
    max_workers: int = 2,
    provider_factory: Callable[[], BaseLLMProvider | FreeProviderRouter] | None = None,
) -> FastAPI:
    """Create and configure the AgentShield FastAPI service."""
    scan_store = ScanStore(db_path)
    trace_store = TraceStore(db_path)
    worker = ScanWorker(
        scan_store,
        trace_store,
        max_workers=max_workers,
        provider_factory=provider_factory,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        worker.shutdown(wait=True)
        scan_store.close()
        trace_store.close()

    app = FastAPI(
        title="AgentShield Security Auditing API",
        version="0.1.0",
        description="Zero-cost AI agent security auditing, tracing, and reporting API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.scan_store = scan_store
    app.state.trace_store = trace_store
    app.state.worker = worker

    @app.get("/api/v1/test-suites", response_model=TestSuitesResponse, tags=["Discovery"])
    def list_test_suites() -> TestSuitesResponse:
        suites = []
        for category, taxonomy in TAXONOMY_BY_CATEGORY.items():
            desc, applicable = _SUITE_DESCRIPTIONS[category]
            suites.append(
                TestSuiteDescriptor(
                    category=category,
                    description=desc,
                    owasp_llm=taxonomy.owasp_llm,
                    mitre_atlas=taxonomy.mitre_atlas,
                    applicable_to=applicable,
                )
            )
        return TestSuitesResponse(suites=suites)

    @app.post(
        "/api/v1/scans",
        response_model=ScanStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Scans"],
    )
    def start_scan(payload: CreateScanRequest) -> ScanStatusResponse:
        if not payload.authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Explicit target authorization (authorized=true) is required to start a scan.",
            )

        if payload.target_type in {"http", "mcp"}:
            if not payload.target_url:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"target_url is required when target_type='{payload.target_type}'.",
                )
            try:
                validate_target_url(payload.target_url, allow_localhost=allow_localhost_targets)
            except TargetValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc

        scan_id = f"scan_{uuid.uuid4().hex[:10]}"
        created = scan_store.create_scan(
            scan_id=scan_id,
            target_name=payload.target_name,
            target_type=payload.target_type,
        )
        worker.submit(scan_id, payload)
        return created

    @app.get("/api/v1/scans/{scan_id}", response_model=ScanStatusResponse, tags=["Scans"])
    def get_scan_status(scan_id: str) -> ScanStatusResponse:
        job = scan_store.get_status(scan_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scan '{scan_id}' not found.")
        return job

    @app.get("/api/v1/scans/{scan_id}/findings", response_model=ScanFindingsResponse, tags=["Scans"])
    def get_scan_findings(scan_id: str) -> ScanFindingsResponse:
        job = scan_store.get_status(scan_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scan '{scan_id}' not found.")
        audit_data, _ = scan_store.get_report(scan_id)
        return ScanFindingsResponse(
            scan_id=scan_id,
            status=job.status,
            overall_verdict=job.overall_verdict,
            findings=audit_data.findings if audit_data else [],
        )

    @app.get("/api/v1/scans/{scan_id}/report", response_model=ReportResponse, tags=["Scans"])
    def get_scan_report(
        scan_id: str,
        format: Literal["json", "html"] = Query("json"),
    ):
        job = scan_store.get_status(scan_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scan '{scan_id}' not found.")
        audit_data, html = scan_store.get_report(scan_id)
        if audit_data is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Report not available yet (scan status: {job.status}).",
            )
        if format == "html":
            return HTMLResponse(content=html or "")
        return ReportResponse(scan_id=scan_id, status=job.status, report=audit_data)

    @app.post("/api/v1/scans/{scan_id}/cancel", response_model=ScanStatusResponse, tags=["Scans"])
    def cancel_scan(scan_id: str) -> ScanStatusResponse:
        ok = worker.cancel(scan_id)
        if not ok:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Scan '{scan_id}' not found.")
        job = scan_store.get_status(scan_id)
        assert job is not None
        return job

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="dashboard")

    return app
