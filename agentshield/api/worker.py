"""Bounded local background worker executing AgentShield scans."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from agentshield.api.schemas import CreateScanRequest
from agentshield.api.store import ScanStore
from agentshield.evaluation.models import EvaluationPolicy
from agentshield.harnesses.base import BaseHarness, TargetConfig
from agentshield.harnesses.http_harness import HTTPHarness
from agentshield.harnesses.langgraph_harness import LangGraphHarness
from agentshield.harnesses.mcp_harness import MCPHarness
from agentshield.orchestrator.workflow import ScanOrchestrator
from agentshield.providers.base import BaseLLMProvider
from agentshield.providers.router import FreeProviderRouter
from agentshield.reporting.html_report import render_html_report
from agentshield.reporting.models import build_audit_report
from agentshield.targets.support.agent import CustomerSupportAgent
from agentshield.targets.support.db import SupportSandboxDB
from agentshield.test_engine.models import TestSuiteConfig
from agentshield.tracing.store import TraceStore


class ScanWorker:
    """Executes scan jobs on a bounded ThreadPoolExecutor with cancellation support."""

    def __init__(
        self,
        scan_store: ScanStore,
        trace_store: TraceStore,
        *,
        max_workers: int = 2,
        provider_factory: Callable[[], BaseLLMProvider | FreeProviderRouter] | None = None,
    ) -> None:
        self.scan_store = scan_store
        self.trace_store = trace_store
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agentshield-scan")
        self._provider_factory = provider_factory or FreeProviderRouter.from_env
        self._cancel_flags: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    def submit(self, scan_id: str, request: CreateScanRequest) -> None:
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_flags[scan_id] = cancel_event
            future = self._executor.submit(self._run_scan, scan_id, request, cancel_event)
            self._futures[scan_id] = future

    def cancel(self, scan_id: str) -> bool:
        with self._lock:
            event = self._cancel_flags.get(scan_id)
            if event is not None:
                event.set()
        current = self.scan_store.get_status(scan_id)
        if current is None:
            return False
        if current.status in {"queued", "running"}:
            self.scan_store.update_status(scan_id, "cancelled")
        return True

    def wait_for(self, scan_id: str, *, timeout: float = 30.0) -> None:
        with self._lock:
            future = self._futures.get(scan_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _run_scan(
        self,
        scan_id: str,
        request: CreateScanRequest,
        cancel_event: threading.Event,
    ) -> None:
        if cancel_event.is_set():
            self.scan_store.update_status(scan_id, "cancelled")
            return

        self.scan_store.update_status(scan_id, "running")
        policy = request.policy or EvaluationPolicy(
            authenticated_subject=request.authenticated_customer_id,
            foreign_identifiers=["ord_2001", "cust_bob"],
        )

        provider: BaseLLMProvider | FreeProviderRouter | None = None
        db: SupportSandboxDB | None = None
        harness: BaseHarness | None = None

        try:
            if request.target_type == "support_demo":
                provider = self._provider_factory()
                db = SupportSandboxDB(":memory:")
                agent = CustomerSupportAgent(
                    provider,
                    db,
                    variant=request.support_variant,
                    authenticated_customer_id=request.authenticated_customer_id,
                )
                harness = LangGraphHarness(
                    agent,
                    TargetConfig(target_name=request.target_name, authorized=request.authorized),
                )
            elif request.target_type == "http":
                assert request.target_url is not None
                harness = HTTPHarness(
                    request.target_url,
                    TargetConfig(target_name=request.target_name, authorized=request.authorized),
                )
            elif request.target_type == "mcp":
                assert request.target_url is not None
                harness = MCPHarness(
                    request.target_url,
                    TargetConfig(target_name=request.target_name, authorized=request.authorized),
                )

            assert harness is not None
            orchestrator = ScanOrchestrator(
                harness,
                policy,
                budget=request.budget,
                suite_config=TestSuiteConfig(
                    max_tests=request.budget.max_tests,
                    use_llm_variations=request.use_llm_variations,
                ),
                trace_store=self.trace_store,
                use_llm_judge=request.use_llm_judge,
            )

            if cancel_event.is_set():
                self.scan_store.update_status(scan_id, "cancelled")
                return

            scan_report = orchestrator.run(scan_id=scan_id)
            if cancel_event.is_set():
                self.scan_store.update_status(scan_id, "cancelled")
                return

            audit_data = build_audit_report(scan_report, trace_store=self.trace_store)
            html = render_html_report(audit_data)
            self.scan_store.save_report(scan_id, audit_report=audit_data, html_report=html)
        except Exception as exc:
            self.scan_store.update_status(scan_id, "failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if harness is not None:
                harness.close()
            if db is not None:
                db.close()
            if provider is not None and hasattr(provider, "close"):
                provider.close()
