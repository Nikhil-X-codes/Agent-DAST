"""Unit tests for Phase 9 FastAPI service, SSRF protection, and SQLite scan persistence."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from agentshield.api import TargetValidationError, create_app, validate_target_url
from agentshield.providers import GroqProvider, ProviderConfig

pytestmark = pytest.mark.unit


def _mock_provider_factory() -> GroqProvider:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        last_msg = body["messages"][-1]
        if last_msg["role"] == "user" and ("ord_2001" in (last_msg.get("content") or "") or "refund" in (last_msg.get("content") or "").lower()):
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-20b",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{call_count}",
                                        "type": "function",
                                        "function": {
                                            "name": "get_order",
                                            "arguments": json.dumps({"order_id": "ord_2001"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30},
            },
        )

    return GroqProvider(
        ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
        transport=httpx.MockTransport(handler),
    )


def test_ssrf_validation_blocks_metadata_private_and_non_http() -> None:
    for blocked in (
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5:8080/invoke",
        "http://192.168.1.10/invoke",
        "http://127.0.0.1:8000/invoke",
        "file:///etc/passwd",
    ):
        with pytest.raises(TargetValidationError):
            validate_target_url(blocked, allow_localhost=False)

    assert validate_target_url("http://127.0.0.1:8000/invoke", allow_localhost=True).startswith("http://127.0.0.1")


def test_openapi_and_test_suites_endpoint(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "api.sqlite", provider_factory=_mock_provider_factory)
    with TestClient(app) as client:
        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        paths = openapi.json()["paths"]
        assert "/api/v1/scans" in paths
        assert "/api/v1/scans/{scan_id}" in paths
        assert "/api/v1/scans/{scan_id}/findings" in paths
        assert "/api/v1/scans/{scan_id}/report" in paths
        assert "/api/v1/scans/{scan_id}/cancel" in paths
        assert "/api/v1/test-suites" in paths

        suites_resp = client.get("/api/v1/test-suites")
        assert suites_resp.status_code == 200
        assert len(suites_resp.json()["suites"]) == 6


def test_api_rejects_unauthorized_and_ssrf_targets(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "api.sqlite", provider_factory=_mock_provider_factory)
    with TestClient(app) as client:
        unauth = client.post(
            "/api/v1/scans",
            json={"target_name": "demo", "target_type": "support_demo", "authorized": False},
        )
        assert unauth.status_code == 403

        ssrf = client.post(
            "/api/v1/scans",
            json={
                "target_name": "ssrf-test",
                "target_type": "http",
                "authorized": True,
                "target_url": "http://169.254.169.254/latest/meta-data",
            },
        )
        assert ssrf.status_code == 400


def test_api_full_scan_lifecycle_and_report_retrieval(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "api.sqlite", provider_factory=_mock_provider_factory)
    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/scans",
            json={
                "target_name": "vuln-demo",
                "target_type": "support_demo",
                "support_variant": "vulnerable",
                "authorized": True,
                "budget": {"max_tests": 2, "max_turns_per_test": 2, "max_total_tokens": 10000},
            },
        )
        assert create_resp.status_code == 202
        scan_id = create_resp.json()["scan_id"]

        app.state.worker.wait_for(scan_id, timeout=15.0)

        status_resp = client.get(f"/api/v1/scans/{scan_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "completed"
        assert status_resp.json()["overall_verdict"] == "confirmed"

        findings_resp = client.get(f"/api/v1/scans/{scan_id}/findings")
        assert findings_resp.status_code == 200
        assert len(findings_resp.json()["findings"]) >= 1

        json_report = client.get(f"/api/v1/scans/{scan_id}/report?format=json")
        assert json_report.status_code == 200
        assert json_report.json()["report"]["scan_id"] == scan_id

        html_report = client.get(f"/api/v1/scans/{scan_id}/report?format=html")
        assert html_report.status_code == 200
        assert "AgentShield Security Audit" in html_report.text


def test_api_cancel_scan_endpoint(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "api.sqlite", provider_factory=_mock_provider_factory)
    with TestClient(app) as client:
        missing = client.post("/api/v1/scans/nonexistent/cancel")
        assert missing.status_code == 404

        create_resp = client.post(
            "/api/v1/scans",
            json={"target_name": "cancel-demo", "target_type": "support_demo", "authorized": True},
        )
        scan_id = create_resp.json()["scan_id"]
        cancel_resp = client.post(f"/api/v1/scans/{scan_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] in {"cancelled", "completed"}
