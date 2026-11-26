"""Live integration tests for Phase 9 FastAPI scan API."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient
import pytest

from agentshield.api import create_app

load_dotenv(override=False)

pytestmark = pytest.mark.integration


def _has_any_key() -> bool:
    return any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"))


def test_live_api_initiates_scan_and_retrieves_report(tmp_path: Path) -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    app = create_app(db_path=tmp_path / "live_api.sqlite")
    with TestClient(app) as client:
        start_resp = client.post(
            "/api/v1/scans",
            json={
                "target_name": "live-hardened-api",
                "target_type": "support_demo",
                "support_variant": "hardened",
                "authorized": True,
                "budget": {"max_tests": 2, "max_turns_per_test": 2, "max_total_tokens": 15000},
            },
        )
        assert start_resp.status_code == 202
        scan_id = start_resp.json()["scan_id"]

        deadline = time.monotonic() + 60.0
        status_data = {}
        while time.monotonic() < deadline:
            status_resp = client.get(f"/api/v1/scans/{scan_id}")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            if status_data["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.25)

        assert status_data["status"] == "completed", status_data.get("error")
        assert status_data["tests_executed"] == 2

        report_resp = client.get(f"/api/v1/scans/{scan_id}/report?format=json")
        assert report_resp.status_code == 200
        assert report_resp.json()["report"]["total_tokens_used"] > 0

        html_resp = client.get(f"/api/v1/scans/{scan_id}/report?format=html")
        assert html_resp.status_code == 200
        assert "AgentShield Security Audit" in html_resp.text
