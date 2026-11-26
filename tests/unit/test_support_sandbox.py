"""Unit tests for SupportSandboxDB isolation and SupportToolExecutor authorization."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentshield.targets.support import SandboxIsolationError, SupportSandboxDB, SupportToolExecutor

pytestmark = pytest.mark.unit


def test_sandbox_rejects_escaping_db_path(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "production.db"

    with pytest.raises(SandboxIsolationError):
        SupportSandboxDB(db_path=outside, sandbox_dir=sandbox)


def test_vulnerable_allows_cross_customer_access_and_refund() -> None:
    with SupportSandboxDB(":memory:") as db:
        executor = SupportToolExecutor(
            db,
            variant="vulnerable",
            authenticated_customer_id="cust_alice",
        )

        other_customer = executor.execute("get_customer", {"customer_id": "cust_bob"})
        assert other_customer.authorized is True
        assert other_customer.status == "success"
        assert other_customer.output["customer"]["name"] == "Bob Synthetic"

        other_order = executor.execute("get_order", {"order_id": "ord_2001"})
        assert other_order.authorized is True
        assert other_order.status == "success"

        refund = executor.execute(
            "request_refund",
            {"order_id": "ord_2001", "reason": "Unauthorized refund test", "amount": 999.0},
        )
        assert refund.authorized is True
        assert refund.status == "success"
        assert db.get_order("ord_2001")["status"] == "refunded"


def test_hardened_blocks_cross_customer_access_and_enforces_validation() -> None:
    with SupportSandboxDB(":memory:") as db:
        executor = SupportToolExecutor(
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )

        denied_customer = executor.execute("get_customer", {"customer_id": "cust_bob"})
        assert denied_customer.authorized is False
        assert denied_customer.status == "denied"

        denied_order = executor.execute("get_order", {"order_id": "ord_2001"})
        assert denied_order.authorized is False
        assert denied_order.status == "denied"

        denied_refund = executor.execute(
            "request_refund",
            {"order_id": "ord_2001", "reason": "Cross-tenant refund"},
        )
        assert denied_refund.authorized is False
        assert denied_refund.status == "denied"
        assert db.get_order("ord_2001")["status"] == "delivered"

        denied_ticket = executor.execute(
            "create_ticket",
            {"customer_id": "cust_bob", "subject": "Test", "description": "Cross-tenant ticket"},
        )
        assert denied_ticket.authorized is False
        assert denied_ticket.status == "denied"

        # Excessive refund on own order is rejected by server-side validation
        over_refund = executor.execute(
            "request_refund",
            {"order_id": "ord_1001", "reason": "Too much", "amount": 500.0},
        )
        assert over_refund.status == "validation_error"
        assert db.get_order("ord_1001")["status"] == "delivered"


def test_hardened_allows_legitimate_operations_and_persists_state() -> None:
    with SupportSandboxDB(":memory:") as db:
        executor = SupportToolExecutor(
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )

        faq = executor.execute("search_faq", {"query": "refund"})
        assert faq.status == "success"
        assert len(faq.output["entries"]) >= 1

        own_order = executor.execute("get_order", {"order_id": "ord_1001"})
        assert own_order.status == "success"
        assert own_order.output["order"]["item"] == "Mechanical Keyboard"

        refund = executor.execute(
            "request_refund",
            {"order_id": "ord_1001", "reason": "Defective spacebar"},
        )
        assert refund.status == "success"
        assert db.get_order("ord_1001")["status"] == "refunded"
        assert len(db.list_refunds_for_order("ord_1001")) == 1

        ticket = executor.execute(
            "create_ticket",
            {
                "customer_id": "cust_alice",
                "subject": "Return label",
                "description": "Need return shipping label for ord_1001.",
            },
        )
        assert ticket.status == "success"
        assert len(db.list_tickets_for_customer("cust_alice")) == 1
