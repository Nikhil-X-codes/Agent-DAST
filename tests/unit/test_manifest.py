"""Unit tests for CapabilityManifest (Phase 3 add-on)."""

from __future__ import annotations

import pytest

from agentshield.harnesses.base import DiscoveredTool, TargetCapabilities
from agentshield.harnesses.manifest import (
    CapabilityManifest,
    ToolManifestEntry,
    build_manifest_from_capabilities,
)


def _make_capabilities() -> TargetCapabilities:
    return TargetCapabilities(
        target_type="langgraph",
        target_name="support-hardened",
        observability="instrumented",
        telemetry_available=True,
        tools=[
            DiscoveredTool(name="get_order", description="Retrieve order details by order_id."),
            DiscoveredTool(name="get_customer", description="Retrieve customer profile details."),
            DiscoveredTool(name="request_refund", description="Process a refund for an order."),
            DiscoveredTool(name="search_faq", description="Search the FAQ knowledge base."),
            DiscoveredTool(name="create_ticket", description="Create a new support ticket."),
        ],
    )


@pytest.mark.unit
class TestCapabilityManifest:
    def test_manifest_has_correct_tool_count(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        assert len(manifest.tools) == 5

    def test_mutating_tools_classified_as_write(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        refund_entry = next(t for t in manifest.tools if t.name == "request_refund")
        assert refund_entry.access_mode in {"write", "read_write"}
        assert refund_entry.is_mutating is True

    def test_read_tools_classified_as_read(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        order_entry = next(t for t in manifest.tools if t.name == "get_order")
        assert order_entry.access_mode == "read"
        assert order_entry.is_mutating is False

    def test_mutating_tools_require_ownership_check(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        mutating = [t for t in manifest.tools if t.is_mutating]
        assert all(t.requires_ownership_check for t in mutating)

    def test_read_tools_do_not_require_ownership(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        # search_faq is read-only and contains no personal data hint
        faq_entry = next(t for t in manifest.tools if t.name == "search_faq")
        assert faq_entry.requires_ownership_check is False

    def test_pii_classified_for_customer_tool(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        customer_entry = next(t for t in manifest.tools if t.name == "get_customer")
        assert "pii" in customer_entry.data_classifications

    def test_attack_surface_summary_structure(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        summary = manifest.attack_surface_summary()
        assert summary["total_tools"] == 5
        assert "request_refund" in summary["mutating_tools"]
        assert "get_order" in summary["read_only_tools"]

    def test_attack_surface_summary_identifies_ownership_required(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        summary = manifest.attack_surface_summary()
        ownership_required = summary["tools_requiring_ownership_check"]
        assert "request_refund" in ownership_required
        assert "create_ticket" in ownership_required

    def test_target_name_preserved_in_manifest(self):
        caps = _make_capabilities()
        manifest = build_manifest_from_capabilities(caps)
        assert manifest.target_name == "support-hardened"
        assert manifest.target_type == "langgraph"

    def test_manual_manifest_entry_risk_level(self):
        entry = ToolManifestEntry(
            name="delete_account",
            access_mode="write",
            data_classifications=["pii", "confidential"],
            trusted_identity_sources=["session"],
        )
        assert entry.risk_level == "high"
        assert entry.is_mutating is True

    def test_read_low_risk_entry(self):
        entry = ToolManifestEntry(
            name="get_faq",
            access_mode="read",
            data_classifications=["public"],
        )
        assert entry.risk_level == "low"
        assert entry.is_mutating is False
