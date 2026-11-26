"""CapabilityManifest — extended tool capability description with security metadata.

Phase 3 add-on: Extends the existing TargetCapabilities/DiscoveredTool model with
read/write classification, trusted identity sources, data classifications, and
allowed targets. Produces a compact attack-surface inventory.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentshield.harnesses.base import DiscoveredTool, TargetCapabilities

AccessMode = Literal["read", "write", "read_write", "unknown"]
DataClassification = Literal["public", "internal", "confidential", "pii", "secret"]
TrustSource = Literal["session", "header", "jwt", "none", "unknown"]


class ToolManifestEntry(BaseModel):
    """Security-annotated manifest entry for a single tool."""

    name: str
    description: str = ""
    access_mode: AccessMode = "unknown"
    data_classifications: list[DataClassification] = Field(default_factory=list)
    trusted_identity_sources: list[TrustSource] = Field(default_factory=list)
    requires_ownership_check: bool = False
    allowed_targets: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def is_mutating(self) -> bool:
        return self.access_mode in {"write", "read_write"}

    @property
    def risk_level(self) -> str:
        if self.access_mode == "write" and DataClassification.__args__ and any(
            d in {"pii", "secret", "confidential"} for d in self.data_classifications
        ):
            return "high"
        if self.access_mode == "write":
            return "medium"
        if any(d in {"pii", "secret"} for d in self.data_classifications):
            return "medium"
        return "low"


class CapabilityManifest(BaseModel):
    """Full security capability manifest for a target agent."""

    target_name: str
    target_type: str
    observability: str
    telemetry_available: bool
    tools: list[ToolManifestEntry] = Field(default_factory=list)
    declared_permissions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def attack_surface_summary(self) -> dict[str, Any]:
        """Produce a compact attack-surface inventory from declared capabilities."""
        mutating = [t for t in self.tools if t.is_mutating]
        read_only = [t for t in self.tools if not t.is_mutating and t.access_mode != "unknown"]
        unknown_access = [t for t in self.tools if t.access_mode == "unknown"]
        high_risk = [t for t in self.tools if t.risk_level == "high"]
        medium_risk = [t for t in self.tools if t.risk_level == "medium"]

        return {
            "target_name": self.target_name,
            "total_tools": len(self.tools),
            "mutating_tools": [t.name for t in mutating],
            "read_only_tools": [t.name for t in read_only],
            "unknown_access_tools": [t.name for t in unknown_access],
            "high_risk_tools": [t.name for t in high_risk],
            "medium_risk_tools": [t.name for t in medium_risk],
            "tools_requiring_ownership_check": [
                t.name for t in self.tools if t.requires_ownership_check
            ],
            "tools_without_trusted_identity": [
                t.name for t in self.tools
                if not t.trusted_identity_sources or t.trusted_identity_sources == ["none"]
            ],
        }


# ---------------------------------------------------------------------------
# Default access-mode classification heuristics for the customer support agent
# ---------------------------------------------------------------------------

_WRITE_HINTS = frozenset(["refund", "create", "update", "delete", "write", "transfer", "approve"])
_READ_HINTS = frozenset(["get", "read", "search", "list", "lookup", "query", "faq", "find"])


def _classify_access_mode(tool: DiscoveredTool) -> AccessMode:
    text = f"{tool.name} {tool.description}".lower()
    is_write = any(h in text for h in _WRITE_HINTS)
    is_read = any(h in text for h in _READ_HINTS)
    if is_write and is_read:
        return "read_write"
    if is_write:
        return "write"
    if is_read:
        return "read"
    return "unknown"


def _classify_data(tool: DiscoveredTool) -> list[DataClassification]:
    text = f"{tool.name} {tool.description}".lower()
    classifications: list[DataClassification] = []
    if any(k in text for k in ["customer", "profile", "email", "address"]):
        classifications.append("pii")
    if any(k in text for k in ["refund", "order", "payment", "ticket"]):
        classifications.append("confidential")
    if not classifications:
        classifications.append("internal")
    return classifications


def build_manifest_from_capabilities(capabilities: TargetCapabilities) -> CapabilityManifest:
    """Derive a CapabilityManifest from a TargetCapabilities using heuristic classification.

    For fully annotated manifests, callers should supply ToolManifestEntry objects directly.
    Heuristic classification is labeled 'unknown' in edge cases and should be reviewed.
    """
    entries: list[ToolManifestEntry] = []
    for tool in capabilities.tools:
        access_mode = _classify_access_mode(tool)
        data_classes = _classify_data(tool)
        requires_ownership = access_mode in {"write", "read_write"}

        entries.append(
            ToolManifestEntry(
                name=tool.name,
                description=tool.description,
                access_mode=access_mode,
                data_classifications=data_classes,
                trusted_identity_sources=["session"] if requires_ownership else ["none"],
                requires_ownership_check=requires_ownership,
                input_schema=tool.input_schema,
                permissions=tool.permissions,
                notes=["Classification derived from tool name/description heuristics."]
                if access_mode == "unknown"
                else [],
            )
        )

    return CapabilityManifest(
        target_name=capabilities.target_name,
        target_type=capabilities.target_type,
        observability=capabilities.observability,
        telemetry_available=capabilities.telemetry_available,
        tools=entries,
        declared_permissions=capabilities.declared_permissions,
        metadata={"provenance": "heuristic_from_target_capabilities"},
    )
