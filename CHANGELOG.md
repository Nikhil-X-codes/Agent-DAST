# Changelog

All notable additions to AgentShield (Agent DAST) are documented in this file.

## Unreleased — Phase 3–9 Add-Ons

### Added
- **AgentShield Guard (`agentshield/guard/`)**: Runtime tool-execution middleware (`AgentShieldGuard`, `PolicyEngine`, `GuardPolicy`, `ApprovalToken`) supporting default-deny unknown tools, tool allowlists/denylists, tenant allowlists, argument/resource owner resolution, session call caps, bound expiring human approval tokens, and `TraceStore` decision logging.
- **SARIF 2.1.0 Reporting (`agentshield/reporting/sarif_report.py`)**: `build_sarif` and `write_sarif_report` emitting stable `agentshield/<category>` rule IDs, OWASP LLM / MITRE ATLAS metadata, and omitting unknown source locations.
- **Regression Replay (`agentshield/reporting/regression.py`)**: `RegressionStore`, `RegressionRecord`, `create_regressions_from_scan_report`, `replay_regression_record`, and `replay_scan_regressions` for before/after remediation tracking (`open`, `fixed`, `regressed`, `inconclusive`).
- **CI Failure Gate & CLI Enhancements (`agentshield/cli.py`, `.github/workflows/ci.yml`)**:
  - `agentshield scan` supports `--sarif`, `--fail-on-confirmed`, and `--regressions-db`.
  - Added `agentshield report` (`--sarif`, `--fail-on-confirmed`), `agentshield replay` (`--execute`, `--variant`), and offline `agentshield manifest`.
  - Added `security-gate` job and SARIF artifact upload in `.github/workflows/ci.yml`.
- **OpenTelemetry File/Console Export (`agentshield/tracing/otel_exporter.py`)**: `export_scan_traces_to_file` and `try_export_with_otel_sdk` converting `LLMTraceEvent` and `ToolTraceEvent` records into OTEL-compatible JSONL spans without external hosting.
- **Capability Manifest (`agentshield/harnesses/manifest.py`)**: `CapabilityManifest`, `ToolManifestEntry`, and `build_manifest_from_capabilities` for read/write classification, data classification, ownership requirements, and attack-surface summaries.
- **RAG & Persistent-Memory Isolation Target (`agentshield/targets/rag/`)**: `RAGMemoryStore` and `RAGAgent` (`search_documents`, `read_memory`, `write_memory`) with vulnerable vs. hardened tenant-scoped retrieval, untrusted document provenance tagging, and session-memory isolation, plus `LangGraphHarness` support.
- **Unit & Integration Tests**: Added `test_guard.py`, `test_sarif.py`, `test_regression.py`, `test_ci_gate.py`, `test_otel_exporter.py`, `test_manifest.py`, `test_rag_isolation.py`, and opt-in `tests/integration/test_live_rag.py`.
