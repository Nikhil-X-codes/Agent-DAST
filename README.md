# AgentShield — Agent DAST

**Dynamic Application Security Testing (DAST) & Runtime Policy Gateway for LLM-Powered AI Agents.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero-Cost Inference](https://img.shields.io/badge/cost-zero--spend%20free%20tier-brightgreen.svg)](#supported-free-tier-providers)
[![OWASP LLM & MITRE ATLAS](https://img.shields.io/badge/taxonomy-OWASP%20LLM%20%2B%20MITRE%20ATLAS-orange.svg)](#features)

AgentShield is an open-source **Agent DAST (Dynamic Application Security Testing)** framework that dynamically probes live LLM-powered agents against real tool-calling workflows, isolated sandboxed state, and deterministic security assertions—without requiring paid LLM APIs, cloud vector databases, or SaaS subscriptions.

Instead of judging agent security from text completions alone, AgentShield executes real model-driven tool calls in isolated sandboxes and measures **verifiable runtime consequences**: unauthorized actions, cross-tenant data disclosure, persistent memory leakage, indirect prompt injection, and resource-budget failures.

---

## Table of Contents

- [Why Agent DAST?](#why-agent-dast)
- [Features](#features)
- [Architecture](#architecture)
- [Supported Free-Tier Providers](#supported-free-tier-providers)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [FastAPI Service & React Dashboard](#fastapi-service--react-dashboard)
- [Project Structure](#project-structure)
- [Authorized Testing & Limitations](#authorized-testing--limitations)
- [License](#license)

---

## Why Agent DAST?

Static prompt scanning and single-turn text evals miss the real security risks of tool-using AI agents. AgentShield applies **Dynamic Application Security Testing (DAST)** principles specifically to agentic systems:

1. **Capability & Attack-Surface Discovery**: Inspects Python callables, HTTP endpoints, LangGraph agents, and MCP servers into a typed `CapabilityManifest` with read/write, data-classification, and ownership-risk metadata.
2. **Context-Aware Dynamic Probing**: Selects bounded single-turn and multi-turn security tests tailored to the target's actual tools and `EvaluationPolicy` boundaries.
3. **Real Sandboxed Tool Execution**: Runs live LLM tool calls against isolated synthetic SQLite sandboxes (Customer Support and Multi-Tenant RAG/Memory) with both `vulnerable` and `hardened` server-side controls.
4. **Consequence-Based Evaluation**: Separates a harmless *tool mention* in text, a *blocked attempt* at the authorization boundary, and a *confirmed policy violation* that reads foreign records or mutates state.
5. **Runtime Guardrails & Regression Replay**: Enforces server-side policies with `AgentShieldGuard`, persists confirmed findings in `RegressionStore` for before/after verification, exports **SARIF 2.1.0**, and gates CI merges.

---

## Features

- **Bounded DAST Orchestration**: LangGraph-based `ScanOrchestrator` enforces strict `max_tests`, `max_turns_per_test`, `max_total_tokens`, and `max_duration_seconds` budgets.
- **Multi-Target Harnesses**: Standardized `BaseHarness` adapters for `PythonCallableHarness`, `HTTPHarness`, `LangGraphHarness`, and `MCPHarness`, plus offline `CapabilityManifest` generation.
- **Evidence-Backed Verdicts**: Emits `confirmed`, `suspected`, `inconclusive`, or `no_violation_observed`. Black-box targets without trace telemetry never fabricate tool findings (`telemetry_available=False` → `inconclusive`).
- **OWASP LLM Top 10 & MITRE ATLAS Taxonomy**: Evaluates 6 security categories:
  - `unauthorized_tool_execution` (`LLM06:2025` / `AML.T0051.000`)
  - `sensitive_information_disclosure` (`LLM02:2025` / `AML.T0024`)
  - `excessive_agency` (`LLM06:2025` / `AML.T0051.000`)
  - `data_isolation_failure` (`LLM02:2025` / `AML.T0024`)
  - `resource_consumption` (`LLM10:2025` / `AML.T0029`)
  - `indirect_prompt_injection` (`LLM01:2025` / `AML.T0051.001`)
- **Runtime Policy Gateway (`AgentShieldGuard`)**: Drop-in tool-execution middleware enforcing default-deny unknown tools, allowlists/denylists, tenant isolation, server-side `resource_owner_resolver`, session call caps, action-bound expiring `ApprovalToken`s, and `TraceStore` audit logging.
- **Multi-Tenant RAG & Memory Isolation Sandbox**: Dedicated `RAGAgent` and `RAGMemoryStore` testing tenant-scoped retrieval, untrusted document provenance warnings, indirect prompt injection resistance, and persistent session-memory isolation.
- **Zero-Cost Provider Routing**: `FreeProviderRouter` routes across configured free-tier providers (`Groq`, `Gemini`, `OpenRouter`) and blocks paid models when `AGENTSHIELD_ZERO_COST=true` (enabled by default).
- **CI/CD Reporting & Observability**:
  - Standalone **HTML**, **JSON**, **CLI**, and **SARIF 2.1.0** (`--sarif`) reports with stable `agentshield/<category>` rule IDs.
  - **CI Failure Gate** (`--fail-on-confirmed` and GitHub Actions `security-gate` job).
  - **Regression Replay** (`RegressionStore` + `agentshield replay`) for reproducible remediation verification.
  - **OpenTelemetry-Compatible Export** (`export_scan_traces_to_file`) writing JSONL spans without external hosting.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                CLI (Typer + Rich)  │  FastAPI (/api/v1)  │  React UI    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
               LangGraph ScanOrchestrator (Budgets & Scope)
                      │                            │
                      ▼                            ▼
       Context-Aware Test Engine            Target Harnesses
       (Static Templates + Bounded          (Python / HTTP / LangGraph /
        LLM Variations + Validator)          MCP + CapabilityManifest)
                      │                            │
                      └─────────────┬──────────────┘
                                    ▼
                        Real Target Agent (LangGraph)
                        [Support Agent  │  RAG Agent]
                                    │
                                    ▼
                    AgentShieldGuard (PolicyEngine Middleware)
                    [Allowlists │ Ownership │ ApprovalTokens]
                                    │
                                    ▼
                     Isolated SQLite Synthetic Sandboxes
                     (Customer Support DB │ RAG & Memory DB)
                                    │
                                    ▼
                TraceCollector & TraceStore (SQLite + Redaction)
                         (+ Optional OTEL JSONL Export)
                                    │
                                    ▼
                    Evidence-Based SecurityEvaluator
                    (Deterministic Assertions + Optional LLM Judge)
                                    │
                                    ▼
        HTML / JSON / SARIF 2.1.0 Reports │ RegressionStore │ CI Gate
```

### Component Overview

| Component | Module Path | Responsibility |
| :--- | :--- | :--- |
| **CLI & REST API** | `agentshield/cli.py`, `agentshield/api/` | Typer CLI (`demo`, `scan`, `report`, `replay`, `manifest`, `benchmark`, `serve`), async scan worker, SSRF protection, SQLite job store |
| **Provider Layer** | `agentshield/providers/` | `GroqProvider`, `GeminiProvider`, `OpenRouterProvider`, `FreeProviderRouter`, bounded retries, zero-cost allowlist enforcement |
| **Target Harnesses** | `agentshield/harnesses/` | Python, HTTP, LangGraph, MCP harnesses and `CapabilityManifest` attack-surface builder |
| **Test Engine** | `agentshield/test_engine/` | `ContextAwareTestPlanner`, parameterized static templates, bounded LLM variation generator, scope/safety `Validator` |
| **Scan Orchestrator** | `agentshield/orchestrator/` | LangGraph workflow enforcing token, turn, test-count, and duration budgets |
| **Runtime Guard** | `agentshield/guard/` | `AgentShieldGuard`, `PolicyEngine`, `GuardPolicy`, and action-bound `ApprovalToken` |
| **Sandbox Targets** | `agentshield/targets/` | `CustomerSupportAgent` (`SupportSandboxDB`) and `RAGAgent` (`RAGMemoryStore`) with `vulnerable` and `hardened` variants |
| **Tracing & Redaction** | `agentshield/tracing/` | `TraceCollector`, `TraceStore` (SQLite), credential `redactor`, and OTEL JSONL span exporter |
| **Evaluation Engine** | `agentshield/evaluation/` | Deterministic policy assertions, OWASP/MITRE mappings, and optional sanitized `llm_judge` |
| **Reporting & Replay** | `agentshield/reporting/` | Jinja2 HTML, JSON, SARIF 2.1.0, CLI summary, `BenchmarkRunner`, and `RegressionStore` |
| **Web Dashboard** | `frontend/` | React + Vite + TypeScript + Tailwind CSS interface for launching authorized scans and viewing findings |

---

## Supported Free-Tier Providers

| Provider | Default Free Model | Environment Variable | Notes |
| :--- | :--- | :--- | :--- |
| **Groq** | `openai/gpt-oss-20b` | `GROQ_API_KEY` | Fast inference & tool calling (free-tier quotas) |
| **Google Gemini** | `gemini-flash-lite-latest` | `GEMINI_API_KEY` | Free-tier inference & optional semantic judge |
| **OpenRouter** | `openrouter/free` | `OPENROUTER_API_KEY` | Free-tier auto-router (`:free` models only in zero-cost mode) |

> Strict zero-cost mode (`AGENTSHIELD_ZERO_COST=true`) is enabled by default and raises `ZeroCostViolationError` if a non-allowlisted model is requested.

---

## Setup & Installation

### Prerequisites

- **Python** `>= 3.11`
- **Node.js** `>= 20` and `npm` (optional, for the React dashboard in `frontend/`)
- At least one free-tier LLM API key (only needed for live scans/demos; offline unit tests, `manifest`, and `report` run without API keys)

### 1. Clone the Repository

```bash
git clone https://github.com/Nikhil-X-codes/Agent-DAST.git
cd Agent-DAST
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and add at least one free-tier provider key:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=

# Strict zero-cost enforcement (true | false)
AGENTSHIELD_ZERO_COST=true

# Optional default provider/model overrides (groq | gemini | openrouter)
AGENTSHIELD_PROVIDER=
AGENTSHIELD_MODEL=
```

### 4. Verify Installation

```bash
# Fast offline unit tests (mocked HTTP transports, no network or API keys required)
pytest -m unit -v

# Opt-in live integration tests (requires at least one free-tier API key in .env)
pytest -m integration -v
```

---

## Usage

### CLI Workflows

```bash
# Side-by-side vulnerable vs. hardened customer support demo (real LLM inference)
python -m agentshield.cli demo --variant both

# Run a bounded DAST scan and generate HTML, JSON, SARIF reports + save regressions
python -m agentshield.cli scan --variant vulnerable --max-tests 3 --sarif --regressions-db regressions.sqlite

# Export SARIF from an existing JSON report and run the CI failure gate
python -m agentshield.cli report reports/scan_1234.json --sarif --fail-on-confirmed

# List or replay saved regression cases against the hardened agent
python -m agentshield.cli replay scan_1234 --db-path regressions.sqlite --execute --variant hardened

# Inspect target capability manifest & attack-surface inventory offline (no API key required)
python -m agentshield.cli manifest --variant hardened --out reports/manifest.json

# Compare vulnerable vs. hardened variants across repeated benchmark trials
python -m agentshield.cli benchmark --trials 1 --max-tests 2
```

### Protecting Tools with `AgentShieldGuard`

```python
from agentshield.guard import AgentShieldGuard, GuardPolicy
from agentshield.targets.support import SupportSandboxDB, SupportToolExecutor

with SupportSandboxDB() as db:
    executor = SupportToolExecutor(db, variant="vulnerable", authenticated_customer_id="cust_alice")
    policy = GuardPolicy(
        allowed_tools=["get_order", "search_faq", "request_refund", "create_ticket"],
        sensitive_tools=["request_refund"],
        require_ownership=True,
        max_calls_per_session=10,
    )
    guard = AgentShieldGuard(
        executor,
        policy,
        session_id="sess_001",
        actor="cust_alice",
        resource_owner_resolver=lambda _tool, args: (
            db.get_order(args["order_id"])["customer_id"]
            if "order_id" in args and db.get_order(args["order_id"])
            else None
        ),
    )
```

---

## FastAPI Service & React Dashboard

Start the local FastAPI backend (binds to `127.0.0.1:8000` by default):

```bash
python -m agentshield.cli serve --host 127.0.0.1 --port 8000
```

Optionally launch the React dashboard in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` (interactive OpenAPI documentation is served at `http://127.0.0.1:8000/docs`).

### Key API Endpoints

- `POST /api/v1/scans` — Start an authorized security scan (`authorized=true` required)
- `GET /api/v1/scans/{scan_id}` — Retrieve scan execution status
- `GET /api/v1/scans/{scan_id}/findings` — Retrieve findings for a scan
- `GET /api/v1/scans/{scan_id}/report?format=json|html` — Retrieve JSON or standalone HTML report
- `POST /api/v1/scans/{scan_id}/cancel` — Cancel a queued or running scan
- `GET /api/v1/test-suites` — List available security test suites and OWASP/MITRE mappings

---

## Project Structure

```text
Agent-DAST/
├── .github/workflows/ci.yml    # Offline unit tests, security-gate job, and frontend build CI
├── agentshield/
│   ├── api/                    # FastAPI app, SSRF target URL validation, SQLite job store & worker
│   ├── evaluation/             # Deterministic policy evaluators, OWASP/MITRE models, optional LLM judge
│   ├── guard/                  # AgentShieldGuard runtime policy middleware & ApprovalToken enforcement
│   ├── harnesses/              # Python, HTTP, LangGraph, MCP harnesses & CapabilityManifest
│   ├── orchestrator/           # Budget-bounded LangGraph ScanOrchestrator
│   ├── providers/              # Groq, Gemini, OpenRouter adapters & FreeProviderRouter
│   ├── reporting/              # HTML, JSON, SARIF 2.1.0, CLI, BenchmarkRunner & RegressionStore
│   ├── targets/
│   │   ├── support/            # CustomerSupportAgent + SupportSandboxDB + HTTP app
│   │   └── rag/                # RAGAgent + RAGMemoryStore (tenant & session-memory isolation)
│   ├── test_engine/            # Context-aware planner, static templates, LLM generator & validator
│   ├── tracing/                # TraceCollector, TraceStore, redactor & OTEL JSONL exporter
│   └── cli.py                  # Typer + Rich CLI entrypoint
├── frontend/                   # React + Vite + Tailwind CSS dashboard
├── tests/
│   ├── unit/                   # Offline unit & policy fixture tests
│   └── integration/            # Opt-in live LLM integration tests
├── .env.example                # Environment template (no secrets)
├── CHANGELOG.md                # Release and add-on history
├── pyproject.toml              # Python packaging & pytest configuration
└── SECURITY.md                 # Authorized testing rules & vulnerability disclosure
```

---

## Authorized Testing & Limitations

- **Explicit Authorization Required**: Only scan agents, APIs, and MCP servers you own or have explicit written authorization to test. Both `BaseHarness` and `/api/v1/scans` reject targets without `authorized=True`, and `validate_target_url` blocks metadata/private/localhost SSRF targets by default. See [SECURITY.md](SECURITY.md).
- **Black-Box vs. Instrumented**: Black-box targets that do not expose tool traces report `telemetry_available=False` and `inconclusive` for tool-level assertions—AgentShield never fabricates trace evidence.
- **Free-Tier Quotas**: Free API tiers enforce rate limits and daily request caps; `ScanOrchestrator` enforces hard token, turn, test, and duration budgets and reports partial results honestly when budgets are exhausted.

## License

MIT — see [LICENSE](LICENSE).
