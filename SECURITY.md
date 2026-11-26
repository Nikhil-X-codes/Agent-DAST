# Security Policy

## Scope & Authorized Testing Only

AgentShield is a defensive AI agent security auditing framework.

- **Explicit Authorization Required**: You may only use AgentShield to audit AI agents, APIs, MCP servers, or repositories that you own or have explicit written permission to test.
- **No Unauthorized Scanning**: Do not point AgentShield's Target Discovery or Security Test Engine at third-party endpoints or production systems without authorization.

---

## Sandboxing & Environment Isolation

Because AgentShield executes real LLM inference and real tool calls during security audits, strict isolation is mandatory:

1. **Containerized Execution**: Run target agents and untrusted tool integrations inside isolated Docker containers with minimal privileges, restricted network egress, and no host filesystem mounts beyond the synthetic workspace.
2. **Synthetic Data Only**: All tool-execution tests (such as SQLite state changes) must operate exclusively on ephemeral, synthetic test data. Never use real PII, production databases, or live customer credentials.
3. **Separation of Mock vs. Live Modes**: Keep mock/unit-test harnesses strictly separated from live security auditing pipelines so test stubs are never mistaken for real security coverage, and live audit payloads are never accidentally executed against real environments.

---

## Secrets & Provider Hygiene

- **No Committed Credentials**: Never commit `.env` files, API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`), tokens, or private certificates to the repository.
- **Third-Party Free Tiers**: When using hosted free-tier providers (Groq, Google Gemini, OpenRouter), be aware that prompts and outputs are transmitted to external APIs—do not send confidential data or real PII to hosted endpoints.

---

## Resource & Budget Controls

To prevent runaway agent loops, quota exhaustion, or denial-of-service on local hardware:

- **Bounded Feedback Loops**: The Scan Orchestrator must enforce strict caps on scan duration, turn counts, tool calls, token usage, and feedback iterations between the Evaluation Engine and Security Test Engine.
- **Container Resource Limits**: Apply CPU, memory, and execution timeouts to sandboxed target containers.

---

## Handling Audit Artifacts & Reports

Generated security reports (HTML, JSON, SQLite) and OpenTelemetry traces may contain adversarial prompts, target system prompts, tool schemas, and policy violation evidence.

- Treat audit reports as sensitive artifacts.
- Do not commit generated scan databases or reports to version control.

---

## Reporting a Vulnerability in AgentShield

If you discover a security vulnerability in AgentShield itself (e.g., sandbox escape, command injection in Target Discovery, or secret leakage in reporting):

1. **Do not** open a public GitHub issue with exploit details.
2. Report the issue privately to the repository maintainers (or via GitHub Security Advisories if enabled on the repository).
3. Include a clear description of the vulnerability, reproduction steps, affected components, and potential impact.

Maintainers will acknowledge receipt, investigate, and coordinate a fix before public disclosure.
