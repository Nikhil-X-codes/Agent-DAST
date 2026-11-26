"""AgentShield Typer + Rich CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentshield.evaluation import EvaluationPolicy
from agentshield.harnesses import LangGraphHarness, TargetConfig
from agentshield.orchestrator import ScanBudget, ScanOrchestrator
from agentshield.providers import FreeProviderRouter, ProviderError
from agentshield.reporting import (
    BenchmarkRunner,
    build_audit_report,
    render_cli_report,
)
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.tracing import TraceStore

app = typer.Typer(name="agentshield", help="Zero-cost AI agent security auditing framework.")
console = Console()


@app.command()
def demo(
    prompt: str = typer.Option(
        "Check order ord_1001 and summarize its status.",
        "--prompt",
        "-p",
        help="User prompt sent to the target customer support agent.",
    ),
    variant: Literal["vulnerable", "hardened", "both"] = typer.Option(
        "both",
        "--variant",
        "-v",
        help="Target agent variant to run.",
    ),
    customer: str = typer.Option(
        "cust_alice",
        "--customer",
        "-c",
        help="Authenticated customer ID bound to the session.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Preferred free provider (groq | gemini | openrouter).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Optional free-tier model override.",
    ),
) -> None:
    """Run the LangGraph + SQLite customer support target agent with real LLM inference."""
    load_dotenv(override=False)
    preferred = provider or os.getenv("AGENTSHIELD_PROVIDER") or None
    model_override = model or os.getenv("AGENTSHIELD_MODEL") or None
    order = [preferred] if preferred else None

    try:
        router = FreeProviderRouter.from_env(preferred_order=order)
    except ProviderError as exc:
        console.print(f"[bold red]Provider configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    variants: list[Literal["vulnerable", "hardened"]] = (
        ["vulnerable", "hardened"] if variant == "both" else [variant]
    )

    with router:
        for current_variant in variants:
            with SupportSandboxDB() as db:
                agent = CustomerSupportAgent(
                    router,
                    db,
                    variant=current_variant,
                    authenticated_customer_id=customer,
                    model=model_override,
                )
                try:
                    result = agent.run(prompt)
                except ProviderError as exc:
                    console.print(f"[bold red]Inference failed ({current_variant}):[/bold red] {exc}")
                    raise typer.Exit(code=1) from exc

            console.print(
                Panel(
                    result.final_response or "(empty response)",
                    title=f"[bold]{current_variant.upper()}[/bold] | {result.provider}:{result.model} | "
                    f"{result.latency_ms:.1f} ms | {result.usage.total_tokens} tokens",
                )
            )
            if result.tool_executions:
                table = Table(title=f"Tool Executions ({current_variant})")
                table.add_column("Tool")
                table.add_column("Arguments")
                table.add_column("Authorized")
                table.add_column("Status")
                for rec in result.tool_executions:
                    table.add_row(
                        rec.tool_name,
                        str(rec.arguments),
                        "yes" if rec.authorized else "[red]no[/red]",
                        rec.status,
                    )
                console.print(table)


@app.command()
def scan(
    variant: Literal["vulnerable", "hardened"] = typer.Option("hardened", "--variant", "-v"),
    customer: str = typer.Option("cust_alice", "--customer", "-c"),
    max_tests: int = typer.Option(3, "--max-tests"),
    max_turns: int = typer.Option(2, "--max-turns"),
    report_dir: Path = typer.Option(Path("reports"), "--report-dir"),
    sarif: bool = typer.Option(False, "--sarif", help="Also write a SARIF 2.1.0 report alongside HTML & JSON."),
    fail_on_confirmed: bool = typer.Option(
        False,
        "--fail-on-confirmed",
        help="Exit with code 1 if any CONFIRMED critical/high finding is detected (CI gate).",
    ),
    regressions_db: Path | None = typer.Option(
        None,
        "--regressions-db",
        help="Optional SQLite path to save confirmed/suspected findings as replayable regressions.",
    ),
) -> None:
    """Run a bounded security scan and automatically generate HTML & JSON reports."""
    from agentshield.reporting.models import has_confirmed_critical_or_high
    from agentshield.reporting.regression import RegressionStore, create_regressions_from_scan_report
    from agentshield.reporting.sarif_report import write_sarif_report

    load_dotenv(override=False)
    try:
        router = FreeProviderRouter.from_env()
    except ProviderError as exc:
        console.print(f"[bold red]Provider configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    policy = EvaluationPolicy(
        authenticated_subject=customer,
        foreign_identifiers=["ord_2001", "cust_bob"],
    )
    with router, SupportSandboxDB() as db, TraceStore(":memory:") as store:
        agent = CustomerSupportAgent(router, db, variant=variant, authenticated_customer_id=customer)
        harness = LangGraphHarness(agent, TargetConfig(target_name=f"support-{variant}", authorized=True))
        orchestrator = ScanOrchestrator(
            harness,
            policy,
            budget=ScanBudget(max_tests=max_tests, max_turns_per_test=max_turns),
            trace_store=store,
        )
        scan_report = orchestrator.run(report_dir=report_dir)
        audit_report = build_audit_report(scan_report, trace_store=store)
        render_cli_report(audit_report, console=console)
        console.print(f"[green]HTML report saved:[/green] {report_dir / f'{scan_report.scan_id}.html'}")
        console.print(f"[green]JSON report saved:[/green] {report_dir / f'{scan_report.scan_id}.json'}")

        if sarif:
            sarif_path = report_dir / f"{scan_report.scan_id}.sarif"
            write_sarif_report(audit_report, sarif_path)
            console.print(f"[green]SARIF report saved:[/green] {sarif_path}")

        if regressions_db is not None:
            recs = create_regressions_from_scan_report(scan_report)
            if recs:
                with RegressionStore(regressions_db) as reg_store:
                    for rec in recs:
                        reg_store.save(rec)
                console.print(f"[yellow]Saved {len(recs)} regression case(s) to:[/yellow] {regressions_db}")

        if fail_on_confirmed and has_confirmed_critical_or_high(audit_report):
            console.print("[bold red]CI GATE FAILED:[/bold red] Confirmed critical/high findings detected.")
            raise typer.Exit(code=1)


@app.command()
def benchmark(
    trials: int = typer.Option(1, "--trials", "-t"),
    max_tests: int = typer.Option(2, "--max-tests"),
) -> None:
    """Compare vulnerable vs. hardened real agents under equivalent conditions."""
    load_dotenv(override=False)
    try:
        router = FreeProviderRouter.from_env()
    except ProviderError as exc:
        console.print(f"[bold red]Provider configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )
    with router:
        runner = BenchmarkRunner(
            router,
            policy,
            budget=ScanBudget(max_tests=max_tests, max_turns_per_test=2),
        )
        comp = runner.run(trials=trials)

    table = Table(title=f"AgentShield Comparative Benchmark ({trials} trial(s))")
    table.add_column("Variant")
    table.add_column("Tests Executed")
    table.add_column("Confirmed Findings")
    table.add_column("Violation Rate")
    table.add_column("Total Tokens")
    table.add_column("Avg Latency (ms)")
    for s in (comp.vulnerable, comp.hardened):
        table.add_row(
            s.variant,
            str(s.total_tests_executed),
            str(s.total_confirmed_findings),
            f"{s.confirmed_violation_rate:.2%}",
            str(s.total_tokens_used),
            f"{s.avg_latency_ms:.1f}",
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    db_path: Path = typer.Option(Path("agentshield_api.sqlite"), "--db-path"),
    allow_localhost_targets: bool = typer.Option(False, "--allow-localhost-targets"),
) -> None:
    """Start the local AgentShield FastAPI service."""
    import uvicorn

    from agentshield.api import create_app

    load_dotenv(override=False)
    api_app = create_app(db_path=db_path, allow_localhost_targets=allow_localhost_targets)
    uvicorn.run(api_app, host=host, port=port)


@app.command()
def report(
    json_path: Path = typer.Argument(..., help="Path to an existing scan JSON report."),
    sarif: bool = typer.Option(False, "--sarif", help="Export SARIF 2.1.0 format alongside HTML."),
    out_dir: Path = typer.Option(Path("reports"), "--out-dir", help="Output directory for exported files."),
    fail_on_confirmed: bool = typer.Option(
        False,
        "--fail-on-confirmed",
        help="Exit with code 1 if any CONFIRMED critical/high finding exists (CI gate).",
    ),
) -> None:
    """Convert an existing scan JSON report to SARIF and/or check the CI failure gate."""
    import json

    from agentshield.reporting.models import AuditReportData, get_confirmed_critical_or_high
    from agentshield.reporting.sarif_report import write_sarif_report

    if not json_path.exists():
        console.print(f"[bold red]File not found:[/bold red] {json_path}")
        raise typer.Exit(code=1)

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        audit = AuditReportData.model_validate(raw)
    except Exception as exc:
        console.print(f"[bold red]Failed to parse report:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    if sarif:
        sarif_path = out_dir / f"{audit.scan_id}.sarif"
        write_sarif_report(audit, sarif_path)
        console.print(f"[green]SARIF report saved:[/green] {sarif_path}")

    confirmed_critical_high = get_confirmed_critical_or_high(audit)

    if confirmed_critical_high:
        table = Table(title="Confirmed Critical/High Findings")
        table.add_column("Test ID")
        table.add_column("Category")
        table.add_column("Severity")
        table.add_column("Summary")
        for f in confirmed_critical_high:
            table.add_row(f.test_id, f.category, f"[bold red]{f.severity}[/bold red]", f.summary)
        console.print(table)

    if fail_on_confirmed and confirmed_critical_high:
        console.print(
            f"[bold red]CI GATE FAILED:[/bold red] {len(confirmed_critical_high)} confirmed "
            "critical/high finding(s) detected. Merge blocked."
        )
        raise typer.Exit(code=1)

    if not confirmed_critical_high:
        console.print("[green]CI gate: No confirmed critical/high findings. ✓[/green]")


@app.command()
def replay(
    scan_id: str = typer.Argument(..., help="Scan ID to load or replay regressions for."),
    db_path: Path = typer.Option(Path("regressions.sqlite"), "--db-path", help="Regression store SQLite path."),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Re-run saved regression test cases against the local support agent and update status.",
    ),
    variant: Literal["vulnerable", "hardened"] = typer.Option("hardened", "--variant", "-v"),
    customer: str = typer.Option("cust_alice", "--customer", "-c"),
) -> None:
    """List or re-run saved regression records for a scan and show before/after comparison."""
    from agentshield.reporting.regression import RegressionStore, replay_scan_regressions

    if not db_path.exists():
        console.print(f"[bold red]Regression store not found:[/bold red] {db_path}")
        raise typer.Exit(code=1)

    with RegressionStore(db_path) as store:
        if execute:
            load_dotenv(override=False)
            try:
                router = FreeProviderRouter.from_env()
            except ProviderError as exc:
                console.print(f"[bold red]Provider configuration error:[/bold red] {exc}")
                raise typer.Exit(code=1) from exc

            policy = EvaluationPolicy(
                authenticated_subject=customer,
                foreign_identifiers=["ord_2001", "cust_bob"],
            )
            with router, SupportSandboxDB() as sandbox_db:
                agent = CustomerSupportAgent(router, sandbox_db, variant=variant, authenticated_customer_id=customer)
                harness = LangGraphHarness(agent, TargetConfig(target_name=f"support-{variant}", authorized=True))
                records = replay_scan_regressions(store, scan_id, harness, policy)
        else:
            records = store.load_by_scan(scan_id)

    if not records:
        console.print(f"No regression records found for scan [bold]{scan_id}[/bold].")
        raise typer.Exit(code=0)

    table = Table(title=f"Regression Records — Scan {scan_id}")
    table.add_column("Regression ID")
    table.add_column("Test ID")
    table.add_column("Category")
    table.add_column("Original Verdict")
    table.add_column("Replay Verdict")
    table.add_column("Status")
    for rec in records:
        status_color = {
            "fixed": "[green]fixed[/green]",
            "regressed": "[bold red]regressed[/bold red]",
            "open": "[yellow]open[/yellow]",
            "inconclusive": "[dim]inconclusive[/dim]",
        }.get(rec.status, rec.status)
        table.add_row(
            rec.regression_id,
            rec.test_id,
            rec.category,
            rec.original_verdict,
            rec.replay_verdict or "(not replayed)",
            status_color,
        )
    console.print(table)


@app.command()
def manifest(
    variant: Literal["vulnerable", "hardened"] = typer.Option("hardened", "--variant", "-v"),
    customer: str = typer.Option("cust_alice", "--customer", "-c"),
    out: Path | None = typer.Option(None, "--out", help="Optional JSON output path for the manifest."),
) -> None:
    """Discover and display the capability manifest and attack-surface inventory offline."""
    import json

    from agentshield.harnesses.base import DiscoveredTool, TargetCapabilities
    from agentshield.harnesses.manifest import build_manifest_from_capabilities
    from agentshield.targets.support.tools import TOOL_DEFINITIONS

    tools = [
        DiscoveredTool(
            name=t.name,
            description=t.description,
            input_schema=t.parameters,
            permissions=[f"variant:{variant}"],
            source="instrumented",
        )
        for t in TOOL_DEFINITIONS
    ]
    caps = TargetCapabilities(
        target_type="langgraph",
        target_name=f"support-{variant}",
        observability="instrumented",
        telemetry_available=True,
        session_handling="session_bound",
        tools=tools,
        declared_permissions=[f"authenticated_customer:{customer}"],
        metadata={"variant": variant},
    )

    manifest_obj = build_manifest_from_capabilities(caps)
    summary = manifest_obj.attack_surface_summary()

    table = Table(title=f"Capability Manifest — {manifest_obj.target_name}")
    table.add_column("Tool")
    table.add_column("Access Mode")
    table.add_column("Data Classes")
    table.add_column("Ownership Required")
    table.add_column("Risk Level")
    for entry in manifest_obj.tools:
        table.add_row(
            entry.name,
            entry.access_mode,
            ", ".join(entry.data_classifications),
            "yes" if entry.requires_ownership_check else "no",
            entry.risk_level,
        )
    console.print(table)

    console.print(Panel(
        json.dumps(summary, indent=2),
        title="Attack Surface Inventory",
    ))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest_obj.model_dump(), indent=2), encoding="utf-8")
        console.print(f"[green]Manifest saved:[/green] {out}")


if __name__ == "__main__":
    app()
