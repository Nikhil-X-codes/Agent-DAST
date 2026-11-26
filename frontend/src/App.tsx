import { useEffect, useMemo, useState } from "react";
import {
  AgentVariant,
  AuditReportData,
  cancelScan,
  fetchScanReport,
  fetchScanStatus,
  fetchTestSuites,
  reportDownloadUrl,
  ScanStatusResponse,
  startScan,
  TargetKind,
  TestSuiteDescriptor,
} from "./api";

type NavTab = "scan" | "findings" | "traces" | "benchmark" | "suites";

const VERDICT_STYLE: Record<string, string> = {
  confirmed: "border-red-500/30 bg-red-500/10 text-red-300",
  suspected: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  inconclusive: "border-zinc-600/40 bg-zinc-700/20 text-zinc-300",
  no_violation_observed: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
};

const SEVERITY_BAR: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-amber-400",
  low: "bg-zinc-500",
};

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>("scan");

  // Scan configuration
  const [targetName, setTargetName] = useState("support-agent");
  const [targetType, setTargetType] = useState<TargetKind>("support_demo");
  const [variant, setVariant] = useState<AgentVariant>("vulnerable");
  const [targetUrl, setTargetUrl] = useState("");
  const [authorized, setAuthorized] = useState(true);
  const [maxTests, setMaxTests] = useState(3);
  const [maxTurns, setMaxTurns] = useState(2);
  const [maxTokens, setMaxTokens] = useState(15000);

  // Runtime state
  const [suites, setSuites] = useState<TestSuiteDescriptor[]>([]);
  const [scanHistory, setScanHistory] = useState<ScanStatusResponse[]>([]);
  const [activeScan, setActiveScan] = useState<ScanStatusResponse | null>(null);
  const [report, setReport] = useState<AuditReportData | null>(null);
  const [verdictFilter, setVerdictFilter] = useState<string>("all");
  const [vulnCompare, setVulnCompare] = useState<AuditReportData | null>(null);
  const [hardenedCompare, setHardenedCompare] = useState<AuditReportData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchTestSuites()
      .then(setSuites)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (!activeScan || !["queued", "running"].includes(activeScan.status)) return;
    const timer = setInterval(async () => {
      try {
        const updated = await fetchScanStatus(activeScan.scan_id);
        setActiveScan(updated);
        setScanHistory((prev) => prev.map((s) => (s.scan_id === updated.scan_id ? updated : s)));
        if (updated.status === "completed") {
          const rep = await fetchScanReport(updated.scan_id);
          setReport(rep);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [activeScan]);

  async function selectHistoricalScan(scan: ScanStatusResponse) {
    setError(null);
    setActiveScan(scan);
    if (scan.status === "completed") {
      try {
        const rep = await fetchScanReport(scan.scan_id);
        setReport(rep);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } else {
      setReport(null);
    }
  }

  async function handleStartScan(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setReport(null);
    setBusy(true);
    try {
      const created = await startScan({
        target_name: targetName,
        target_type: targetType,
        authorized,
        support_variant: variant,
        target_url: targetType === "support_demo" ? undefined : targetUrl,
        budget: { max_tests: maxTests, max_turns_per_test: maxTurns, max_total_tokens: maxTokens },
      });
      setActiveScan(created);
      setScanHistory((prev) => [created, ...prev.slice(0, 9)]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function pollUntilDone(scanId: string): Promise<AuditReportData> {
    while (true) {
      const st = await fetchScanStatus(scanId);
      if (st.status === "completed") return fetchScanReport(scanId);
      if (st.status === "failed" || st.status === "cancelled") {
        throw new Error(st.error || `Scan ${scanId} ended with status ${st.status}`);
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  async function handleRunComparison() {
    setError(null);
    setBusy(true);
    setActiveTab("benchmark");
    try {
      const [vScan, hScan] = await Promise.all([
        startScan({
          target_name: "benchmark-vulnerable",
          target_type: "support_demo",
          authorized: true,
          support_variant: "vulnerable",
          budget: { max_tests: maxTests, max_turns_per_test: maxTurns, max_total_tokens: maxTokens },
        }),
        startScan({
          target_name: "benchmark-hardened",
          target_type: "support_demo",
          authorized: true,
          support_variant: "hardened",
          budget: { max_tests: maxTests, max_turns_per_test: maxTurns, max_total_tokens: maxTokens },
        }),
      ]);
      const [vRep, hRep] = await Promise.all([pollUntilDone(vScan.scan_id), pollUntilDone(hScan.scan_id)]);
      setVulnCompare(vRep);
      setHardenedCompare(hRep);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const filteredFindings = useMemo(() => {
    if (!report) return [];
    if (verdictFilter === "all") return report.findings;
    return report.findings.filter((f) => f.verdict === verdictFilter);
  }, [report, verdictFilter]);

  const maxSev = report
    ? Math.max(
        report.severity_counts.critical ?? 0,
        report.severity_counts.high ?? 0,
        report.severity_counts.medium ?? 0,
        report.severity_counts.low ?? 0,
        1
      )
    : 1;

  const progressPct =
    activeScan && activeScan.tests_planned > 0
      ? Math.round((activeScan.tests_executed / activeScan.tests_planned) * 100)
      : activeScan?.status === "completed"
        ? 100
        : 0;

  return (
    <div className="min-h-screen bg-[#090a0d] text-zinc-100 flex flex-col md:flex-row">
      {/* Sidebar */}
      <aside className="w-full md:w-64 shrink-0 border-b md:border-b-0 md:border-r border-zinc-800/80 bg-[#0d0f14] flex flex-col justify-between p-4">
        <div className="space-y-6">
          <div className="flex items-center justify-between px-2">
            <div>
              <div className="text-sm font-semibold tracking-tight text-white">AgentShield</div>
              <div className="text-[11px] text-zinc-500">AI Agent DAST Workbench</div>
            </div>
            <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
              ₹0 Mode
            </span>
          </div>

          <nav className="space-y-1 text-sm">
            {(
              [
                ["scan", "Overview & Scan"],
                ["findings", `Findings${report ? ` (${report.findings.length})` : ""}`],
                ["traces", `Trace Explorer${report ? ` (${report.trace_highlights.length})` : ""}`],
                ["benchmark", "Benchmark"],
                ["suites", `Test Suites (${suites.length})`],
              ] as [NavTab, string][]
            ).map(([tab, label]) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveTab(tab)}
                className={`w-full text-left px-3 py-2 rounded-lg transition-colors text-xs font-medium ${
                  activeTab === tab
                    ? "bg-zinc-800/90 text-white border border-zinc-700/70"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          {scanHistory.length > 0 && (
            <div className="pt-4 border-t border-zinc-800/80">
              <div className="px-2 mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                Session Scans
              </div>
              <div className="space-y-1 max-h-52 overflow-y-auto">
                {scanHistory.map((s) => (
                  <button
                    key={s.scan_id}
                    type="button"
                    onClick={() => selectHistoricalScan(s)}
                    className={`w-full text-left px-2.5 py-1.5 rounded-md text-xs transition-colors ${
                      activeScan?.scan_id === s.scan_id
                        ? "bg-zinc-800 text-zinc-100"
                        : "text-zinc-400 hover:bg-zinc-800/40"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-medium">{s.target_name}</span>
                      <span className="text-[10px] font-mono text-zinc-500">{s.status}</span>
                    </div>
                    <div className="text-[10px] font-mono text-zinc-500 truncate">{s.scan_id}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="hidden md:block pt-4 border-t border-zinc-800/80 px-2 text-[11px] text-zinc-500">
          Authorized pre-production auditing only.
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 min-w-0 p-5 md:p-7 max-w-6xl space-y-5">
        {/* Top Status Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-white">
              {activeTab === "scan" && "AgentShield Security Audit"}
              {activeTab === "findings" && "Vulnerability Findings & Remediation"}
              {activeTab === "traces" && "Tool-Call Trace Explorer"}
              {activeTab === "benchmark" && "Vulnerable vs. Hardened Benchmark"}
              {activeTab === "suites" && "Available Security Test Suites"}
            </h1>
            <p className="text-xs text-zinc-400">
              {activeScan
                ? `Active scan ${activeScan.scan_id} (${activeScan.target_type}) • ${activeScan.status}`
                : "Configure an authorized target and launch a bounded security scan."}
            </p>
          </div>

          {activeScan?.status === "completed" && (
            <div className="flex items-center gap-2">
              <a
                href={reportDownloadUrl(activeScan.scan_id, "json")}
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-200"
              >
                Export JSON
              </a>
              <a
                href={reportDownloadUrl(activeScan.scan_id, "html")}
                target="_blank"
                rel="noreferrer"
                className="rounded-md bg-emerald-600 hover:bg-emerald-500 px-3 py-1.5 text-xs font-medium text-white"
              >
                Export HTML Report
              </a>
            </div>
          )}
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2.5 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Active Scan Status Strip */}
        {activeScan && (
          <section className="rounded-lg border border-zinc-800 bg-[#111318] p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-mono text-zinc-300">{activeScan.scan_id}</span>
                <span
                  className={`rounded border px-2 py-0.5 font-mono text-[11px] ${
                    activeScan.status === "completed"
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                      : activeScan.status === "failed"
                        ? "border-red-500/30 bg-red-500/10 text-red-300"
                        : "border-blue-500/30 bg-blue-500/10 text-blue-300"
                  }`}
                >
                  {activeScan.status}
                </span>
                <span className="text-zinc-400 tabular-nums">
                  Tests: <strong className="text-zinc-200">{activeScan.tests_executed} / {activeScan.tests_planned}</strong>
                </span>
                <span className="text-zinc-400 tabular-nums">
                  Tokens: <strong className="text-zinc-200">{activeScan.total_tokens_used}</strong>
                </span>
                {activeScan.termination_reason && (
                  <span className="text-zinc-500 font-mono">{activeScan.termination_reason}</span>
                )}
              </div>

              {["queued", "running"].includes(activeScan.status) && (
                <button
                  type="button"
                  onClick={() => cancelScan(activeScan.scan_id).then(setActiveScan)}
                  className="rounded border border-red-500/40 px-2.5 py-1 text-[11px] text-red-300 hover:bg-red-500/10"
                >
                  Cancel Scan
                </button>
              )}
            </div>

            <div className="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${
                  activeScan.status === "failed"
                    ? "bg-red-500"
                    : activeScan.status === "completed"
                      ? "bg-emerald-500"
                      : "bg-blue-500"
                }`}
                style={{ width: `${progressPct}%` }}
              />
            </div>

            {activeScan.status === "failed" && activeScan.error && (
              <div className="rounded border border-red-500/30 bg-red-950/30 px-3 py-2 text-xs text-red-300 font-mono">
                {activeScan.error}
              </div>
            )}
          </section>
        )}

        {/* TAB 1: Overview & Scan */}
        {activeTab === "scan" && (
          <div className="space-y-5">
            <section className="rounded-lg border border-zinc-800 bg-[#111318] p-5">
              <form onSubmit={handleStartScan} className="space-y-4 text-xs">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="target-name" className="block text-zinc-400 mb-1.5 font-medium">
                      Target Name
                    </label>
                    <input
                      id="target-name"
                      name="target_name"
                      className="w-full rounded-md border border-zinc-700/80 bg-[#090a0d] px-3 py-2 text-zinc-100 focus:border-blue-500 focus:outline-none"
                      value={targetName}
                      onChange={(e) => setTargetName(e.target.value)}
                    />
                  </div>

                  <div>
                    <span className="block text-zinc-400 mb-1.5 font-medium">Target Type</span>
                    <div className="grid grid-cols-3 rounded-md border border-zinc-700/80 bg-[#090a0d] p-0.5">
                      {(
                        [
                          ["support_demo", "Demo"],
                          ["http", "HTTP"],
                          ["mcp", "MCP"],
                        ] as [TargetKind, string][]
                      ).map(([kind, label]) => (
                        <button
                          key={kind}
                          type="button"
                          onClick={() => setTargetType(kind)}
                          className={`rounded py-1.5 text-center font-medium transition-colors ${
                            targetType === kind
                              ? "bg-zinc-800 text-white"
                              : "text-zinc-400 hover:text-zinc-200"
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {targetType === "support_demo" ? (
                    <div>
                      <label htmlFor="demo-variant" className="block text-zinc-400 mb-1.5 font-medium">
                        Demo Variant
                      </label>
                      <select
                        id="demo-variant"
                        name="support_variant"
                        className="w-full rounded-md border border-zinc-700/80 bg-[#090a0d] px-3 py-2 text-zinc-100 focus:border-blue-500 focus:outline-none"
                        value={variant}
                        onChange={(e) => setVariant(e.target.value as AgentVariant)}
                      >
                        <option value="vulnerable">vulnerable (weak authz)</option>
                        <option value="hardened">hardened (enforced authz)</option>
                      </select>
                    </div>
                  ) : (
                    <div>
                      <label htmlFor="target-url" className="block text-zinc-400 mb-1.5 font-medium">
                        Target URL
                      </label>
                      <input
                        id="target-url"
                        name="target_url"
                        className="w-full rounded-md border border-zinc-700/80 bg-[#090a0d] px-3 py-2 font-mono text-zinc-100 focus:border-blue-500 focus:outline-none"
                        placeholder="http://127.0.0.1:8001"
                        value={targetUrl}
                        onChange={(e) => setTargetUrl(e.target.value)}
                      />
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label htmlFor="max-tests" className="block text-zinc-400 mb-1.5 font-medium">
                      Max Tests
                    </label>
                    <input
                      id="max-tests"
                      name="max_tests"
                      type="number"
                      min={1}
                      max={20}
                      className="w-full rounded-md border border-zinc-700/80 bg-[#090a0d] px-3 py-2 tabular-nums text-zinc-100"
                      value={maxTests}
                      onChange={(e) => setMaxTests(Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label htmlFor="max-turns" className="block text-zinc-400 mb-1.5 font-medium">
                      Max Turns / Test
                    </label>
                    <input
                      id="max-turns"
                      name="max_turns"
                      type="number"
                      min={1}
                      max={5}
                      className="w-full rounded-md border border-zinc-700/80 bg-[#090a0d] px-3 py-2 tabular-nums text-zinc-100"
                      value={maxTurns}
                      onChange={(e) => setMaxTurns(Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label htmlFor="max-tokens" className="block text-zinc-400 mb-1.5 font-medium">
                      Token Budget
                    </label>
                    <input
                      id="max-tokens"
                      name="max_tokens"
                      type="number"
                      min={500}
                      className="w-full rounded-md border border-zinc-700/80 bg-[#090a0d] px-3 py-2 tabular-nums text-zinc-100"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                    />
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-zinc-800/80">
                  <label htmlFor="authz-confirm" className="flex items-center gap-2 text-zinc-300 cursor-pointer">
                    <input
                      id="authz-confirm"
                      name="authorized"
                      type="checkbox"
                      checked={authorized}
                      onChange={(e) => setAuthorized(e.target.checked)}
                    />
                    I confirm explicit authorization to audit this target
                  </label>
                  <div className="flex gap-2.5">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={handleRunComparison}
                      className="rounded-md border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 disabled:opacity-50 px-3.5 py-2 font-medium text-zinc-200"
                    >
                      Compare Vulnerable vs. Hardened
                    </button>
                    <button
                      type="submit"
                      disabled={busy}
                      className="rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-4 py-2 font-medium text-white"
                    >
                      Start Security Scan
                    </button>
                  </div>
                </div>
              </form>
            </section>

            {report && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="rounded-lg border border-zinc-800 bg-[#111318] p-4">
                    <div className="text-[11px] text-zinc-400">Security tests</div>
                    <div className="text-2xl font-semibold tabular-nums mt-1">
                      {report.completed_tests} / {report.total_tests}
                    </div>
                  </div>
                  <div className="rounded-lg border border-zinc-800 bg-[#111318] p-4">
                    <div className="text-[11px] text-zinc-400">Confirmed findings</div>
                    <div className="text-2xl font-semibold tabular-nums text-red-400 mt-1">
                      {report.confirmed_count}
                    </div>
                  </div>
                  <div className="rounded-lg border border-zinc-800 bg-[#111318] p-4">
                    <div className="text-[11px] text-zinc-400">No violation observed</div>
                    <div className="text-2xl font-semibold tabular-nums text-emerald-400 mt-1">
                      {report.no_violation_count}
                    </div>
                  </div>
                  <div className="rounded-lg border border-zinc-800 bg-[#111318] p-4">
                    <div className="text-[11px] text-zinc-400">Inconclusive</div>
                    <div className="text-2xl font-semibold tabular-nums text-amber-300 mt-1">
                      {report.inconclusive_count}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <section className="rounded-lg border border-zinc-800 bg-[#111318] p-4">
                    <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-3">
                      Findings by severity
                    </h2>
                    {(["critical", "high", "medium", "low"] as const).map((sev) => {
                      const count = report.severity_counts[sev] ?? 0;
                      const pct = Math.round((count / maxSev) * 100);
                      return (
                        <div key={sev} className="grid grid-cols-[72px_1fr_28px] items-center gap-3 mb-2 text-xs">
                          <span className="capitalize text-zinc-300">{sev}</span>
                          <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                            <div className={`h-full rounded-full ${SEVERITY_BAR[sev]}`} style={{ width: `${pct}%` }} />
                          </div>
                          <span className="text-right font-mono tabular-nums">{count}</span>
                        </div>
                      );
                    })}
                    <div className="mt-3 pt-3 border-t border-zinc-800 flex justify-between text-xs">
                      <span className="text-zinc-400">API expenditure</span>
                      <strong className="text-emerald-300">{report.api_expenditure_display}</strong>
                    </div>
                  </section>

                  <section className="rounded-lg border border-zinc-800 bg-[#111318] p-4">
                    <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-3">
                      Provider &amp; Resource Telemetry
                    </h2>
                    <dl className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <dt className="text-zinc-500">Providers</dt>
                        <dd className="font-mono text-zinc-200 mt-0.5">
                          {report.providers_used.join(", ") || "N/A"}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-zinc-500">Models</dt>
                        <dd className="font-mono text-zinc-200 mt-0.5 truncate">
                          {report.models_used.join(", ") || "N/A"}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-zinc-500">Total Tokens</dt>
                        <dd className="font-mono tabular-nums text-zinc-200 mt-0.5">
                          {report.total_tokens_used}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-zinc-500">Avg / Total Latency</dt>
                        <dd className="font-mono tabular-nums text-zinc-200 mt-0.5">
                          {report.avg_latency_ms} ms / {report.total_latency_ms} ms
                        </dd>
                      </div>
                    </dl>
                  </section>
                </div>
              </>
            )}
          </div>
        )}

        {/* TAB 2: Findings */}
        {activeTab === "findings" && (
          <section className="rounded-lg border border-zinc-800 bg-[#111318] p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex gap-1.5 text-xs">
                {["all", "confirmed", "suspected", "inconclusive", "no_violation_observed"].map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setVerdictFilter(v)}
                    className={`rounded-md px-2.5 py-1 font-medium transition-colors ${
                      verdictFilter === v
                        ? "bg-zinc-800 text-white"
                        : "text-zinc-400 hover:bg-zinc-800/40"
                    }`}
                  >
                    {v === "all" ? "All" : v}
                  </button>
                ))}
              </div>
            </div>

            {!report ? (
              <p className="text-xs text-zinc-400">Run a scan to inspect measured vulnerability findings.</p>
            ) : filteredFindings.length === 0 ? (
              <p className="text-xs text-zinc-400">No findings match the selected filter.</p>
            ) : (
              <div className="space-y-3">
                {filteredFindings.map((f, i) => (
                  <div
                    key={`${f.test_id}-${f.category}-${i}`}
                    className="rounded-lg border border-zinc-800 bg-[#090a0d] p-4 space-y-2 text-xs"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <span className="font-semibold text-zinc-100">{f.test_title}</span>
                        <span className="ml-2 font-mono text-zinc-500">{f.category}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`rounded border px-2 py-0.5 font-mono text-[11px] ${VERDICT_STYLE[f.verdict] || ""}`}>
                          {f.verdict}
                        </span>
                        <span className="text-zinc-400 font-mono">
                          {f.severity} / {f.confidence} conf.
                        </span>
                      </div>
                    </div>
                    <p className="text-zinc-300">{f.summary}</p>
                    <div className="text-zinc-500">
                      {f.taxonomy.owasp_llm} &bull; {f.taxonomy.mitre_atlas}
                    </div>
                    {f.evidence.length > 0 && (
                      <div className="rounded bg-zinc-900/80 border border-zinc-800 px-3 py-2 font-mono text-[11px] text-zinc-300">
                        {f.evidence.join(" | ")}
                      </div>
                    )}
                    <div className="text-zinc-400">
                      <strong className="text-zinc-300">Remediation:</strong> {f.remediation}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* TAB 3: Trace Explorer */}
        {activeTab === "traces" && (
          <section className="rounded-lg border border-zinc-800 bg-[#111318] p-5 overflow-x-auto">
            {!report ? (
              <p className="text-xs text-zinc-400">Run a scan to inspect runtime tool-call traces.</p>
            ) : report.trace_highlights.length === 0 ? (
              <p className="text-xs text-zinc-400">
                No tool-call traces recorded (either black-box target or no tools invoked).
              </p>
            ) : (
              <table className="w-full text-left text-xs font-mono">
                <thead className="text-zinc-500 border-b border-zinc-800">
                  <tr>
                    <th className="py-2 pr-4">Session</th>
                    <th className="py-2 pr-4">Tool</th>
                    <th className="py-2 pr-4">Authorization</th>
                    <th className="py-2 pr-4">Status</th>
                    <th className="py-2 pr-4">State Changed</th>
                    <th className="py-2">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/80">
                  {report.trace_highlights.map((tr, idx) => (
                    <tr key={`${tr.session_id}-${idx}`}>
                      <td className="py-2.5 pr-4 text-zinc-400">{tr.session_id}</td>
                      <td className="py-2.5 pr-4 text-blue-300">{tr.tool_name}</td>
                      <td className="py-2.5 pr-4">
                        <span
                          className={
                            tr.authorization === "missing"
                              ? "text-red-400"
                              : tr.authorization === "denied"
                                ? "text-emerald-400"
                                : "text-zinc-300"
                          }
                        >
                          {tr.authorization}
                        </span>
                      </td>
                      <td className="py-2.5 pr-4">{tr.execution_status}</td>
                      <td className="py-2.5 pr-4">{String(tr.state_changed)}</td>
                      <td className="py-2.5 tabular-nums">{tr.latency_ms ?? 0} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )}

        {/* TAB 4: Benchmark */}
        {activeTab === "benchmark" && (
          <section className="rounded-lg border border-zinc-800 bg-[#111318] p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-xs text-zinc-400">
                Runs equivalent security scans against the vulnerable and hardened demo agents.
              </p>
              <button
                type="button"
                disabled={busy}
                onClick={handleRunComparison}
                className="rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-3.5 py-1.5 text-xs font-medium text-white"
              >
                {busy ? "Running Comparison..." : "Run Vulnerable vs. Hardened"}
              </button>
            </div>

            {vulnCompare && hardenedCompare ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {[vulnCompare, hardenedCompare].map((rep) => (
                  <div key={rep.scan_id} className="rounded-lg border border-zinc-800 bg-[#090a0d] p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-sm text-white">{rep.target_name}</span>
                      <span className={`rounded border px-2 py-0.5 font-mono text-[11px] ${VERDICT_STYLE[rep.overall_verdict] || ""}`}>
                        {rep.overall_verdict}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 pt-2 tabular-nums">
                      <div className="text-zinc-400">Confirmed Findings</div>
                      <div className="font-mono text-right">{rep.confirmed_count}</div>
                      <div className="text-zinc-400">No Violation Observed</div>
                      <div className="font-mono text-right">{rep.no_violation_count}</div>
                      <div className="text-zinc-400">Total Tokens</div>
                      <div className="font-mono text-right">{rep.total_tokens_used}</div>
                      <div className="text-zinc-400">Avg Latency</div>
                      <div className="font-mono text-right">{rep.avg_latency_ms} ms</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-zinc-500">Click "Run Vulnerable vs. Hardened" to compare variants.</p>
            )}
          </section>
        )}

        {/* TAB 5: Test Suites */}
        {activeTab === "suites" && (
          <section className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            {suites.map((s) => (
              <div key={s.category} className="rounded-lg border border-zinc-800 bg-[#111318] p-4 space-y-1.5">
                <div className="font-mono text-blue-300 font-medium">{s.category}</div>
                <div className="text-zinc-300">{s.description}</div>
                <div className="text-zinc-500 pt-1">
                  {s.owasp_llm} &bull; {s.mitre_atlas}
                </div>
              </div>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}
