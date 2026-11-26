export type ScanStatus = "queued" | "running" | "completed" | "cancelled" | "failed";
export type TargetKind = "support_demo" | "http" | "mcp";
export type AgentVariant = "vulnerable" | "hardened";

export interface ScanStatusResponse {
  scan_id: string;
  status: ScanStatus;
  target_name: string;
  target_type: string;
  created_at: string;
  updated_at: string;
  overall_verdict: string | null;
  tests_planned: number;
  tests_executed: number;
  total_tokens_used: number;
  termination_reason: string | null;
  error: string | null;
}

export interface FindingItem {
  test_id: string;
  test_title: string;
  category: string;
  verdict: string;
  severity: string;
  confidence: string;
  summary: string;
  taxonomy: { owasp_llm: string; mitre_atlas: string };
  evaluator: string;
  evidence: string[];
  tool_call_ids: string[];
  limitations: string[];
  remediation: string;
}

export interface TraceHighlight {
  session_id: string;
  tool_call_id: string | null;
  tool_name: string;
  authorization: string;
  execution_status: string;
  state_changed: boolean | null;
  latency_ms: number | null;
}

export interface AuditReportData {
  scan_id: string;
  target_name: string;
  target_type: string;
  observability: string;
  telemetry_available: boolean;
  providers_used: string[];
  models_used: string[];
  policies_tested: string[];
  started_at: string;
  completed_at: string;
  termination_reason: string;
  overall_verdict: string;
  total_tests: number;
  completed_tests: number;
  incomplete_tests: number;
  confirmed_count: number;
  suspected_count: number;
  inconclusive_count: number;
  no_violation_count: number;
  severity_counts: Record<string, number>;
  total_tokens_used: number;
  total_turns_executed: number;
  total_latency_ms: number;
  avg_latency_ms: number;
  api_expenditure_display: string;
  findings: FindingItem[];
  trace_highlights: TraceHighlight[];
}

export interface CreateScanPayload {
  target_name: string;
  target_type: TargetKind;
  authorized: boolean;
  support_variant: AgentVariant;
  target_url?: string;
  budget: {
    max_tests: number;
    max_turns_per_test: number;
    max_total_tokens: number;
  };
}

export interface TestSuiteDescriptor {
  category: string;
  description: string;
  owasp_llm: string;
  mitre_atlas: string;
  applicable_to: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchTestSuites(): Promise<TestSuiteDescriptor[]> {
  const res = await fetch(`${API_BASE}/api/v1/test-suites`);
  if (!res.ok) throw new Error(`Failed to fetch test suites (${res.status})`);
  const data = await res.json();
  return data.suites;
}

export async function startScan(payload: CreateScanPayload): Promise<ScanStatusResponse> {
  const res = await fetch(`${API_BASE}/api/v1/scans`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to start scan (${res.status})`);
  }
  return res.json();
}

export async function fetchScanStatus(scanId: string): Promise<ScanStatusResponse> {
  const res = await fetch(`${API_BASE}/api/v1/scans/${scanId}`);
  if (!res.ok) throw new Error(`Failed to fetch status (${res.status})`);
  return res.json();
}

export async function fetchScanReport(scanId: string): Promise<AuditReportData> {
  const res = await fetch(`${API_BASE}/api/v1/scans/${scanId}/report?format=json`);
  if (!res.ok) throw new Error(`Failed to fetch report (${res.status})`);
  const data = await res.json();
  return data.report;
}

export async function cancelScan(scanId: string): Promise<ScanStatusResponse> {
  const res = await fetch(`${API_BASE}/api/v1/scans/${scanId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to cancel scan (${res.status})`);
  return res.json();
}

export function reportDownloadUrl(scanId: string, format: "json" | "html"): string {
  return `${API_BASE}/api/v1/scans/${scanId}/report?format=${format}`;
}

