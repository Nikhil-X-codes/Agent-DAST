"""Standalone Jinja2 HTML security audit report generator."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, select_autoescape

from agentshield.reporting.models import AuditReportData

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AgentShield Security Audit — {{ report.scan_id }}</title>
  <style>
    :root {
      --bg: #0f1115;
      --card: #171a21;
      --border: #2a2f3a;
      --text: #e6edf3;
      --muted: #8b949e;
      --red: #f85149;
      --orange: #f0883e;
      --amber: #d29922;
      --slate: #6e7681;
      --green: #3fb950;
      --blue: #58a6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 28px;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
    }
    .container { max-width: 1080px; margin: 0 auto; }
    .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 20px; }
    .header h1 { margin: 0; font-size: 22px; }
    .meta { color: var(--muted); font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 10px; padding: 16px;
    }
    .card-label { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
    .card-value { font-size: 28px; font-weight: 700; }
    .val-red { color: var(--red); }
    .val-green { color: var(--green); }
    .val-amber { color: var(--amber); }
    .section { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 18px; }
    .section h2 { margin: 0 0 14px 0; font-size: 16px; }
    .sev-row { display: grid; grid-template-columns: 90px 1fr 40px; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 13px; }
    .bar-bg { background: #222631; border-radius: 999px; height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; }
    .fill-critical { background: var(--red); }
    .fill-high { background: var(--orange); }
    .fill-medium { background: var(--amber); }
    .fill-low { background: var(--slate); }
    .kv-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 13px; }
    .kv-item strong { color: var(--muted); display: block; font-weight: 500; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { color: var(--muted); font-weight: 600; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; background: #262c38; }
    .badge-confirmed { background: rgba(248, 81, 73, 0.2); color: var(--red); }
    .badge-suspected { background: rgba(210, 153, 34, 0.2); color: var(--amber); }
    .badge-inconclusive { background: rgba(110, 118, 129, 0.25); color: var(--text); }
    .badge-no_violation_observed { background: rgba(63, 185, 80, 0.2); color: var(--green); }
    .footer-bar { display: flex; justify-content: space-between; font-size: 14px; padding-top: 10px; border-top: 1px solid var(--border); margin-top: 12px; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background: #10131a; padding: 2px 5px; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>AgentShield Security Audit</h1>
        <div class="meta">Scan ID: <code>{{ report.scan_id }}</code> &bull; Target: <strong>{{ report.target_name }}</strong> ({{ report.target_type }} / {{ report.observability }})</div>
      </div>
      <div class="meta">{{ report.started_at }} &rarr; {{ report.completed_at }}</div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-label">Security tests</div>
        <div class="card-value">{{ report.completed_tests }} / {{ report.total_tests }}</div>
        <div class="meta">Incomplete: {{ report.incomplete_tests }}</div>
      </div>
      <div class="card">
        <div class="card-label">Confirmed findings</div>
        <div class="card-value val-red">{{ report.confirmed_count }}</div>
        <div class="meta">Suspected: {{ report.suspected_count }}</div>
      </div>
      <div class="card">
        <div class="card-label">No violation observed</div>
        <div class="card-value val-green">{{ report.no_violation_count }}</div>
        <div class="meta">Overall: {{ report.overall_verdict }}</div>
      </div>
      <div class="card">
        <div class="card-label">Inconclusive</div>
        <div class="card-value val-amber">{{ report.inconclusive_count }}</div>
        <div class="meta">Termination: {{ report.termination_reason }}</div>
      </div>
    </div>

    <div class="section">
      <h2>Findings by severity</h2>
      {% set max_sev = [report.severity_counts.get('critical', 0), report.severity_counts.get('high', 0), report.severity_counts.get('medium', 0), report.severity_counts.get('low', 0), 1] | max %}
      {% for sev in ['critical', 'high', 'medium', 'low'] %}
      {% set count = report.severity_counts.get(sev, 0) %}
      <div class="sev-row">
        <span>{{ sev | capitalize }}</span>
        <div class="bar-bg">
          <div class="bar-fill fill-{{ sev }}" style="width: {{ ((count / max_sev) * 100) | round(0) }}%;"></div>
        </div>
        <span>{{ count }}</span>
      </div>
      {% endfor %}
      <div class="footer-bar">
        <span>API expenditure</span>
        <strong>{{ report.api_expenditure_display }}</strong>
      </div>
    </div>

    <div class="section">
      <h2>Target, Models &amp; Execution Telemetry</h2>
      <div class="kv-grid">
        <div class="kv-item"><strong>Providers</strong>{{ report.providers_used | join(', ') or 'N/A' }}</div>
        <div class="kv-item"><strong>Models</strong>{{ report.models_used | join(', ') or 'N/A' }}</div>
        <div class="kv-item"><strong>Policies Tested</strong>{{ report.policies_tested | join(', ') or 'None' }}</div>
        <div class="kv-item"><strong>Total Tokens</strong>{{ report.total_tokens_used }}</div>
        <div class="kv-item"><strong>Total / Avg Latency</strong>{{ report.total_latency_ms }} ms / {{ report.avg_latency_ms }} ms</div>
        <div class="kv-item"><strong>Conversation Turns</strong>{{ report.total_turns_executed }}</div>
      </div>
    </div>

    <div class="section">
      <h2>Findings &amp; Remediation</h2>
      {% if report.findings %}
      <table>
        <thead>
          <tr>
            <th>Test &amp; Category</th>
            <th>Verdict / Severity</th>
            <th>Confidence &amp; Taxonomy</th>
            <th>Evidence &amp; Limitations</th>
            <th>Remediation</th>
          </tr>
        </thead>
        <tbody>
          {% for item in report.findings %}
          <tr>
            <td>
              <strong>{{ item.test_title }}</strong><br/>
              <code>{{ item.category }}</code>
            </td>
            <td>
              <span class="badge badge-{{ item.verdict }}">{{ item.verdict }}</span><br/>
              <span class="meta">Severity: {{ item.severity }}</span>
            </td>
            <td>
              <span>Confidence: <strong>{{ item.confidence }}</strong></span><br/>
              <span class="meta">{{ item.taxonomy.owasp_llm }}</span><br/>
              <span class="meta">{{ item.taxonomy.mitre_atlas }}</span>
            </td>
            <td>
              <div>{{ item.summary }}</div>
              {% for ev in item.evidence %}
              <div class="meta">&bull; {{ ev }}</div>
              {% endfor %}
              {% for lim in item.limitations %}
              <div class="meta">Limitation: {{ lim }}</div>
              {% endfor %}
            </td>
            <td>{{ item.remediation }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <div class="meta">No findings recorded.</div>
      {% endif %}
    </div>

    {% if report.trace_highlights %}
    <div class="section">
      <h2>Execution Trace Evidence</h2>
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>Tool</th>
            <th>Authorization</th>
            <th>Status</th>
            <th>State Changed</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {% for tr in report.trace_highlights %}
          <tr>
            <td><code>{{ tr.session_id }}</code></td>
            <td><code>{{ tr.tool_name }}</code></td>
            <td>{{ tr.authorization }}</td>
            <td>{{ tr.execution_status }}</td>
            <td>{{ tr.state_changed }}</td>
            <td>{{ tr.latency_ms }} ms</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

_ENV = Environment(autoescape=select_autoescape(default_for_string=True, default=True))


def render_html_report(report: AuditReportData) -> str:
    """Render AuditReportData into a standalone, zero-dependency HTML report."""
    template = _ENV.from_string(_HTML_TEMPLATE)
    return template.render(report=report)


def write_html_report(report: AuditReportData, output_path: str | Path) -> Path:
    """Write a standalone HTML report to disk and return the resolved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report), encoding="utf-8")
    return path
