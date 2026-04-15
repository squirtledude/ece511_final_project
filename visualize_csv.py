#!/usr/bin/env python3
"""
visualize_csv.py — Turn a NeuSim inference CSV into a readable HTML report.

Usage:
    python visualize_csv.py path/to/inference.csv
    python visualize_csv.py path/to/inference.csv -o output.html
"""

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Column groups — controls which columns appear in each tab
# ---------------------------------------------------------------------------
COLUMN_GROUPS = {
    "Overview": [
        "Fusion index",
        "Description",
        "OpType",
        "Count",
        "Bounded-by",
        "Execution time",
        "Compute time",
        "Memory time",
        "FLOP Count",
        "tflops_per_sec",
        "hbm_bw_GBps",
        "flops_util",
        "hbm_bw_util",
    ],
    "Timing": [
        "Fusion index",
        "Description",
        "OpType",
        "Bounded-by",
        "Execution time",
        "Compute time",
        "Memory time",
        "ICI/NVLink time",
        "Aggregated DCN time",
        "DCN 0 time",
        "MXU time",
        "VPU time",
        "Transpose time",
        "Permute time",
        "Vmem time",
    ],
    "Memory & Traffic": [
        "Fusion index",
        "Description",
        "OpType",
        "Bytes accessed",
        "Temporary memory size",
        "Persistent memory size",
        "ICI/NVLink outbound traffic",
        "ICI/NVLink inbound traffic",
        "Weight Size",
        "max_vmem_demand_bytes",
        "Input Tensor Shapes",
        "Output Tensor Shapes",
    ],
    "Einsum": [
        "Fusion index",
        "Description",
        "OpType",
        "dim_labels",
        "tile_shapes",
        "num_tiles",
        "num_mxu_ops",
        "einsum_B_size",
        "einsum_M_size",
        "einsum_N_size",
        "einsum_K_size",
        "FLOP Count",
        "tflops_per_sec",
        "flops_util",
    ],
    "Energy & Power": [
        "Fusion index",
        "Description",
        "OpType",
        "total_energy_J",
        "static_energy_J",
        "dynamic_energy_J",
        "total_power_W",
        "static_power_W",
        "dynamic_power_W",
        "dynamic_energy_sa_J",
        "dynamic_energy_vu_J",
        "dynamic_energy_sram_J",
        "dynamic_energy_hbm_J",
        "dynamic_energy_ici_J",
        "static_energy_sa_J",
        "static_energy_vu_J",
        "static_energy_sram_J",
        "static_energy_hbm_J",
        "static_energy_ici_J",
    ],
    "DVFS": [
        "Fusion index",
        "Description",
        "OpType",
        "DVFS SA Policy",
        "DVFS SA Voltage (V)",
        "DVFS SA Frequency (GHz)",
        "DVFS VU Policy",
        "DVFS VU Voltage (V)",
        "DVFS VU Frequency (GHz)",
        "DVFS SRAM Policy",
        "DVFS SRAM Voltage (V)",
        "DVFS SRAM Frequency (GHz)",
        "DVFS HBM Policy",
        "DVFS HBM Voltage (V)",
        "DVFS HBM Frequency (GHz)",
        "DVFS ICI Policy",
        "DVFS ICI Voltage (V)",
        "DVFS ICI Frequency (GHz)",
        "DVFS SA Scaling Time (ns)",
        "DVFS SA Power Efficiency (%)",
        "DVFS VU Scaling Time (ns)",
        "DVFS VU Power Efficiency (%)",
        "DVFS SRAM Scaling Time (ns)",
        "DVFS SRAM Power Efficiency (%)",
        "DVFS HBM Scaling Time (ns)",
        "DVFS HBM Power Efficiency (%)",
        "DVFS ICI Scaling Time (ns)",
        "DVFS ICI Power Efficiency (%)",
        "DVFS SA Activity Factor",
        "DVFS VU Activity Factor",
        "DVFS SRAM Activity Factor",
        "DVFS HBM Activity Factor",
        "DVFS ICI Activity Factor",
    ],
}

# Columns to format as nanoseconds with commas
NS_COLS = {
    "Execution time", "Compute time", "Memory time", "ICI/NVLink time",
    "Aggregated DCN time", "DCN 0 time", "MXU time", "VPU time",
    "Transpose time", "Permute time", "Vmem time",
    "DVFS SA Scaling Time (ns)", "DVFS VU Scaling Time (ns)",
    "DVFS SRAM Scaling Time (ns)", "DVFS HBM Scaling Time (ns)",
    "DVFS ICI Scaling Time (ns)", "Compute Time (ns)",
}

# Columns to format as bytes
BYTE_COLS = {
    "Bytes accessed", "Temporary memory size", "Persistent memory size",
    "Weight Size", "max_vmem_demand_bytes", "ICI/NVLink outbound traffic",
    "ICI/NVLink inbound traffic",
}

# Badge colors per OpType
OPTYPE_COLORS = {
    "MXU":           "#4f46e5",  # indigo
    "VPU":           "#0891b2",  # cyan
    "Elementwise":   "#059669",  # green
    "RMSNorm":       "#16a34a",  # green
    "Collective":    "#d97706",  # amber
    "Embedding":     "#7c3aed",  # violet
    "Attention":     "#db2777",  # pink
    "FlashAttention":"#db2777",
    "Other":         "#6b7280",  # gray
}

# Badge colors per Bounded-by
BOUNDED_COLORS = {
    "Compute": "#15803d",
    "Memory":  "#b45309",
    "ICI":     "#1d4ed8",
    "DCN":     "#7c3aed",
}


def fmt_ns(val: str) -> str:
    try:
        v = float(val)
        if v >= 1_000_000:
            return f"{v/1_000_000:.2f} ms"
        if v >= 1_000:
            return f"{v/1_000:.2f} µs"
        return f"{v:.0f} ns"
    except (ValueError, TypeError):
        return val


def fmt_bytes(val: str) -> str:
    try:
        v = float(val)
        for unit in ("B", "KB", "MB", "GB"):
            if abs(v) < 1024:
                return f"{v:.1f} {unit}"
            v /= 1024
        return f"{v:.1f} TB"
    except (ValueError, TypeError):
        return val


def fmt_float(val: str, decimals: int = 4) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return val


def display_value(col: str, raw: str) -> str:
    if col in NS_COLS:
        return fmt_ns(raw)
    if col in BYTE_COLS:
        return fmt_bytes(raw)
    if col in ("flops_util", "hbm_bw_util"):
        try:
            return f"{float(raw)*100:.1f}%"
        except (ValueError, TypeError):
            return raw
    if col in ("tflops_per_sec", "hbm_bw_GBps"):
        return fmt_float(raw, 2)
    if col in ("DVFS SA Voltage (V)", "DVFS VU Voltage (V)", "DVFS SRAM Voltage (V)",
               "DVFS HBM Voltage (V)", "DVFS ICI Voltage (V)"):
        return fmt_float(raw, 3)
    return raw


def badge(text: str, color_map: dict, default: str = "#6b7280") -> str:
    color = default
    for key, c in color_map.items():
        if key.lower() in text.lower():
            color = c
            break
    escaped = html.escape(str(text))
    return (
        f'<span style="background:{color};color:#fff;padding:2px 7px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600;white-space:nowrap">'
        f"{escaped}</span>"
    )


def build_summary(rows: list[dict]) -> str:
    total = len(rows)
    op_counts: dict[str, int] = {}
    bounded_counts: dict[str, int] = {}
    total_exec_ns = 0.0
    total_energy_j = 0.0

    for r in rows:
        op = r.get("OpType", "Other")
        op_counts[op] = op_counts.get(op, 0) + 1
        b = r.get("Bounded-by", "")
        if b:
            bounded_counts[b] = bounded_counts.get(b, 0) + 1
        try:
            total_exec_ns += float(r.get("Execution time", 0) or 0)
        except ValueError:
            pass
        try:
            total_energy_j += float(r.get("total_energy_J", 0) or 0)
        except ValueError:
            pass

    cards = []

    cards.append(f"""
    <div class="stat-card">
      <div class="stat-label">Total Operators</div>
      <div class="stat-value">{total:,}</div>
    </div>""")

    cards.append(f"""
    <div class="stat-card">
      <div class="stat-label">Total Execution Time</div>
      <div class="stat-value">{fmt_ns(str(total_exec_ns))}</div>
    </div>""")

    if total_energy_j > 0:
        cards.append(f"""
        <div class="stat-card">
          <div class="stat-label">Total Energy</div>
          <div class="stat-value">{total_energy_j*1000:.2f} mJ</div>
        </div>""")

    # Op type breakdown
    op_items = "".join(
        f'<div class="breakdown-item">'
        f'{badge(k, OPTYPE_COLORS)} '
        f'<span class="breakdown-count">{v}</span></div>'
        for k, v in sorted(op_counts.items(), key=lambda x: -x[1])
    )
    cards.append(f"""
    <div class="stat-card wide">
      <div class="stat-label">Op Type Breakdown</div>
      <div class="breakdown">{op_items}</div>
    </div>""")

    # Bounded-by breakdown
    b_items = "".join(
        f'<div class="breakdown-item">'
        f'{badge(k, BOUNDED_COLORS)} '
        f'<span class="breakdown-count">{v}</span></div>'
        for k, v in sorted(bounded_counts.items(), key=lambda x: -x[1])
    )
    cards.append(f"""
    <div class="stat-card wide">
      <div class="stat-label">Bounded-by Breakdown</div>
      <div class="breakdown">{b_items}</div>
    </div>""")

    return f'<div class="summary-row">{"".join(cards)}</div>'


def build_table(rows: list[dict], all_cols: list[str], group_name: str) -> tuple[str, str]:
    """Return (table_html, table_id)."""
    cols = [c for c in COLUMN_GROUPS.get(group_name, []) if c in all_cols]
    if not cols:
        # fallback: first 15 columns
        cols = all_cols[:15]

    table_id = f"tbl_{group_name.replace(' ', '_').replace('&', 'n')}"

    # Header
    th_cells = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    header = f"<thead><tr>{th_cells}</tr></thead>"

    # Rows
    body_rows = []
    for i, r in enumerate(rows):
        cells = []
        for c in cols:
            raw = r.get(c, "")
            formatted = display_value(c, raw)
            if c == "OpType":
                cell = badge(raw, OPTYPE_COLORS)
            elif c == "Bounded-by":
                cell = badge(raw, BOUNDED_COLORS)
            elif c == "Description":
                cell = f'<span class="desc" title="{html.escape(raw)}">{html.escape(raw[:60])}{"…" if len(raw) > 60 else ""}</span>'
            else:
                cell = html.escape(str(formatted))
            cells.append(f"<td>{cell}</td>")

        row_data = html.escape(json.dumps(r))
        body_rows.append(
            f'<tr class="data-row" data-row="{row_data}" data-idx="{i}">'
            + "".join(cells)
            + "</tr>"
        )

    body = f"<tbody>{''.join(body_rows)}</tbody>"
    table = f'<table id="{table_id}" class="data-table">{header}{body}</table>'
    return table, table_id


def build_html(csv_path: str, rows: list[dict], all_cols: list[str]) -> str:
    filename = os.path.basename(csv_path)
    summary_html = build_summary(rows)

    tabs_html = ""
    panels_html = ""
    table_ids: list[str] = []

    for gi, group_name in enumerate(COLUMN_GROUPS):
        active = "active" if gi == 0 else ""
        tabs_html += f'<button class="tab-btn {active}" data-group="{group_name}">{html.escape(group_name)}</button>\n'
        table_html, table_id = build_table(rows, all_cols, group_name)
        table_ids.append(table_id)
        panels_html += (
            f'<div class="tab-panel {active}" data-group="{group_name}">'
            f'<div class="table-wrap">{table_html}</div>'
            f"</div>\n"
        )

    # Detail modal content (filled by JS)
    modal_html = """
    <div id="detail-overlay" class="overlay hidden">
      <div id="detail-panel" class="detail-panel">
        <div class="detail-header">
          <span id="detail-title">Operator Detail</span>
          <button id="detail-close" onclick="closeDetail()">✕</button>
        </div>
        <div id="detail-body" class="detail-body"></div>
      </div>
    </div>
    """

    # JS for search, tabs, detail panel
    js = r"""
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const g = btn.dataset.group;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.querySelector(`.tab-panel[data-group="${g}"]`).classList.add('active');
        // re-apply search filter
        applySearch();
      });
    });

    // Search
    const searchInput = document.getElementById('search');
    searchInput.addEventListener('input', applySearch);

    function applySearch() {
      const q = searchInput.value.toLowerCase();
      const activePanel = document.querySelector('.tab-panel.active');
      if (!activePanel) return;
      activePanel.querySelectorAll('tr.data-row').forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(q) ? '' : 'none';
      });
    }

    // Detail panel
    document.querySelectorAll('tr.data-row').forEach(row => {
      row.addEventListener('click', () => showDetail(row));
    });

    function showDetail(row) {
      const data = JSON.parse(row.dataset.row);
      const title = data['Description'] || data['Name'] || `Row ${row.dataset.idx}`;
      document.getElementById('detail-title').textContent = title;

      // Group fields
      const groups = """ + json.dumps({k: v for k, v in COLUMN_GROUPS.items()}) + r""";
      const allCols = Object.keys(data);

      let html = '';

      // Build sections
      for (const [groupName, cols] of Object.entries(groups)) {
        const relevant = cols.filter(c => allCols.includes(c));
        if (!relevant.length) continue;
        html += `<div class="detail-section"><div class="detail-section-title">${groupName}</div><table class="kv-table">`;
        for (const col of relevant) {
          const raw = data[col] ?? '';
          html += `<tr><td class="kv-key">${escHtml(col)}</td><td class="kv-val">${escHtml(String(raw))}</td></tr>`;
        }
        html += '</table></div>';
      }

      // Remaining columns not in any group
      const shownCols = new Set(Object.values(groups).flat());
      const remaining = allCols.filter(c => !shownCols.has(c));
      if (remaining.length) {
        html += `<div class="detail-section"><div class="detail-section-title">Other</div><table class="kv-table">`;
        for (const col of remaining) {
          const raw = data[col] ?? '';
          html += `<tr><td class="kv-key">${escHtml(col)}</td><td class="kv-val">${escHtml(String(raw))}</td></tr>`;
        }
        html += '</table></div>';
      }

      document.getElementById('detail-body').innerHTML = html;
      document.getElementById('detail-overlay').classList.remove('hidden');
    }

    function closeDetail() {
      document.getElementById('detail-overlay').classList.add('hidden');
    }

    document.getElementById('detail-overlay').addEventListener('click', e => {
      if (e.target === document.getElementById('detail-overlay')) closeDetail();
    });

    function escHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    // Sortable columns
    document.querySelectorAll('.data-table th').forEach(th => {
      th.style.cursor = 'pointer';
      th.addEventListener('click', () => {
        const table = th.closest('table');
        const colIdx = Array.from(th.parentElement.children).indexOf(th);
        const asc = th.dataset.sortAsc !== 'true';
        th.dataset.sortAsc = asc;
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
          const aVal = a.children[colIdx]?.textContent.trim() ?? '';
          const bVal = b.children[colIdx]?.textContent.trim() ?? '';
          const aNum = parseFloat(aVal.replace(/[^0-9.\-]/g, ''));
          const bNum = parseFloat(bVal.replace(/[^0-9.\-]/g, ''));
          if (!isNaN(aNum) && !isNaN(bNum)) return asc ? aNum - bNum : bNum - aNum;
          return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        });
        rows.forEach(r => tbody.appendChild(r));
      });
    });
    """

    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           font-size: 13px; background: #0f172a; color: #e2e8f0; }
    a { color: #7dd3fc; }

    .topbar { background: #1e293b; padding: 14px 24px; border-bottom: 1px solid #334155;
              display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .topbar h1 { font-size: 1rem; font-weight: 700; color: #f1f5f9; }
    .topbar .filename { font-size: 0.8rem; color: #94a3b8; font-family: monospace; }

    #search { background: #0f172a; border: 1px solid #475569; color: #e2e8f0;
              padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; width: 280px; }
    #search::placeholder { color: #64748b; }
    #search:focus { outline: none; border-color: #7dd3fc; }

    .main { padding: 20px 24px; }

    /* Summary */
    .summary-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
    .stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px;
                 padding: 14px 18px; min-width: 160px; }
    .stat-card.wide { min-width: 260px; }
    .stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .05em;
                  color: #94a3b8; margin-bottom: 6px; }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; }
    .breakdown { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .breakdown-item { display: flex; align-items: center; gap: 5px; }
    .breakdown-count { font-weight: 600; color: #cbd5e1; }

    /* Tabs */
    .tabs { display: flex; gap: 4px; margin-bottom: 0; border-bottom: 1px solid #334155;
            padding-bottom: 0; }
    .tab-btn { background: transparent; border: none; color: #94a3b8; padding: 8px 16px;
               cursor: pointer; font-size: 0.8rem; font-weight: 500; border-radius: 6px 6px 0 0;
               border-bottom: 2px solid transparent; transition: all .15s; }
    .tab-btn:hover { color: #e2e8f0; background: #1e293b; }
    .tab-btn.active { color: #7dd3fc; border-bottom-color: #7dd3fc; background: #1e293b; }

    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    /* Table */
    .table-wrap { overflow-x: auto; border: 1px solid #334155; border-radius: 0 8px 8px 8px; }
    .data-table { width: 100%; border-collapse: collapse; }
    .data-table thead tr { background: #1e293b; }
    .data-table th { padding: 9px 12px; text-align: left; font-size: 0.72rem;
                     text-transform: uppercase; letter-spacing: .04em; color: #94a3b8;
                     white-space: nowrap; border-bottom: 1px solid #334155;
                     user-select: none; }
    .data-table th:hover { color: #e2e8f0; }
    .data-table td { padding: 7px 12px; border-bottom: 1px solid #1e293b;
                     color: #cbd5e1; white-space: nowrap; }
    .data-table tr.data-row { cursor: pointer; transition: background .1s; }
    .data-table tr.data-row:hover { background: #1e3a5f; }
    .data-table tr.data-row:nth-child(even) { background: #0f1e2f; }
    .data-table tr.data-row:nth-child(even):hover { background: #1e3a5f; }
    .desc { display: block; max-width: 320px; overflow: hidden; text-overflow: ellipsis; }

    /* Detail overlay */
    .overlay { position: fixed; inset: 0; background: rgba(0,0,0,.6);
               display: flex; align-items: flex-start; justify-content: flex-end;
               z-index: 100; padding: 0; }
    .overlay.hidden { display: none; }
    .detail-panel { background: #1e293b; width: 520px; max-width: 95vw; height: 100vh;
                    overflow-y: auto; box-shadow: -4px 0 24px rgba(0,0,0,.4); }
    .detail-header { position: sticky; top: 0; background: #1e293b;
                     padding: 16px 20px; border-bottom: 1px solid #334155;
                     display: flex; justify-content: space-between; align-items: center; z-index: 1; }
    #detail-title { font-weight: 600; font-size: 0.85rem; color: #f1f5f9;
                    word-break: break-all; max-width: 420px; }
    #detail-close { background: transparent; border: none; color: #94a3b8;
                    font-size: 1.1rem; cursor: pointer; padding: 4px 8px; border-radius: 4px; }
    #detail-close:hover { color: #f1f5f9; background: #334155; }
    .detail-body { padding: 16px 20px; }
    .detail-section { margin-bottom: 20px; }
    .detail-section-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .06em;
                             color: #7dd3fc; font-weight: 700; margin-bottom: 8px; }
    .kv-table { width: 100%; border-collapse: collapse; }
    .kv-table tr:hover { background: #0f172a; }
    .kv-key { color: #94a3b8; padding: 4px 10px 4px 0; vertical-align: top;
              white-space: nowrap; font-size: 0.75rem; width: 40%; }
    .kv-val { color: #e2e8f0; padding: 4px 0; word-break: break-all;
              font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.75rem; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NeuSim — {html.escape(filename)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="topbar">
    <h1>NeuSim Operator Report</h1>
    <span class="filename">{html.escape(filename)}</span>
    <input id="search" type="text" placeholder="Filter operators…" />
  </div>
  <div class="main">
    {summary_html}
    <div class="tabs">
      {tabs_html}
    </div>
    {panels_html}
  </div>
  {modal_html}
  <script>{js}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a NeuSim inference CSV as an HTML report."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file path (default: <csv_file>.html)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else csv_path.with_suffix(".html")

    print(f"Reading {csv_path} …")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        all_cols = reader.fieldnames or []

    print(f"  {len(rows):,} rows, {len(all_cols)} columns")

    html_content = build_html(str(csv_path), rows, list(all_cols))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
