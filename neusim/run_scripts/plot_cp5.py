#!/usr/bin/env python3
"""
Generates CP5 plots from sweep CSV.
"""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---- Config ----
CP4_BASELINE_RPS = 185.36
CP = CD = 16   # chips per instance for both pools


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x):
    """Parse float from CSV string."""
    return float(x)


# ---------------------------------------------------------------------------
# Plot 1: Λ_sys vs HBM BW (num_sa=8, best batch per BW)
# ---------------------------------------------------------------------------
def plot_lsys_vs_hbm_bw(rows, out_dir):
    # Keep num_sa=8 only; pick best Λ_sys per HBM BW
    bw_best = {}
    for r in rows:
        if r["num_sa"] != "8":
            continue
        bw = int(r["hbm_bw_GBps"])
        tput = f(r["system_throughput_rps"])
        if bw not in bw_best or tput > bw_best[bw]["tput"]:
            bw_best[bw] = {
                "tput":    tput,
                "d_batch": int(r["decode_batch"]),
                "chips":   int(r["max_chips"]),
                "TPOT":    f(r["TPOT_ms"]),
            }

    bws   = sorted(bw_best.keys())
    tputs = [bw_best[b]["tput"] for b in bws]
    batches = [bw_best[b]["d_batch"] for b in bws]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bws, tputs, "o-", color="#1f77b4", linewidth=2, markersize=7, zorder=3)

    # Annotate optimal point
    opt_idx = tputs.index(max(tputs))
    ax.annotate(
        f"Optimal: {max(tputs):.1f} rps\n(batch={batches[opt_idx]}, "
        f"{bw_best[bws[opt_idx]]['chips']} chips)",
        xy=(bws[opt_idx], max(tputs)),
        xytext=(bws[opt_idx] - 1000, max(tputs) - 25),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    # CP4 baseline reference line
    ax.axhline(CP4_BASELINE_RPS, color="red", linestyle="--", linewidth=1.2,
               label=f"CP4 baseline ({CP4_BASELINE_RPS} rps)")

    # Annotate dip points
    for i, bw in enumerate(bws):
        if i > 0 and tputs[i] < tputs[i - 1]:
            ax.annotate("chip count\ndrop", xy=(bw, tputs[i]),
                        xytext=(bw + 100, tputs[i] - 18),
                        fontsize=7.5, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("HBM Bandwidth (GB/s)", fontsize=11)
    ax.set_ylabel("System Throughput $\\Lambda_{sys}$ (req/s)", fontsize=11)
    ax.set_title("System Throughput vs. HBM Bandwidth\n(num\\_sa=8, best decode batch per BW)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bws)
    ax.set_xticklabels([str(b) for b in bws], rotation=30, ha="right")

    fig.tight_layout()
    path = out_dir / "cp5_lsys_vs_hbm_bw.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: TPOT vs Decode Batch at two HBM BW points
# ---------------------------------------------------------------------------
def plot_tpot_vs_batch(rows, out_dir):
    # Extract TPOT vs d_batch for two BW levels (num_sa=8)
    # Use hbm_size=128 for 5500, hbm_size=64 for 2765 (largest valid size at low BW)
    target_configs = [
        {"hbm_bw": "2765", "hbm_size": "64",  "num_sa": "8",
         "label": "Baseline BW (2765 GB/s)", "color": "#ff7f0e"},
        {"hbm_bw": "5500", "hbm_size": "128", "num_sa": "8",
         "label": "Optimized BW (5500 GB/s)", "color": "#1f77b4"},
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cfg in target_configs:
        # Collect unique (d_batch, TPOT) pairs — use p_batch=64 to avoid duplicates
        pts = {}
        for r in rows:
            if (r["hbm_bw_GBps"] == cfg["hbm_bw"] and
                    r["hbm_size_GB"] == cfg["hbm_size"] and
                    r["num_sa"] == cfg["num_sa"] and
                    r["prefill_batch"] == "64"):
                db = int(r["decode_batch"])
                pts[db] = f(r["TPOT_ms"])
        if not pts:
            print(f"  Warning: no data for config {cfg}")
            continue
        batches = sorted(pts.keys())
        tpots   = [pts[b] for b in batches]
        ax.plot(batches, tpots, "o-", color=cfg["color"],
                linewidth=2, markersize=7, label=cfg["label"], zorder=3)

    # SLO line
    ax.axhline(100, color="red", linestyle="--", linewidth=1.5,
               label="TPOT SLO (100 ms)", zorder=2)

    # Annotate the operating points
    ax.annotate("CP4 operating\npoint (batch=64)", xy=(64, 37.76),
                xytext=(90, 45),
                fontsize=8, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.8))
    ax.annotate("CP5 operating\npoint (batch=320)", xy=(320, 95.8),
                xytext=(230, 80),
                fontsize=8, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))

    ax.set_xlabel("Decode Batch Size", fontsize=11)
    ax.set_ylabel("TPOT (ms/token)", fontsize=11)
    ax.set_title("TPOT vs. Decode Batch Size\nat Baseline and Optimized HBM Bandwidth",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "cp5_tpot_vs_batch.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Best Λ_sys per num_sa (bar chart)
# ---------------------------------------------------------------------------
def plot_lsys_per_num_sa(rows, out_dir):
    # Best Λ_sys per num_sa
    sa_best = {}
    for r in rows:
        sa   = int(r["num_sa"])
        tput = f(r["system_throughput_rps"])
        if sa not in sa_best or tput > sa_best[sa]["tput"]:
            sa_best[sa] = {
                "tput":    tput,
                "hbm_bw":  int(r["hbm_bw_GBps"]),
                "chips":   int(r["max_chips"]),
                "TPOT":    f(r["TPOT_ms"]),
                "TTFT":    f(r["TTFT_total_ms"]),
                "d_batch": int(r["decode_batch"]),
            }

    sas   = sorted(sa_best.keys())
    tputs = [sa_best[s]["tput"] for s in sas]
    colors = ["#d62728", "#ff7f0e", "#1f77b4"]  # red, orange, blue

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar([str(s) for s in sas], tputs, color=colors,
                  edgecolor="white", linewidth=1.2, zorder=3)

    # Value labels on bars
    for bar, tput, sa in zip(bars, tputs, sas):
        d = sa_best[sa]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{tput:.1f} rps\n(bw={d['hbm_bw']}, {d['chips']} chips)",
                ha="center", va="bottom", fontsize=8)

    # CP4 baseline line
    ax.axhline(CP4_BASELINE_RPS, color="red", linestyle="--", linewidth=1.2,
               label=f"CP4 baseline ({CP4_BASELINE_RPS} rps)")

    ax.set_xlabel("Number of Systolic Arrays (num\\_sa)", fontsize=11)
    ax.set_ylabel("Best $\\Lambda_{sys}$ (req/s)", fontsize=11)
    ax.set_title("Best Achievable Throughput per num\\_sa\n"
                 "(each at its globally optimal HBM BW and batch size)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(tputs) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "cp5_lsys_per_num_sa.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: NP Allocation Sweep at Final Chip Config
# ---------------------------------------------------------------------------
def plot_np_allocation(rows, out_dir):
    # Find the best row to get λ_P, λ_D, max_chips
    best = max(rows, key=lambda r: f(r["system_throughput_rps"]))
    lambda_P  = f(best["lambda_P_rps"])
    lambda_D  = f(best["lambda_D_rps"])
    max_chips = int(best["max_chips"])

    NP_vals, prefill_cap, decode_cap, lsys_vals = [], [], [], []
    for NP in range(CP, max_chips, CP):
        ND = max_chips - NP
        if ND <= 0 or ND % CD != 0:
            continue
        n_p = NP // CP
        n_d = ND // CD
        pc = n_p * lambda_P
        dc = n_d * lambda_D
        NP_vals.append(NP)
        prefill_cap.append(pc)
        decode_cap.append(dc)
        lsys_vals.append(min(pc, dc))

    opt_NP  = NP_vals[lsys_vals.index(max(lsys_vals))]
    opt_tput = max(lsys_vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(NP_vals, prefill_cap, "b--",  linewidth=2, label="Prefill pool capacity")
    ax.plot(NP_vals, decode_cap,  "g--",  linewidth=2, label="Decode pool capacity")
    ax.plot(NP_vals, lsys_vals,   "k-",   linewidth=2.5, label="$\\Lambda_{sys}$")
    ax.axvline(opt_NP, color="red", linestyle=":", linewidth=1.5,
               label=f"Optimal $N_P$={opt_NP} ({opt_tput:.1f} rps)")

    ax.set_xlabel("$N_P$ (Prefill Chips)", fontsize=11)
    ax.set_ylabel("Throughput (req/s)", fontsize=11)
    ax.set_title(f"Chip Allocation Sweep — CP5 Optimized Chip\n"
                 f"(HBM=5500 GB/s, 128 GB, num\\_sa=8, {max_chips} total chips)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "cp5_np_allocation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/cp5_sweep/cp5_sweep_results.csv")
    parser.add_argument("--output_dir", default="plots/cp5")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}\n")

    plot_lsys_vs_hbm_bw(rows, out_dir)
    plot_tpot_vs_batch(rows, out_dir)
    plot_lsys_per_num_sa(rows, out_dir)
    plot_np_allocation(rows, out_dir)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
