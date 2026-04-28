#!/usr/bin/env python3
"""
sweep_pd_allocation.py — Sweep chip allocations for PD disaggregated serving.

Enumerates all valid NP (prefill chip) splits for GPT-OSS-120B on 256 chips,
runs run_pd_disagg.py for each, adds a co-located baseline row, computes KV
transfer time at multiple DCN bandwidths, and writes everything to one CSV.

Usage:
    python neusim/run_scripts/sweep_pd_allocation.py
    python neusim/run_scripts/sweep_pd_allocation.py --output_dir results/pd_sweep
"""

import argparse
import csv
import json
import subprocess
import sys
import copy
from pathlib import Path

import neusim.npusim.frontend.run_sim_lib as run_sim_lib
from neusim.npusim.frontend import memory_footprint_analysis_lib
from neusim.run_scripts.sweep_chip_params import load_base_config, _get_ops_generator
from neusim.run_scripts.run_pd_disagg import build_config, compute_kv_cache_transfer_time_ms

# ---- Pool configurations from CP3 best-throughput results (§3.2) ----
MODEL       = "configs/models/gpt-oss-120b.json"
CHIP        = "configs/chips/tpuv5p.json"
CHIP_VER    = "5p"
INPUT_SEQLEN  = 2048
OUTPUT_SEQLEN = 128
TOTAL_CHIPS   = 256

PREFILL_TP    = 16
PREFILL_PP    = 1
PREFILL_BATCH = 64

DECODE_TP     = 8
DECODE_PP     = 2
DECODE_BATCH  = 64

CP = PREFILL_TP * PREFILL_PP   # chips per prefill instance = 16
CD = DECODE_TP  * DECODE_PP    # chips per decode instance  = 16

# DCN bandwidths to sweep for KV transfer sensitivity (GB/s)
DCN_BANDWIDTHS = [50, 100, 200, 400, 800]


def run_disagg(NP, output_dir, dcn_bw=200.0):
    """Call run_pd_disagg.py for a given NP and return the summary dict."""
    ND = TOTAL_CHIPS - NP
    run_output_dir = str(output_dir / f"NP{NP}")
    result = subprocess.run([
        sys.executable, "neusim/run_scripts/run_pd_disagg.py",
        "--model",              MODEL,
        "--chip",               CHIP,
        "--chip_version",       CHIP_VER,
        "--prefill_tp",         str(PREFILL_TP),
        "--prefill_pp",         str(PREFILL_PP),
        "--prefill_batch_size", str(PREFILL_BATCH),
        "--decode_tp",          str(DECODE_TP),
        "--decode_pp",          str(DECODE_PP),
        "--decode_batch_size",  str(DECODE_BATCH),
        "--total_chips",        str(TOTAL_CHIPS),
        "--prefill_chips",      str(NP),
        "--input_seqlen",       str(INPUT_SEQLEN),
        "--output_seqlen",      str(OUTPUT_SEQLEN),
        "--dcn_bw_GBps",        str(dcn_bw),
        "--output_dir",         run_output_dir,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR for NP={NP}:\n{result.stderr}")
        return None

    summary_path = Path(run_output_dir) / "summary.json"
    with open(summary_path) as f:
        return json.load(f)


def run_colocated(output_dir):
    """
    Run co-located baseline: all 256 chips, identical prefill/decode config.
    Uses same TP/PP/batch as prefill pool (best prefill config) since in
    co-located serving one config governs both phases.
    """
    print("Running co-located baseline...")
    run_output_dir = str(output_dir / "colocated")
    result = subprocess.run([
        sys.executable, "neusim/run_scripts/run_pd_disagg.py",
        "--model",              MODEL,
        "--chip",               CHIP,
        "--chip_version",       CHIP_VER,
        "--prefill_tp",         str(PREFILL_TP),
        "--prefill_pp",         str(PREFILL_PP),
        "--prefill_batch_size", str(PREFILL_BATCH),
        "--decode_tp",          str(PREFILL_TP),   # same config for both
        "--decode_pp",          str(PREFILL_PP),
        "--decode_batch_size",  str(PREFILL_BATCH),
        "--total_chips",        str(TOTAL_CHIPS),
        "--input_seqlen",       str(INPUT_SEQLEN),
        "--output_seqlen",      str(OUTPUT_SEQLEN),
        "--dcn_bw_GBps",        "200",
        "--output_dir",         run_output_dir,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR for co-located:\n{result.stderr}")
        return None

    summary_path = Path(run_output_dir) / "summary.json"
    with open(summary_path) as f:
        return json.load(f)


def compute_kv_bw_sensitivity(base_config):
    """
    Compute KV transfer time and fraction of TTFT at multiple DCN bandwidths.
    No simulation needed — purely math using the prefill config.
    Returns list of dicts.
    """
    # Build prefill config to get KV size
    p_cfg = build_config(
        base_config, PREFILL_TP, PREFILL_PP, PREFILL_BATCH,
        "results/tmp/inference.csv", CHIP_VER
    )
    rows = []
    for bw in DCN_BANDWIDTHS:
        t_kv = compute_kv_cache_transfer_time_ms(p_cfg, bw)
        rows.append({"dcn_bw_GBps": bw, "kv_transfer_time_ms": round(t_kv, 4)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/pd_sweep")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Valid NP values: multiples of CP where ND = 256 - NP is also multiple of CD
    valid_NP = [
        NP for NP in range(CP, TOTAL_CHIPS, CP)
        if (TOTAL_CHIPS - NP) % CD == 0 and (TOTAL_CHIPS - NP) > 0
    ]
    print(f"Valid NP splits: {valid_NP}")
    print(f"Total sweep runs: {len(valid_NP)}\n")

    # ---- Allocation sweep ----
    allocation_rows = []
    best_sys_tput = -1
    best_NP = -1

    for i, NP in enumerate(valid_NP):
        ND = TOTAL_CHIPS - NP
        print(f"[{i+1}/{len(valid_NP)}] NP={NP}, ND={ND}...")
        summary = run_disagg(NP, output_dir)
        if summary is None:
            continue

        n_p = summary["n_prefill_instances"]
        n_d = summary["n_decode_instances"]
        lP  = summary["prefill_instance_req_per_sec"]
        lD  = summary["decode_instance_req_per_sec"]
        sys_tput = summary["system_throughput_req_per_sec"]

        if sys_tput > best_sys_tput:
            best_sys_tput = sys_tput
            best_NP = NP

        allocation_rows.append({
            "row_type":                    "allocation_sweep",
            "NP":                          NP,
            "ND":                          ND,
            "n_prefill_instances":         n_p,
            "n_decode_instances":          n_d,
            "prefill_pool_capacity_rps":   round(n_p * lP, 6),
            "decode_pool_capacity_rps":    round(n_d * lD, 6),
            "system_throughput_rps":       sys_tput,
            "TTFT_prefill_only_ms":        summary["TTFT_prefill_only_ms"],
            "kv_transfer_time_ms":         summary["kv_transfer_time_ms"],
            "TTFT_total_ms":               summary["TTFT_total_ms"],
            "TPOT_ms":                     summary["TPOT_ms"],
            "is_optimal":                  False,   # updated below
            "dcn_bw_GBps":                 "",
            "kv_transfer_sensitivity_ms":  "",
        })
        print(f"  sys_tput={sys_tput:.3f} req/s")

    # Mark optimal
    for row in allocation_rows:
        if row["NP"] == best_NP:
            row["is_optimal"] = True

    print(f"\nOptimal NP={best_NP}, system throughput={best_sys_tput:.3f} req/s")

    # ---- Co-located baseline ----
    colocated = run_colocated(output_dir)
    colocated_tput = colocated["system_throughput_req_per_sec"] if colocated else None
    colocated_row = {
        "row_type":                    "colocated_baseline",
        "NP":                          TOTAL_CHIPS,
        "ND":                          0,
        "n_prefill_instances":         colocated["n_prefill_instances"] if colocated else "",
        "n_decode_instances":          colocated["n_decode_instances"]  if colocated else "",
        "prefill_pool_capacity_rps":   "",
        "decode_pool_capacity_rps":    "",
        "system_throughput_rps":       colocated_tput or "",
        "TTFT_prefill_only_ms":        colocated["TTFT_prefill_only_ms"] if colocated else "",
        "kv_transfer_time_ms":         colocated["kv_transfer_time_ms"]  if colocated else "",
        "TTFT_total_ms":               colocated["TTFT_total_ms"]        if colocated else "",
        "TPOT_ms":                     colocated["TPOT_ms"]              if colocated else "",
        "is_optimal":                  "",
        "dcn_bw_GBps":                 "",
        "kv_transfer_sensitivity_ms":  "",
    }

    # ---- KV bandwidth sensitivity ----
    print("\nComputing KV transfer sensitivity...")
    base_config = load_base_config(MODEL, CHIP)
    base_config["input_seqlen"]  = INPUT_SEQLEN
    base_config["output_seqlen"] = OUTPUT_SEQLEN
    kv_bw_rows = compute_kv_bw_sensitivity(base_config)

    # Get TTFT_prefill_only from the best NP run to compute fractions
    best_row = next(r for r in allocation_rows if r["NP"] == best_NP)
    TTFT_prefill_ms = best_row["TTFT_prefill_only_ms"]

    bw_rows = []
    for r in kv_bw_rows:
        frac = r["kv_transfer_time_ms"] / (TTFT_prefill_ms + r["kv_transfer_time_ms"])
        bw_rows.append({
            "row_type":                    "kv_bw_sensitivity",
            "NP":                          "",
            "ND":                          "",
            "n_prefill_instances":         "",
            "n_decode_instances":          "",
            "prefill_pool_capacity_rps":   "",
            "decode_pool_capacity_rps":    "",
            "system_throughput_rps":       "",
            "TTFT_prefill_only_ms":        TTFT_prefill_ms,
            "kv_transfer_time_ms":         r["kv_transfer_time_ms"],
            "TTFT_total_ms":               round(TTFT_prefill_ms + r["kv_transfer_time_ms"], 4),
            "TPOT_ms":                     "",
            "is_optimal":                  "",
            "dcn_bw_GBps":                 r["dcn_bw_GBps"],
            "kv_transfer_sensitivity_ms":  r["kv_transfer_time_ms"],
        })
        print(f"  DCN={r['dcn_bw_GBps']} GB/s → t_KV={r['kv_transfer_time_ms']:.2f} ms  ({frac*100:.1f}% of TTFT)")

    # ---- Write combined CSV ----
    all_rows = allocation_rows + [colocated_row] + bw_rows
    csv_path = output_dir / "pd_sweep_results.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nAll results written to: {csv_path}")
    print(f"Optimal NP={best_NP} (system throughput={best_sys_tput:.3f} req/s)")
    if colocated_tput:
        improvement = (best_sys_tput - colocated_tput) / colocated_tput * 100
        print(f"Co-located baseline:    {colocated_tput:.3f} req/s")
        print(f"Disaggregation improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
