#!/usr/bin/env python3
"""
Joint sweep over HBM bandwidth, HBM size, num_sa, and batch sizes.

Finds the chip configuration + batch sizes that maximizes system throughput
Λ_sys while satisfying all feasibility, SLO, and budget constraints.
"""

import argparse
import copy
import csv
import json
import os
from math import floor
from itertools import product
from pathlib import Path

from neusim.configs.chips.ChipConfig import ChipConfig
from feasibility import estimate_area, estimate_perimeter_usage, estimate_cost
import neusim.npusim.frontend.run_sim_lib as run_sim_lib
from neusim.run_scripts.sweep_chip_params import load_base_config, _get_ops_generator
from neusim.run_scripts.run_pd_disagg import build_config, compute_kv_cache_transfer_time_ms

NEUSIM_ROOT = Path(__file__).resolve().parent.parent.parent

# ---- Fixed pool parallelism from CP3/CP4 best throughput configs ----
PREFILL_TP, PREFILL_PP = 16, 1
DECODE_TP,  DECODE_PP  = 8,  2
CP = PREFILL_TP * PREFILL_PP   # chips per prefill instance = 16
CD = DECODE_TP  * DECODE_PP    # chips per decode instance  = 16

# ---- Fixed workload ----
MODEL         = "configs/models/gpt-oss-120b.json"
CHIP_BASE     = "configs/chips/tpuv5p.json"
CHIP_VER      = "5p"
INPUT_SEQLEN  = 2048
OUTPUT_SEQLEN = 128
DCN_BW_GBPS   = 200.0

# ---- Constraints ----
TOTAL_BUDGET  = 450_000
TTFT_SLO_MS   = 2000
TPOT_SLO_MS   = 100
MAX_AREA_MM2  = 858
# HBM3e physical consistency: bandwidth/capacity ratio must be 20-45 GB/s per GB
HBM_RATIO_MIN = 20
HBM_RATIO_MAX = 45

# ---- Sweep ranges ----
# HBM BW: TPUv5p baseline (2765) up to perimeter limit (~5515 GB/s)
HBM_BW_VALUES      = [2765, 3000, 3500, 4000, 4500, 5000, 5500]
# HBM size: above minimum footprint (16.61 GB); physically consistent with BW via ratio constraint
HBM_SIZE_VALUES    = [64, 96, 128, 160, 192]
# num_sa: CP3 showed TPOT plateaus at 4 SAs; decode gets no benefit beyond that.
NUM_SA_VALUES      = [2, 4, 8]
# Decode batch: CP4 baseline (64) up to ~TPOT_SLO/TPOT_current * 64 ≈ 169
# extended for higher BW configs where TPOT headroom is larger
DECODE_BATCH_SIZES = [64, 96, 128, 192, 256, 320]
# Prefill batch: already compute-saturated at 64 (confirmed CP3 §8.1.3)
PREFILL_BATCH_SIZES = [64, 128]


# ---------------------------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------------------------

def check_chip_feasibility(chip_cfg_dict):
    """
    Check area, perimeter, HBM ratio, and per-chip cost.
    Returns (feasible, cost_per_chip, details_dict).
    """
    chip = ChipConfig(**chip_cfg_dict)
    area  = estimate_area(chip)
    perim = estimate_perimeter_usage(chip, area)
    cost  = estimate_cost(chip, area)

    hbm_ratio = chip_cfg_dict["hbm_bw_GBps"] / chip_cfg_dict["hbm_size_GB"]
    feasible = (
        area.total_mm2  <= MAX_AREA_MM2 and
        perim.is_feasible and
        HBM_RATIO_MIN <= hbm_ratio <= HBM_RATIO_MAX
    )
    return feasible, cost.total_usd, {
        "area_mm2":       round(area.total_mm2, 2),
        "perimeter_util": round(perim.utilization, 4),
        "hbm_ratio":      round(hbm_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_pool(base_config, tp, pp, batch, out_dir, chip_ver):
    """
    Run simulation for one pool config.
    Returns (prefill_stats, decode_stats, pool_config_dict) or raises on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(base_config, tp, pp, batch,
                       str(out_dir / "inference.csv"), chip_ver)
    _get_ops_generator(cfg).generate(dump_to_file=True, separate_prefill_decode=True)
    p_stats = run_sim_lib.get_statistics_from_trace_file(
        str(out_dir / "inference_prefill.csv"))
    d_stats = run_sim_lib.get_statistics_from_trace_file(
        str(out_dir / "inference_decode.csv"))
    return p_stats, d_stats, cfg


def compute_ttft_ms(prefill_stats, pp):
    """TTFT in ms — mirrors dump_stats_llm() formula (pp_dcn=1)."""
    return ((prefill_stats["total_execution_time_non_pp_ns"] +
             prefill_stats["total_pp_ici_time_ns"]) * pp +
            prefill_stats["total_pp_dcn_time_ns"]) / 1e6


def compute_tpot_ms(decode_stats, pp, output_seqlen):
    """TPOT in ms — mirrors dump_stats_llm() formula (pp_dcn=1)."""
    return ((decode_stats["total_execution_time_non_pp_ns"] +
             decode_stats["total_pp_ici_time_ns"]) * pp / output_seqlen +
            decode_stats["total_pp_dcn_time_ns"] / output_seqlen) / 1e6


# ---------------------------------------------------------------------------
# NP split optimizer (analytical — no extra simulations needed)
# ---------------------------------------------------------------------------

def find_best_np(max_chips, lambda_P, lambda_D):
    """
    Analytically find the NP split that maximizes Λ_sys = min(nP*λP, nD*λD).
    Since CP=CD=16, all multiples of 16 are valid splits.
    Returns (best_NP, best_Λ_sys).
    """
    best_tput, best_np = -1, -1
    for NP in range(CP, max_chips, CP):
        ND = max_chips - NP
        if ND <= 0 or ND % CD != 0:
            continue
        tput = min((NP // CP) * lambda_P, (ND // CD) * lambda_D)
        if tput > best_tput:
            best_tput, best_np = tput, NP
    return best_np, best_tput


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/cp5_sweep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base configs
    base_model_config = load_base_config(MODEL, CHIP_BASE)
    base_model_config["input_seqlen"]  = INPUT_SEQLEN
    base_model_config["output_seqlen"] = OUTPUT_SEQLEN

    with open(CHIP_BASE) as f:
        base_chip_dict = json.load(f)

    # Simulation result caches
    # prefill depends on (hbm_bw, hbm_size, num_sa, p_batch)
    # decode depends on (hbm_bw, hbm_size, num_sa, d_batch)
    # num_sa is included because it affects TTFT (prefill) and slightly TPOT (decode)
    prefill_cache = {}
    decode_cache  = {}

    results = []
    stats = {"feasibility_skip": 0, "budget_skip": 0,
             "slo_skip": 0, "sim_error": 0}

    combos = list(product(HBM_BW_VALUES, HBM_SIZE_VALUES,
                          NUM_SA_VALUES,
                          PREFILL_BATCH_SIZES, DECODE_BATCH_SIZES))
    print(f"Total combinations: {len(combos)}")
    print("=" * 70)

    for hbm_bw, hbm_size, num_sa, p_batch, d_batch in combos:

        # --- 1. Build chip config and check feasibility ---
        chip_dict = {**base_chip_dict,
                     "hbm_bw_GBps": hbm_bw,
                     "hbm_size_GB": hbm_size,
                     "num_sa":      num_sa}
        feasible, cost_per_chip, feas_details = check_chip_feasibility(chip_dict)
        if not feasible:
            stats["feasibility_skip"] += 1
            continue

        # --- 2. Budget → max chips (must be multiple of 16) ---
        max_chips = floor(TOTAL_BUDGET / cost_per_chip / 16) * 16
        if max_chips < CP + CD:   # need at least one instance of each pool
            stats["budget_skip"] += 1
            continue

        # Base config patched with this chip's specs
        run_base = {**base_model_config,
                    "hbm_bw_GBps": hbm_bw,
                    "hbm_size_GB": hbm_size,
                    "num_sa":      num_sa}

        # --- 3. Run / retrieve prefill simulation ---
        p_key = (hbm_bw, hbm_size, num_sa, p_batch)
        if p_key not in prefill_cache:
            p_out = output_dir / f"prefill_bw{hbm_bw}_sz{hbm_size}_sa{num_sa}_bs{p_batch}"
            try:
                p_stats, _, p_cfg = run_pool(
                    run_base, PREFILL_TP, PREFILL_PP, p_batch, p_out, CHIP_VER)
                TTFT_ms  = compute_ttft_ms(p_stats, PREFILL_PP)
                t_KV_ms  = compute_kv_cache_transfer_time_ms(p_cfg, DCN_BW_GBPS)
                lambda_P = p_batch / (TTFT_ms / 1000)
                prefill_cache[p_key] = {
                    "TTFT_ms":       TTFT_ms,
                    "TTFT_total_ms": TTFT_ms + t_KV_ms,
                    "t_KV_ms":       t_KV_ms,
                    "lambda_P":      lambda_P,
                    "ok":            True,
                }
            except Exception as e:
                print(f"  [PREFILL ERROR] {p_key}: {e}")
                prefill_cache[p_key] = {"ok": False}

        p_res = prefill_cache[p_key]
        if not p_res["ok"]:
            stats["sim_error"] += 1
            continue

        # --- 4. Run / retrieve decode simulation ---
        d_key = (hbm_bw, hbm_size, num_sa, d_batch)
        if d_key not in decode_cache:
            d_out = output_dir / f"decode_bw{hbm_bw}_sz{hbm_size}_sa{num_sa}_bs{d_batch}"
            try:
                _, d_stats, _ = run_pool(
                    run_base, DECODE_TP, DECODE_PP, d_batch, d_out, CHIP_VER)
                TPOT_ms  = compute_tpot_ms(d_stats, DECODE_PP, OUTPUT_SEQLEN)
                lambda_D = d_batch / (TPOT_ms * OUTPUT_SEQLEN / 1000)
                decode_cache[d_key] = {
                    "TPOT_ms":  TPOT_ms,
                    "lambda_D": lambda_D,
                    "ok":       True,
                }
            except Exception as e:
                print(f"  [DECODE ERROR] {d_key}: {e}")
                decode_cache[d_key] = {"ok": False}

        d_res = decode_cache[d_key]
        if not d_res["ok"]:
            stats["sim_error"] += 1
            continue

        # --- 5. Check SLOs ---
        if p_res["TTFT_total_ms"] > TTFT_SLO_MS:
            stats["slo_skip"] += 1
            continue
        if d_res["TPOT_ms"] > TPOT_SLO_MS:
            stats["slo_skip"] += 1
            continue

        # --- 6. Find optimal NP split analytically ---
        best_np, best_sys_tput = find_best_np(
            max_chips, p_res["lambda_P"], d_res["lambda_D"])
        if best_np < 0:
            continue

        best_nd = max_chips - best_np
        result = {
            "hbm_bw_GBps":           hbm_bw,
            "hbm_size_GB":           hbm_size,
            "num_sa":                num_sa,
            "prefill_batch":         p_batch,
            "decode_batch":          d_batch,
            "cost_per_chip":         round(cost_per_chip, 2),
            "max_chips":             max_chips,
            "total_cost":            round(max_chips * cost_per_chip, 2),
            "area_mm2":              feas_details["area_mm2"],
            "perimeter_util":        feas_details["perimeter_util"],
            "hbm_ratio_GBps_per_GB": feas_details["hbm_ratio"],
            "TTFT_prefill_ms":       round(p_res["TTFT_ms"], 3),
            "kv_transfer_ms":        round(p_res["t_KV_ms"], 3),
            "TTFT_total_ms":         round(p_res["TTFT_total_ms"], 3),
            "TPOT_ms":               round(d_res["TPOT_ms"], 3),
            "lambda_P_rps":          round(p_res["lambda_P"], 4),
            "lambda_D_rps":          round(d_res["lambda_D"], 4),
            "optimal_NP":            best_np,
            "optimal_ND":            best_nd,
            "n_prefill_instances":   best_np // CP,
            "n_decode_instances":    best_nd // CD,
            "system_throughput_rps": round(best_sys_tput, 4),
        }
        results.append(result)

        print(f"bw={hbm_bw} sz={hbm_size} sa={num_sa} "
              f"p_bs={p_batch} d_bs={d_batch} | "
              f"chips={max_chips} NP={best_np} | "
              f"TTFT={p_res['TTFT_total_ms']:.0f}ms "
              f"TPOT={d_res['TPOT_ms']:.1f}ms | "
              f"Λ={best_sys_tput:.2f} rps")

    # --- Write results ---
    results.sort(key=lambda x: x["system_throughput_rps"], reverse=True)
    csv_path = output_dir / "cp5_sweep_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Valid configs found: {len(results)}")
    print(f"Skipped — feasibility: {stats['feasibility_skip']}, "
          f"budget: {stats['budget_skip']}, "
          f"SLO: {stats['slo_skip']}, "
          f"errors: {stats['sim_error']}")

    if results:
        best = results[0]
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION:")
        print(f"  HBM:      {best['hbm_bw_GBps']} GB/s,  {best['hbm_size_GB']} GB  "
              f"(ratio={best['hbm_ratio_GBps_per_GB']} GB/s/GB)")
        print(f"  num_sa:   {best['num_sa']}  (TPUv5p baseline: 8)")
        print(f"  Batches:  prefill={best['prefill_batch']}, "
              f"decode={best['decode_batch']}")
        print(f"  Chips:    {best['max_chips']} total  "
              f"(NP={best['optimal_NP']}, ND={best['optimal_ND']})")
        print(f"  Prefill:  {best['n_prefill_instances']} instances × "
              f"{best['lambda_P_rps']:.3f} rps = "
              f"{best['n_prefill_instances']*best['lambda_P_rps']:.2f} rps")
        print(f"  Decode:   {best['n_decode_instances']} instances × "
              f"{best['lambda_D_rps']:.3f} rps = "
              f"{best['n_decode_instances']*best['lambda_D_rps']:.2f} rps")
        print(f"  TTFT:     {best['TTFT_total_ms']:.1f} ms  (SLO: {TTFT_SLO_MS} ms)")
        print(f"  TPOT:     {best['TPOT_ms']:.1f} ms  (SLO: {TPOT_SLO_MS} ms)")
        print(f"  Λ_sys:    {best['system_throughput_rps']:.2f} req/s  "
              f"(CP4 baseline: 185.36 req/s)")
        print(f"  Cost:     ${best['total_cost']:,.0f} / ${TOTAL_BUDGET:,}")
        print(f"  Area:     {best['area_mm2']} mm²  (limit: {MAX_AREA_MM2} mm²)")

        # Summary table: best config per num_sa value
        print(f"\nBest config per num_sa:")
        seen_sa = {}
        for r in results:
            sa = r["num_sa"]
            if sa not in seen_sa:
                seen_sa[sa] = r
        for sa in sorted(seen_sa.keys()):
            r = seen_sa[sa]
            print(f"  num_sa={sa}: Λ={r['system_throughput_rps']:.2f} rps  "
                  f"hbm_bw={r['hbm_bw_GBps']} d_batch={r['decode_batch']} "
                  f"chips={r['max_chips']} "
                  f"TTFT={r['TTFT_total_ms']:.0f}ms TPOT={r['TPOT_ms']:.1f}ms")

        print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
