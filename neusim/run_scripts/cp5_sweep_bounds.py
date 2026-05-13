#!/usr/bin/env python3
"""
cp5_sweep_bounds.py — Determine parameter bounds for the CP5 sweep.

Computes and prints:
  1. CP4 baseline feasibility (area, perimeter, cost at 256 chips)
  2. HBM bandwidth ceiling (highest BW before perimeter constraint fails)
  3. Minimum HBM size (per-chip memory footprint for GPT-OSS-120B)
"""

import json
from math import ceil
from pathlib import Path

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.frontend.llm_ops_generator import GptOssOpsGenerator
from neusim.run_scripts.sweep_chip_params import load_base_config
from feasibility import estimate_area, estimate_perimeter_usage, estimate_cost

NEUSIM_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_120B  = "configs/models/gpt-oss-120b.json"
CHIP_BASE   = "configs/chips/tpuv5p.json"

# Decode pool parallelism from CP4 best-throughput config
DECODE_TP   = 8
DECODE_PP   = 2
BATCH       = 64
INPUT_SEQ   = 2048
OUTPUT_SEQ  = 128
TOTAL_CHIPS = 256
BUDGET      = 450_000


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    with open(CHIP_BASE) as f:
        base_chip = json.load(f)

    # ------------------------------------------------------------------
    # 1. CP4 baseline feasibility check
    # ------------------------------------------------------------------
    section("1. CP4 Baseline Feasibility (TPUv5p, 256 chips)")

    chip = ChipConfig(**base_chip)
    area  = estimate_area(chip)
    perim = estimate_perimeter_usage(chip, area)
    cost  = estimate_cost(chip, area)

    print(f"  Die area:        {area.total_mm2:.2f} mm²  / 858 mm²  "
          f"({area.total_mm2/858*100:.1f}% of reticle)")
    print(f"  Perimeter util:  {perim.utilization:.2%}  (must be <= 100%)")
    print(f"  Cost per chip:   ${cost.total_usd:,.2f}")
    print(f"    Die:           ${cost.die_usd:,.2f}")
    print(f"    HBM:           ${cost.hbm_usd:,.2f}")
    print(f"    ICI:           ${cost.ici_usd:,.2f}")
    print(f"  Cluster cost:    ${TOTAL_CHIPS * cost.total_usd:,.0f} / ${BUDGET:,}  "
          f"({'OVER' if TOTAL_CHIPS * cost.total_usd > BUDGET else 'OK'} by "
          f"${abs(TOTAL_CHIPS * cost.total_usd - BUDGET):,.0f})")

    # ------------------------------------------------------------------
    # 2. HBM bandwidth ceiling (perimeter constraint)
    # ------------------------------------------------------------------
    section("2. HBM Bandwidth Ceiling (perimeter-limited)")

    prev_bw = base_chip["hbm_bw_GBps"]
    ceiling_bw = None
    print(f"  Sweeping HBM BW from {prev_bw} GB/s upward (step=250):")
    for hbm_bw in range(int(base_chip["hbm_bw_GBps"]), 12000, 250):
        test_chip = {**base_chip, "hbm_bw_GBps": hbm_bw}
        c = ChipConfig(**test_chip)
        a = estimate_area(c)
        p = estimate_perimeter_usage(c, a)
        status = "OK" if p.is_feasible else "FAIL"
        print(f"    hbm_bw={hbm_bw:5d}: perimeter={p.utilization:.2%}  [{status}]")
        if not p.is_feasible:
            ceiling_bw = prev_bw
            print(f"\n  --> Perimeter limit hit at {hbm_bw} GB/s")
            print(f"  --> Maximum feasible HBM BW: {ceiling_bw} GB/s")
            break
        prev_bw = hbm_bw

    # ------------------------------------------------------------------
    # 3. Minimum HBM size (model footprint per chip)
    # ------------------------------------------------------------------
    section("3. Minimum HBM Size (GPT-OSS-120B footprint per decode chip)")

    base_config = load_base_config(MODEL_120B, CHIP_BASE)
    base_config.update({
        "input_seqlen":               INPUT_SEQ,
        "output_seqlen":              OUTPUT_SEQ,
        "num_chips":                  DECODE_TP * DECODE_PP,
        "global_batch_size":          BATCH,
        "microbatch_size_ici":        ceil(BATCH / DECODE_PP),
        "microbatch_size_dcn":        BATCH,
        "data_parallelism_degree":    1,
        "tensor_parallelism_degree":  DECODE_TP,
        "pipeline_parallelism_degree": DECODE_PP,
        "data_parallel_degree_dcn":   1,
        "tensor_parallel_degree_dcn": 1,
        "pipeline_parallel_degree_dcn": 1,
    })

    ops = GptOssOpsGenerator(base_config)
    footprint_bytes = ops.compute_memory_footprint_bytes("decode")
    footprint_gb    = footprint_bytes / 1e9

    print(f"  Model:           GPT-OSS-120B")
    print(f"  Decode pool:     TP={DECODE_TP}, PP={DECODE_PP}, batch={BATCH}")
    print(f"  Chips per inst:  {DECODE_TP * DECODE_PP}")
    print(f"  Footprint/chip:  {footprint_gb:.2f} GB")
    print(f"  --> Minimum hbm_size_GB sweep lower bound: "
          f"{footprint_gb:.1f} GB  (use next standard size above this)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("Summary of Sweep Bounds")
    print(f"  hbm_bw_GBps:   [{int(base_chip['hbm_bw_GBps'])} (baseline), "
          f"..., {ceiling_bw} (perimeter limit)]")
    print(f"  hbm_size_GB:   [{footprint_gb:.1f} GB minimum → use 64, 96, 128, 160, 192]")
    print(f"  num_sa:        [2, 4, 8]  (>= 2 to stay within TTFT SLO; <= 8 baseline)")
    print(f"  decode_batch:  [64 (baseline), ..., ~{int(64 * 100 / 37.76)} (TPOT SLO at baseline BW)]")
    print(f"  prefill_batch: [64, 128]  (throughput saturated at 64 per CP3)")


if __name__ == "__main__":
    main()
