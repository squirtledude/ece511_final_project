#!/usr/bin/env python3
"""
run_pd_disagg.py — PD-disaggregated LLM serving simulation.
"""

import argparse, copy, json, os
from math import ceil
from pathlib import Path

import neusim.npusim.frontend.run_sim_lib as run_sim_lib
from neusim.npusim.frontend import memory_footprint_analysis_lib
from neusim.run_scripts.sweep_chip_params import load_base_config, _get_ops_generator

NEUSIM_ROOT = Path(__file__).resolve().parent.parent.parent


def build_config(base, tp, pp, batch, outfile, chip_version="5p"):
    cfg = copy.deepcopy(base)
    cfg.update({
        "num_chips": tp * pp,
        "global_batch_size": batch,
        "microbatch_size_ici": ceil(batch / pp),
        "microbatch_size_dcn": batch,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": tp,
        "pipeline_parallelism_degree": pp,
        "data_parallel_degree_dcn": 1,
        "tensor_parallel_degree_dcn": 1,
        "pipeline_parallel_degree_dcn": 1,
        "output_file_path": outfile,
    })
    model_name = cfg.get("model_name", "")
    is_moe = "deepseek" in model_name.lower() or "gpt-oss" in model_name.lower()
    pconfig = {"data_parallelism_degree": 1, "tensor_parallelism_degree": tp,
               "pipeline_parallelism_degree": pp, "data_parallel_degree_dcn": 1,
               "pipeline_parallel_degree_dcn": 1}
    if is_moe:
        pconfig["expert_parallelism_degree"] = 1
    axes = run_sim_lib.map_parallelism_to_ici_axes(model_name, chip_version, pconfig)
    cfg["num_data_parallel_axes"], cfg["num_tensor_parallel_axes"], cfg["num_pipeline_parallel_axes"] = axes[:3]
    if is_moe:
        cfg["num_expert_parallel_axes"] = axes[3]
    return cfg

def compute_kv_cache_transfer_time_ms(config, dcn_bw_GBps):
    # Force tp=1 since entire KV cache sent over DCN (not sharded)
    tp1_cfg = {**config, "tensor_parallelism_degree": 1, "tensor_parallel_degree_dcn": 1}
    kv_bytes = memory_footprint_analysis_lib.get_llm_inference_kv_cache_mem_requirement(tp1_cfg)
    return kv_bytes / (dcn_bw_GBps * 1e6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--chip", default="configs/chips/tpuv5p.json")
    parser.add_argument("--chip_version", default="5p")
    parser.add_argument("--prefill_tp",         type=int, required=True)
    parser.add_argument("--prefill_pp",         type=int, required=True)
    parser.add_argument("--prefill_batch_size", type=int, required=True)
    parser.add_argument("--decode_tp",          type=int, required=True)
    parser.add_argument("--decode_pp",          type=int, required=True)
    parser.add_argument("--decode_batch_size",  type=int, required=True)
    parser.add_argument("--total_chips",   type=int, default=None)
    parser.add_argument("--prefill_chips", type=int, default=None)
    parser.add_argument("--decode_chips",  type=int, default=None)
    parser.add_argument("--input_seqlen",  type=int,   default=None)
    parser.add_argument("--output_seqlen", type=int,   default=None)
    parser.add_argument("--dcn_bw_GBps",  type=float, default=200.0)
    parser.add_argument("--output_dir",   default="results/pd_disagg")
    args = parser.parse_args()

    out = Path(args.output_dir)
    base = load_base_config(args.model, args.chip)
    if args.input_seqlen:  base["input_seqlen"]  = args.input_seqlen
    if args.output_seqlen: base["output_seqlen"] = args.output_seqlen
    in_seq, out_seq = base["input_seqlen"], base["output_seqlen"]

    # Build configs
    p_cfg = build_config(base, args.prefill_tp, args.prefill_pp, args.prefill_batch_size,
                         str(out / "prefill/inference.csv"), args.chip_version)
    d_cfg = build_config(base, args.decode_tp,  args.decode_pp,  args.decode_batch_size,
                         str(out / "decode/inference.csv"),  args.chip_version)

    # Simulate
    for cfg in [p_cfg, d_cfg]:
        os.makedirs(os.path.dirname(cfg["output_file_path"]), exist_ok=True)
        _get_ops_generator(cfg).generate(dump_to_file=True, separate_prefill_decode=True)

    # Read stats from CSVs
    p_stats = run_sim_lib.get_statistics_from_trace_file(str(out / "prefill/inference_prefill.csv"))
    d_stats = run_sim_lib.get_statistics_from_trace_file(str(out / "decode/inference_decode.csv"))

    # TTFT (prefill pool) and TPOT (decode pool) — mirrors dump_stats_llm() from run_sim.py
    pp_p, pp_d = args.prefill_pp, args.decode_pp
    TTFT_sec = ((p_stats["total_execution_time_non_pp_ns"] + p_stats["total_pp_ici_time_ns"]) * pp_p
                + p_stats["total_pp_dcn_time_ns"]) / 1e9
    TPOT_ms  = ((d_stats["total_execution_time_non_pp_ns"] + d_stats["total_pp_ici_time_ns"]) * pp_d
                / out_seq + d_stats["total_pp_dcn_time_ns"] / out_seq) / 1e6

    # KV transfer
    t_KV_ms = compute_kv_cache_transfer_time_ms(p_cfg, args.dcn_bw_GBps)

    TTFT_total_ms = TTFT_sec * 1000 + t_KV_ms

    # Resolve chip allocation — three valid ways to specify:
    # 1) --total_chips alone → 50/50 split
    # 2) --total_chips + --prefill_chips → decode gets the remainder
    # 3) --prefill_chips + --decode_chips → total inferred
    #prefill_chips = -1
    #decode_chips = -1
    if args.prefill_chips and args.decode_chips:
        prefill_chips = args.prefill_chips
        decode_chips  = args.decode_chips
    elif args.total_chips and args.prefill_chips:
        prefill_chips = args.prefill_chips
        decode_chips  = args.total_chips - args.prefill_chips
    elif args.total_chips:
        prefill_chips = args.total_chips // 2
        decode_chips  = args.total_chips // 2
    else:
        raise ValueError("Must specify either --total_chips, or --prefill_chips + --decode_chips")


    # System throughput
    # =============================================================================================================
    # Each instance needs tp*pp chips (tp chips per pipeline stage, pp stages)
    # Divide chip count by chips-per-instance to get number of parallel independent instances
    n_p = prefill_chips // (args.prefill_tp * args.prefill_pp)
    n_d = decode_chips  // (args.decode_tp  * args.decode_pp)

    # Per-instance request throughput: one instance processes batch_size requests in TTFT seconds
    lambda_P = args.prefill_batch_size / TTFT_sec
    # Decode: one instance processes decode_batch_size requests, each taking TPOT * output_seqlen ms total
    lambda_D = args.decode_batch_size / (TPOT_ms * out_seq / 1000)

    # Total pool throughput = instances * per-instance rate
    # System is bottlenecked by whichever pool is slower
    sys_tput = min(n_p * lambda_P, n_d * lambda_D)
    # =============================================================================================================

    # Summary
    summary = {
        "TTFT_prefill_only_ms": round(TTFT_sec * 1000, 4),
        "kv_transfer_time_ms":  round(t_KV_ms, 4),
        "TTFT_total_ms":        round(TTFT_total_ms, 4),
        "TPOT_ms":              round(TPOT_ms, 4),
        "n_prefill_instances":  n_p,
        "n_decode_instances":   n_d,
        "prefill_instance_req_per_sec": round(lambda_P, 6),
        "decode_instance_req_per_sec":  round(lambda_D, 6),
        "system_throughput_req_per_sec": round(sys_tput, 6),
        "config": vars(args),
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"TTFT (prefill only): {TTFT_sec*1000:.2f} ms")
    print(f"KV transfer:         {t_KV_ms:.2f} ms")
    print(f"TTFT (total):        {TTFT_total_ms:.2f} ms")
    print(f"TPOT:                {TPOT_ms:.2f} ms/token")
    print(f"Prefill pool: {n_p} instances x {lambda_P:.3f} req/s = {n_p*lambda_P:.3f} req/s")
    print(f"Decode pool:  {n_d} instances x {lambda_D:.3f} req/s = {n_d*lambda_D:.3f} req/s")
    print(f"System throughput: {sys_tput:.3f} req/s")
    print(f"Summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
