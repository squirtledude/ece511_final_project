import json
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_BASE = "/home/marshall/ece511/mp4/ece511_final_project/results/raw"
SEQ_LENS = [1024, 2048, 4096, 8192]
PARALLELISM = "dp1-tp1-pp1-ep1-dpdcn1-tpdcn1-ppdcn1-epdcn1-bs4"
VERSION = "v5p"

records = []

for seqlen in SEQ_LENS:
    folder = os.path.join(
        RESULTS_BASE,
        f"gpt-oss-20b_{seqlen}_128",
        PARALLELISM
    )

    for phase in ["prefill", "decode"]:
        # ── JSON: memory footprint ─────────────────────────────────────────────
        json_path = os.path.join(folder, f"inference-{VERSION}_{phase}.json")
        with open(json_path) as f:
            stats = json.load(f)
        mem_footprint_GB = stats["mem_footprint_GB"]

        # ── CSV: memory traffic and FLOPs ──────────────────────────────────────
        csv_path = os.path.join(folder, f"inference-{VERSION}_{phase}.csv")
        df = pd.read_csv(csv_path)

        # Each row already accounts for count in Execution time,
        # but Bytes accessed and FLOP Count are per-invocation.
        # Multiply by Count to get totals.
        total_mem_traffic_GB = (df["Bytes accessed"] * df["Count"]).sum() / 1e9
        total_flops_G        = (df["FLOP Count"]     * df["Count"]).sum() / 1e9

        records.append({
            "seqlen":           seqlen,
            "phase":            phase,
            "mem_footprint_GB": mem_footprint_GB,
            "mem_traffic_GB":   total_mem_traffic_GB,
            "flops_G":          total_flops_G,
        })
        print(f"seqlen={seqlen:5d} {phase:7s} | "
              f"footprint={mem_footprint_GB:.2f} GB | "
              f"traffic={total_mem_traffic_GB:.2f} GB | "
              f"FLOPs={total_flops_G:.2f} GFLOPs")

result_df = pd.DataFrame(records)
prefill = result_df[result_df["phase"] == "prefill"]
decode  = result_df[result_df["phase"] == "decode"]

os.makedirs("plots", exist_ok=True)

# ── Plot 1: Memory Footprint ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(prefill["seqlen"], prefill["mem_footprint_GB"], marker="o", color="steelblue", linewidth=2, label="Prefill")
ax.plot(decode["seqlen"],  decode["mem_footprint_GB"],  marker="s", color="tomato",    linewidth=2, label="Decode")
ax.set_xlabel("Input Sequence Length", fontsize=12)
ax.set_ylabel("Memory Footprint (GB)", fontsize=12)
ax.set_title("Memory Footprint vs. Input Sequence Length", fontsize=13)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("plots/scaling_mem_footprint.png", dpi=150)
plt.close(fig)
print("Saved plots/scaling_mem_footprint.png")

# ── Plot 2: Memory Traffic ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(prefill["seqlen"], prefill["mem_traffic_GB"], marker="o", color="steelblue", linewidth=2, label="Prefill")
ax.plot(decode["seqlen"],  decode["mem_traffic_GB"],  marker="s", color="tomato",    linewidth=2, label="Decode")
ax.set_xlabel("Input Sequence Length", fontsize=12)
ax.set_ylabel("Memory Traffic (GB)", fontsize=12)
ax.set_title("Memory Traffic vs. Input Sequence Length", fontsize=13)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("plots/scaling_mem_traffic.png", dpi=150)
plt.close(fig)
print("Saved plots/scaling_mem_traffic.png")

# ── Plot 3: FLOPs ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(prefill["seqlen"], prefill["flops_G"], marker="o", color="steelblue", linewidth=2, label="Prefill")
ax.plot(decode["seqlen"],  decode["flops_G"],  marker="s", color="tomato",    linewidth=2, label="Decode")
ax.set_xlabel("Input Sequence Length", fontsize=12)
ax.set_ylabel("FLOPs (GFLOPs)", fontsize=12)
ax.set_title("Compute (FLOPs) vs. Input Sequence Length", fontsize=13)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("plots/scaling_flops.png", dpi=150)
plt.close(fig)
print("Saved plots/scaling_flops.png")
