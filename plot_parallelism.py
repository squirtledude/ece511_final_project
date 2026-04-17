"""
Section 3.2: Parallelism Strategies Plotting Script
Produces 18 plots:
  - 6 latency plots (TTFT/TPOT vs TP degree, one per model per phase)
  - 6 throughput plots (tokens/sec vs TP degree, one per model per phase)
  - 6 ICI communication plots (bytes vs TP degree, one per model per phase)
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "neusim/run_scripts/parallelism_sweep.csv"
df = pd.read_csv(CSV_PATH)

MODELS = ["gpt-oss-120b", "deepseekv2-236b", "llama3_1-405b"]
BATCH_SIZES = [1, 4, 16, 64]
COLORS = {1: "steelblue", 4: "tomato", 16: "green", 64: "purple"}
MODEL_LABELS = {
    "gpt-oss-120b":    "GPT-OSS-120B",
    "deepseekv2-236b": "DeepSeek-v2-236B",
    "llama3_1-405b":   "Llama-3.1-405B",
}

os.makedirs("plots/parallelism", exist_ok=True)

# ── Helper ─────────────────────────────────────────────────────────────────────
def make_plot(metric, ylabel, title_prefix, filename_prefix, model, phase):
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["model"] == model]
    for bs in BATCH_SIZES:
        bsub = sub[sub["batch_size"] == bs].sort_values("tp")
        ax.plot(bsub["tp"], bsub[metric], marker="o", linewidth=2,
                color=COLORS[bs], label=f"batch={bs}")
    ax.set_xlabel("Tensor Parallelism Degree (TP)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title_prefix}\n{MODEL_LABELS[model]} — {phase}", fontsize=12)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = f"plots/parallelism/{filename_prefix}_{model}_{phase.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ── 1. Latency plots ───────────────────────────────────────────────────────────
for model in MODELS:
    make_plot("ttft_ms",  "TTFT (ms)",         "Prefill Latency vs. TP Degree",
              "latency", model, "Prefill")
    make_plot("tpot_ms",  "TPOT (ms / token)", "Decode Latency vs. TP Degree",
              "latency", model, "Decode")

# ── 2. Throughput plots ────────────────────────────────────────────────────────
for model in MODELS:
    make_plot("ss_pf_tps", "Throughput (tokens/sec)", "Prefill Throughput vs. TP Degree",
              "throughput", model, "Prefill")
    make_plot("ss_dc_tps", "Throughput (tokens/sec)", "Decode Throughput vs. TP Degree",
              "throughput", model, "Decode")

# ── 3. ICI communication plots ─────────────────────────────────────────────────
for model in MODELS:
    # Convert bytes to GB for readability
    df["pf_ici_GB"] = df["pf_ici_bytes"] / 1e9
    df["dc_ici_GB"] = df["dc_ici_bytes"] / 1e9
    make_plot("pf_ici_GB", "ICI Traffic (GB)", "Prefill ICI Communication vs. TP Degree",
              "ici", model, "Prefill")
    make_plot("dc_ici_GB", "ICI Traffic (GB)", "Decode ICI Communication vs. TP Degree",
              "ici", model, "Decode")

print("\nAll 18 plots saved to plots/parallelism/")
