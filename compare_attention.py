import json
import matplotlib.pyplot as plt
import os

from neusim.npusim.frontend.run_single_op import run_sim_single_op
from neusim.npusim.frontend import llm_ops_lib as ops_lib
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.LLMConfig import GptOssConfig

# ── Config ─────────────────────────────────────────────────────────────────────
CHIP_CONFIG_PATH = "configs/chips/tpuv5p.json"
MODEL_CONFIG_PATH = "configs/models/gpt-oss-20b.json"

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
BATCH_SIZE = 1

# ── Load configs ───────────────────────────────────────────────────────────────
with open(CHIP_CONFIG_PATH) as f:
    chip_cfg = ChipConfig(**json.load(f))

with open(MODEL_CONFIG_PATH) as f:
    model_cfg = GptOssConfig(**json.load(f))

# Extract model params
d_model     = model_cfg.d_model        # 2880
num_heads   = model_cfg.num_heads      # 64
num_kv_heads = model_cfg.num_kv_heads  # 8
d_head      = model_cfg.d_head         # 64
sliding_window_size = model_cfg.sliding_window_size  # 128

# ── Helper: sum flops and memory across a list of ops ─────────────────────────
def run_and_sum(op_list, chip_cfg):
    total_flops = 0
    total_mem   = 0
    for op in op_list:
        result = run_sim_single_op(op, chip_cfg)
        total_flops += result.stats.flop_count * result.stats.count
        total_mem   += result.stats.memory_traffic_bytes * result.stats.count
    return total_flops, total_mem

# ── Sweep sequence lengths ─────────────────────────────────────────────────────
full_flops, full_mem   = [], []
slide_flops, slide_mem = [], []

for seqlen in SEQ_LENS:
    print(f"Running seqlen={seqlen}...")

    # Full attention (prefill, flash attention, num_layers=1 for single-op comparison)
    full_ops = ops_lib.create_multi_head_attention(
        batch_size=BATCH_SIZE,
        input_seqlen=seqlen,
        output_seqlen=seqlen,
        decode_width=1,
        num_heads=num_heads,
        d_model=d_model,
        d_head=d_head,
        num_layers=1,
        config=model_cfg,
        is_decode=False,
        use_flash_attention=True,
        num_kv_heads=num_kv_heads,
    )
    f_flops, f_mem = run_and_sum(full_ops, chip_cfg)
    full_flops.append(f_flops)
    full_mem.append(f_mem / 1e9)  # convert to GB

    # Sliding window attention (prefill, num_layers=1)
    slide_ops = ops_lib.create_sliding_window_attention(
        batch_size=BATCH_SIZE,
        q_seqlen=seqlen,
        sliding_window_size=sliding_window_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_model=d_model,
        d_head=d_head,
        num_layers=1,
        config=model_cfg,
        is_decode=False,
        use_flash_attention=True,
    )
    s_flops, s_mem = run_and_sum(slide_ops, chip_cfg)
    slide_flops.append(s_flops)
    slide_mem.append(s_mem / 1e9)  # convert to GB

# ── Plot 1: FLOPs vs sequence length ──────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(SEQ_LENS, [f/1e9 for f in full_flops],  marker="o", label="Full Attention",           color="steelblue", linewidth=2)
ax.plot(SEQ_LENS, [f/1e9 for f in slide_flops], marker="s", label="Sliding Window (w=128)", color="tomato",    linewidth=2)
ax.set_xlabel("Input Sequence Length", fontsize=12)
ax.set_ylabel("FLOPs (GFLOPs)", fontsize=12)
ax.set_title("FLOPs: Full Attention vs. Sliding Window Attention", fontsize=13)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("plots/attn_flops_comparison.png", dpi=150)
plt.close(fig)
print("Saved plots/attn_flops_comparison.png")

# ── Plot 2: Memory traffic vs sequence length ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(SEQ_LENS, full_mem,  marker="o", label="Full Attention",           color="steelblue", linewidth=2)
ax.plot(SEQ_LENS, slide_mem, marker="s", label="Sliding Window (w=128)", color="tomato",    linewidth=2)
ax.set_xlabel("Input Sequence Length", fontsize=12)
ax.set_ylabel("Memory Traffic (GB)", fontsize=12)
ax.set_title("Memory Traffic: Full Attention vs. Sliding Window Attention", fontsize=13)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("plots/attn_memory_comparison.png", dpi=150)
plt.close(fig)
print("Saved plots/attn_memory_comparison.png")

# ── Print raw numbers for report reference ─────────────────────────────────────
print("\n=== Raw Results ===")
print(f"{'SeqLen':>8} | {'Full FLOPs (G)':>16} | {'Slide FLOPs (G)':>16} | {'Full Mem (GB)':>14} | {'Slide Mem (GB)':>14}")
print("-" * 80)
for i, s in enumerate(SEQ_LENS):
    print(f"{s:>8} | {full_flops[i]/1e9:>16.2f} | {slide_flops[i]/1e9:>16.2f} | {full_mem[i]:>14.4f} | {slide_mem[i]:>14.4f}")
