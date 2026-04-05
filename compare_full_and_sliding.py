import json
import pandas as pd
import matplotlib.pyplot as plt

from neusim.configs.models.LLMConfig import GptOssConfig
from neusim.npusim.frontend import llm_ops_lib as ops_lib
from neusim.npusim.frontend import op_analysis_lib as analysis_lib

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
WINDOW = 128

def run_sim_op_list(ops, cfg):
    return analysis_lib.fill_operators_execution_info(
        ops, cfg, analyze_energy=False
    )

def total_flops(ops):
    return sum(op.stats.flop_count * op.stats.count for op in ops)

def total_mem_bytes(ops):
    return sum(op.stats.memory_traffic_bytes * op.stats.count for op in ops)

with open("configs/models/gpt-oss-20b.json", "r") as f:
    cfg_dict = json.load(f)

config = GptOssConfig.model_validate(cfg_dict)

rows = []

for L in SEQ_LENS:
    # Full attention, prefill-style
    full_ops = ops_lib.create_multi_head_attention(
        batch_size=1,
        input_seqlen=L,
        output_seqlen=128,
        decode_width=1,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        d_model=config.d_model,
        d_head=config.d_head,
        num_layers=1,
        config=config,
        is_decode=False,
        use_flash_attention=config.use_flash_attention,
        tensor_parallelism_axes=[1],
        ici_bw_GBps=config.ici_bw_GBps,
        description_prefix="compare-full-",
    )
    full_ops = run_sim_op_list(full_ops, config)

    print("FULL OPS LENGTH:", len(full_ops))
    print("FIRST FULL OP NAME:", full_ops[0].name)
    print("FIRST FULL OP STATS OBJECT:", full_ops[0].stats)
    print("FIRST FULL OP STATS DICT:", vars(full_ops[0].stats))

    rows.append({
        "attention_type": "full",
        "input_seqlen": L,
        "flops": total_flops(full_ops),
        "memory_bytes": total_mem_bytes(full_ops),
    })

    print("FULL TOTAL FLOPS:", total_flops(full_ops))
    print("FULL TOTAL MEM BYTES:", total_mem_bytes(full_ops))

    # Sliding window attention, prefill-style
    sw_ops = ops_lib.create_sliding_window_attention(
        batch_size=1,
        q_seqlen=L,
        sliding_window_size=WINDOW,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        d_model=config.d_model,
        d_head=config.d_head,
        num_layers=1,
        config=config,
        is_decode=False,
        use_flash_attention=config.use_flash_attention,
        tensor_parallelism_axes=[1],
        ici_bw_GBps=config.ici_bw_GBps,
        description_prefix="compare-sw-",
    )
    sw_ops = run_sim_op_list(sw_ops, config)

    print("SW OPS LENGTH:", len(sw_ops))
    print("FIRST SW OP NAME:", sw_ops[0].name)
    print("FIRST SW OP STATS OBJECT:", sw_ops[0].stats)
    print("FIRST SW OP STATS DICT:", vars(sw_ops[0].stats))

    rows.append({
        "attention_type": "sliding_window",
        "input_seqlen": L,
        "flops": total_flops(sw_ops),
        "memory_bytes": total_mem_bytes(sw_ops),
    })

    print("SW TOTAL FLOPS:", total_flops(sw_ops))
    print("SW TOTAL MEM BYTES:", total_mem_bytes(sw_ops))

df = pd.DataFrame(rows)
print(df)
df.to_csv("attention_compare.csv", index=False)

# FLOPs plot
plt.figure()
for attn_type in ["full", "sliding_window"]:
    sub = df[df["attention_type"] == attn_type]
    plt.plot(sub["input_seqlen"], sub["flops"], marker="o", label=attn_type)
plt.xlabel("Input Sequence Length")
plt.ylabel("FLOPs")
plt.xscale("log", base=2)
plt.legend()
plt.tight_layout()
plt.savefig("attention_flops.png")
plt.close()

# Memory traffic plot
plt.figure()
for attn_type in ["full", "sliding_window"]:
    sub = df[df["attention_type"] == attn_type]
    plt.plot(sub["input_seqlen"], sub["memory_bytes"], marker="o", label=attn_type)
plt.xlabel("Input Sequence Length")
plt.ylabel("Memory Traffic (Bytes * 1e8)")
plt.xscale("log", base=2)
plt.legend()
plt.tight_layout()
plt.savefig("attention_memory.png")
plt.close()