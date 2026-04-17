import matplotlib.pyplot as plt
import numpy as np
import os

batch_sizes = [1, 4, 16, 64]
prefill_thr = [58988, 59099, 59102, 59103]
decode_thr  = [380,   851,   1228,  1380]

os.makedirs("plots", exist_ok=True)

fig, ax1 = plt.subplots(figsize=(7, 4))

x = np.array([0, 1, 2, 3])
offset = 0.05

ax1.plot(x - offset, prefill_thr, marker="o", color="steelblue", linewidth=2.5,
         markersize=8, label="Prefill Throughput", zorder=3)
ax1.set_xlabel("Batch Size", fontsize=12)
ax1.set_ylabel("Prefill Throughput (tokens/sec)", fontsize=12, color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xticks(x)
ax1.set_xticklabels(batch_sizes)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(x + offset, decode_thr, marker="s", color="tomato", linewidth=2.5,
         markersize=8, label="Decode Throughput", zorder=3)
ax2.set_ylabel("Decode Throughput (tokens/sec)", fontsize=12, color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.suptitle("Prefill and Decode Throughput vs. Batch Size", fontsize=13)
fig.tight_layout()
fig.savefig("plots/batch_throughput.png", dpi=150)
plt.close(fig)
print("Saved plots/batch_throughput.png")
