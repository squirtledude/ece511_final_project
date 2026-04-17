import matplotlib.pyplot as plt
import numpy as np
import os

batch_sizes = [1, 4, 16, 64]
ttft = [0.0347, 0.1386, 0.5544, 2.2177]
tpot = [2.63,   4.70,   13.03,  46.38]

os.makedirs("plots", exist_ok=True)

fig, ax1 = plt.subplots(figsize=(7, 4))

# Slight x offset so markers don't perfectly overlap
x = np.array([0, 1, 2, 3])
offset = 0.05

ax1.plot(x - offset, ttft, marker="o", color="steelblue", linewidth=2.5,
         markersize=8, label="Prefill (TTFT)", zorder=3)
ax1.set_xlabel("Batch Size", fontsize=12)
ax1.set_ylabel("TTFT (seconds)", fontsize=12, color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xticks(x)
ax1.set_xticklabels(batch_sizes)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(x + offset, tpot, marker="s", color="tomato", linewidth=2.5,
         markersize=8, label="Decode (TPOT)", zorder=3)
ax2.set_ylabel("TPOT (ms / token)", fontsize=12, color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.suptitle("Prefill and Decode Latency vs. Batch Size", fontsize=13)
fig.tight_layout()
fig.savefig("plots/batch_latency.png", dpi=150)
plt.close(fig)
print("Saved plots/batch_latency.png")
