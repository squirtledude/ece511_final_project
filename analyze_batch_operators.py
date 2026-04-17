# analyze_batch_operators.py
import pandas as pd

PARALLELISM = "dp1-tp1-pp1-ep1-dpdcn1-tpdcn1-ppdcn1-epdcn1-bs"
BASE = "results/raw/gpt-oss-20b_2048_128"
VERSION = "v5p"

# Arithmetic intensity threshold for TPUv5p (FLOPs/byte)
THRESHOLD = 166.0

def load_decode_csv(batch_size):
    path = f"{BASE}/{PARALLELISM}{batch_size}/inference-{VERSION}_decode.csv"
    df = pd.read_csv(path)
    # Use per-invocation values (not multiplied by count)
    df["arith_intensity"] = df["FLOP Count"] / df["Bytes accessed"]
    df["bound"] = df["arith_intensity"].apply(
        lambda x: "Compute" if x > THRESHOLD else "Memory"
    )
    return df

bs1  = load_decode_csv(1)
bs64 = load_decode_csv(64)

# Print summary for each batch size
for bs, df in [(1, bs1), (64, bs64)]:
    print(f"\n{'='*60}")
    print(f"Batch Size = {bs}")
    print(f"{'='*60}")
    print(f"{'Description':<50} {'FLOPs':>12} {'Bytes':>12} {'AI':>8} {'Bound'}")
    print("-" * 100)
    for _, row in df.iterrows():
        desc = str(row["Description"])[:50]
        flops = row["FLOP Count"]
        byt   = row["Bytes accessed"]
        ai    = row["arith_intensity"]
        bound = row["bound"]
        print(f"{desc:<50} {flops:>12.0f} {byt:>12.0f} {ai:>8.2f} {bound}")

# Find operators that changed classification
print(f"\n{'='*60}")
print("Operators that CHANGED bound classification (bs1 -> bs64)")
print(f"{'='*60}")

merged = bs1[["Description", "bound"]].merge(
    bs64[["Description", "bound"]],
    on="Description",
    suffixes=("_bs1", "_bs64")
)
changed = merged[merged["bound_bs1"] != merged["bound_bs64"]]

if len(changed) == 0:
    print("No operators changed classification.")
else:
    for _, row in changed.iterrows():
        print(f"{row['Description'][:60]:<60} {row['bound_bs1']} -> {row['bound_bs64']}")
