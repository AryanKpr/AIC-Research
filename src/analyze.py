"""Step 6: per-bit class shares with 95% Wilson intervals, plus plots. Usage: python analyze.py run1"""
import math, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
df = pd.read_csv(os.path.join(ROOT, "logs", f"campaign_{tag}.csv"))
nbits = df["bit"].max() + 1
CLASSES = ["masked", "SDC", "non-finite"]

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def summarize(sub):
    rows = []
    for bit, g in sub.groupby("bit"):
        n = len(g)
        row = {"bit": bit, "field": g["field"].iloc[0], "n": n}
        for cls in CLASSES:
            k = int((g["class"] == cls).sum())
            lo, hi = wilson(k, n)
            row.update({f"{cls}_k": k, f"{cls}_share": k / n, f"{cls}_lo": lo, f"{cls}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bit", ascending=False)

def plot(summary, title, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    exp_lo = 10 if nbits == 16 else 23
    ax.axvspan(-0.5, exp_lo - 0.5, color="tab:green", alpha=0.08, label="mantissa")
    ax.axvspan(exp_lo - 0.5, nbits - 1.5, color="tab:orange", alpha=0.08, label="exponent")
    ax.axvspan(nbits - 1.5, nbits - 0.5, color="tab:purple", alpha=0.08, label="sign")
    for cls, color, shift in [("SDC", "tab:blue", -0.12), ("non-finite", "tab:red", 0.12)]:
        p = summary[f"{cls}_share"]
        err = [p - summary[f"{cls}_lo"], summary[f"{cls}_hi"] - p]  # Wilson is asymmetric
        ax.errorbar(summary["bit"] + shift, p, yerr=err, fmt="o", color=color, capsize=3, label=cls)
    ax.set_xticks(range(nbits)); ax.set_xlim(nbits - 0.5, -0.5)  # bit 15 on the left
    ax.set_xlabel("bit position"); ax.set_ylabel("share of faults (95% Wilson CI)")
    ax.set_ylim(-0.02, 1.02); ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("Saved", path)

os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
overall = summarize(df)
overall.to_csv(os.path.join(ROOT, "logs", f"summary_{tag}.csv"), index=False)
print(overall[["bit", "field", "n", "masked_k", "SDC_k", "non-finite_k"]].to_string(index=False))
plot(overall, f"Outcome by bit position (n = {overall['n'].iloc[0]} per bit)",
     os.path.join(ROOT, "figures", f"per_bit_{tag}.png"))

for name, sub in df.groupby("tensor"):  # extra figure: split by matrix
    s = summarize(sub)
    s.to_csv(os.path.join(ROOT, "logs", f"summary_{tag}_{name}.csv"), index=False)
    plot(s, f"{name}: outcome by bit position (n = {s['n'].iloc[0]} per bit)",
         os.path.join(ROOT, "figures", f"per_bit_{tag}_{name}.png"))
