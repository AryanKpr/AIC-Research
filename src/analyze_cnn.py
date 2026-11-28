"""Per-bit, per-layer SDC and non-finite rates with 95% Wilson intervals, plus figures.

Usage: python src/analyze_cnn.py vgg cifar10 run1

Writes
  logs/summary_<model>_<data>_<tag>_layer_bit.csv   one row per (layer, bit)
  logs/summary_<model>_<data>_<tag>_bit.csv         one row per bit, all layers pooled
  figures/<model>_<data>_<tag>_heatmap.png          layers x bits, SDC and non-finite
  figures/<model>_<data>_<tag>_layers.png           one per-bit plot per layer, with intervals
  figures/<model>_<data>_<tag>_bits.png             pooled per-bit plot (comparable to the MLP)
  figures/<model>_compare_<tag>.png                 Fashion-MNIST vs CIFAR-10, once both exist
"""
import math, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
arch, data, tag = sys.argv[1], sys.argv[2], sys.argv[3]
LOGS, FIGS = os.path.join(ROOT, "logs"), os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)
DATA_NAMES = {"fmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
CLASSES = ["masked", "SDC", "non-finite"]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(df, keys):
    rows = []
    for key, g in df.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key)); row["field"] = g["field"].iloc[0]; row["n"] = len(g)
        for cls in CLASSES:
            k = int((g["class"] == cls).sum()); lo, hi = wilson(k, len(g))
            row.update({f"{cls}_k": k, f"{cls}_share": k / len(g), f"{cls}_lo": lo, f"{cls}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def load(d):
    return pd.read_csv(os.path.join(LOGS, f"campaign_{arch}_{d}_{tag}.csv"))


def yerr(s, cls):
    """Asymmetric Wilson error bars. Clipped at 0: at a share of exactly 0 or 1 the interval
    edge can land a rounding error (~1e-16) on the wrong side of the point."""
    p = s[f"{cls}_share"]
    return np.clip(np.vstack([p - s[f"{cls}_lo"], s[f"{cls}_hi"] - p]), 0, None)


def per_bit_axes(ax, s, nbits, title, legend=True):
    exp_lo = 10 if nbits == 16 else 23
    ax.axvspan(-0.5, exp_lo - 0.5, color="tab:green", alpha=0.07, label="mantissa")
    ax.axvspan(exp_lo - 0.5, nbits - 1.5, color="tab:orange", alpha=0.07, label="exponent")
    ax.axvspan(nbits - 1.5, nbits - 0.5, color="tab:purple", alpha=0.07, label="sign")
    for cls, color, shift in [("SDC", "tab:blue", -0.12), ("non-finite", "tab:red", 0.12)]:
        p = s[f"{cls}_share"]
        ax.errorbar(s["bit"] + shift, p, yerr=yerr(s, cls),
                    fmt="o", ms=3, color=color, capsize=2, lw=1, label=cls)
    ax.set_xlim(nbits - 0.5, -0.5); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(nbits)); ax.tick_params(labelsize=7); ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(fontsize=7, loc="upper right")


df = load(data)
nbits = int(df["bit"].max()) + 1
layers = list(dict.fromkeys(df["layer"]))           # network order, as in the file
base = f"{arch}_{data}_{tag}"
title_base = f"{arch.upper()} on {DATA_NAMES[data]}"

# ---------------- Tables ----------------
lb = summarize(df, ["layer", "bit"])
lb.to_csv(os.path.join(LOGS, f"summary_{base}_layer_bit.csv"), index=False)
pb = summarize(df, ["bit"]).sort_values("bit", ascending=False)
pb.to_csv(os.path.join(LOGS, f"summary_{base}_bit.csv"), index=False)

n_layer = df.groupby("layer").size() // nbits
print(f"{title_base}: {len(df)} faults, {len(layers)} layers, elements per layer: "
      + ", ".join(f"{l} {n_layer[l]}" for l in layers))
print("\nPooled over layers (layers weighted equally):")
print(pb[["bit", "field", "n", "masked_k", "SDC_k", "non-finite_k"]].to_string(index=False))
print("\nFailure rate (SDC + non-finite) per layer and field:")
fail = df.assign(failed=df["class"] != "masked")
print((fail.pivot_table(index="layer", columns="field", values="failed", aggfunc="mean", sort=False)
       [["sign", "exponent", "mantissa"]] * 100).round(1).to_string())

# ---------------- Heatmap: layers x bits ----------------
bits_order = list(range(nbits - 1, -1, -1))
fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(layers) + 1.8), sharey=True)
for ax, cls, cmap in [(axes[0], "SDC", "Blues"), (axes[1], "non-finite", "Reds")]:
    grid = lb.pivot(index="layer", columns="bit", values=f"{cls}_share").reindex(index=layers, columns=bits_order)
    im = ax.imshow(grid.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 else "black")
    ax.set_xticks(range(nbits)); ax.set_xticklabels(bits_order, fontsize=7)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers, fontsize=8)
    ax.set_xlabel("bit position (15 = sign, 14-10 exponent, 9-0 mantissa)" if nbits == 16 else "bit position")
    ax.set_title(f"{cls} share (%)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025)
fig.suptitle(f"{title_base}: outcome by layer and bit ({n_layer.iloc[0]} elements per layer)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"{base}_heatmap.png"), dpi=150); plt.close(fig)

# ---------------- Small multiples: one per-bit plot per layer ----------------
cols = 3 if len(layers) > 4 else len(layers)
rows = math.ceil(len(layers) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.6 * rows), squeeze=False)
for i, layer in enumerate(layers):
    s = lb[lb["layer"] == layer]
    per_bit_axes(axes[i // cols][i % cols], s, nbits, f"{layer} (n = {s['n'].iloc[0]} per bit)", legend=(i == 0))
for j in range(len(layers), rows * cols):
    axes[j // cols][j % cols].axis("off")
fig.supxlabel("bit position"); fig.supylabel("share of faults (95% Wilson CI)")
fig.suptitle(f"{title_base}: SDC and non-finite by bit, per layer", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"{base}_layers.png"), dpi=150); plt.close(fig)

# ---------------- Pooled per-bit plot ----------------
fig, ax = plt.subplots(figsize=(10, 4.5))
per_bit_axes(ax, pb, nbits, f"{title_base}: outcome by bit, all layers pooled (n = {pb['n'].iloc[0]} per bit)")
ax.set_xlabel("bit position"); ax.set_ylabel("share of faults (95% Wilson CI)")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"{base}_bits.png"), dpi=150); plt.close(fig)

# ---------------- Dataset comparison, once both campaigns exist ----------------
paths = {d: os.path.join(LOGS, f"campaign_{arch}_{d}_{tag}.csv") for d in DATA_NAMES}
if all(os.path.exists(p) for p in paths.values()):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, cls in [(axes[0], "SDC"), (axes[1], "non-finite")]:
        for (d, label), color, shift in zip(DATA_NAMES.items(), ["tab:green", "tab:purple"], [-0.12, 0.12]):
            s = summarize(load(d), ["bit"])
            p = s[f"{cls}_share"]
            ax.errorbar(s["bit"] + shift, p, yerr=yerr(s, cls),
                        fmt="o", ms=4, capsize=2, color=color, label=label)
        ax.set_xlim(nbits - 0.5, -0.5); ax.set_xticks(range(nbits)); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("bit position"); ax.set_title(f"{cls} share, all layers pooled"); ax.legend()
    axes[0].set_ylabel("share of faults (95% Wilson CI)")
    fig.suptitle(f"{arch.upper()}: Fashion-MNIST vs CIFAR-10", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"{arch}_compare_{tag}.png"), dpi=150); plt.close(fig)
    print(f"\nSaved comparison figure figures/{arch}_compare_{tag}.png")

print(f"\nSaved tables logs/summary_{base}_*.csv and figures figures/{base}_*.png")
