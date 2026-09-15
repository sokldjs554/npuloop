"""Figures for the ESL letter, drawn from results/*.json at IEEE column width.

    python paper/make_figs.py        # writes paper/fig1_decomposition.pdf, paper/fig2_rounding.pdf
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
COL = 3.45                      # IEEE two-column text width, in inches
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.dpi": 300,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "grid.linewidth": 0.3})


def load(name):
    return json.load(open(os.path.join(ROOT, "results", f"{name}.json")))


SKIP = ("input", "output", "flatten", "transpose", "reshape", "const")


def fig_decomposition():
    """Per-node local (teacher-forced) vs propagated code mismatch for a CNN and the transformer."""
    rs = load("e15_fidelity")["records"]
    panels = [("resnet20_relu", "ResNet-20 (CNN)"), ("cust_vit", "ViT-128/6 (transformer)")]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.7), sharex=False)
    for ax, (model, title) in zip(axes, panels):
        r = next(x for x in rs if x["model"] == model and x["scheme"] == "npu-default")
        rows = [x for x in r["per_layer_agreement"] if x["op"] not in SKIP]
        i = np.arange(len(rows))
        prop = np.array([x["mismatch_frac"] for x in rows]) * 100
        loc = np.array([x["local_mismatch_frac"] for x in rows]) * 100
        ax.plot(i, np.maximum(prop, 1e-2), "-", color="#1f77b4", label="propagated")
        nz = loc > 0
        ax.plot(i[nz], loc[nz], ".", ms=3.5, color="#ff7f0e", label="local (teacher-forced)")
        ln = [k for k, x in enumerate(rows) if x["op"] == "layernorm" and loc[k] > 0]
        if ln:
            ax.plot(ln, loc[ln], "s", ms=3.5, color="#d62728", label="local, LayerNorm")
        ax.set_yscale("log"); ax.set_ylim(1e-2, 200); ax.set_xlim(-1, len(rows))
        ax.grid(alpha=.3, which="both")
        ax.set_ylabel("codes differing (\\%)")
        ax.set_title(title, pad=2)
        ax.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    axes[-1].set_xlabel("node index (graph order)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig1_decomposition.pdf"))
    plt.close(fig)


def fig_rounding():
    """Rounding-mode cost against the reference implementation, three seeds per model."""
    d = load("e14_rounding_seeds"); rs = d["records"]
    models = [("resnet20_relu", "ResNet-20 ReLU", "#1f77b4"),
              ("resnet20_silu", "ResNet-20 SiLU", "#ff7f0e"),
              ("mnv2_050_relu6", "MobileNetV2-0.5", "#2ca02c")]
    rounds = [r for r in (d["meta"].get("roundings") or []) if r != "tflite"]
    seeds = sorted({r["seed"] for r in rs})
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    w = 0.78 / len(models)
    for i, (m, label, color) in enumerate(models):
        for j, sd in enumerate(seeds):
            xs = [k + (i - (len(models) - 1) / 2) * w + (j - 1) * w / 4 for k in range(len(rounds))]
            ys = [next((r["vs_reference"]["delta"] * 100 for r in rs if r["model"] == m and r["seed"] == sd and r["rounding"] == rd), np.nan) for rd in rounds]
            es = [next((r["vs_reference"]["se"] * 100 for r in rs if r["model"] == m and r["seed"] == sd and r["rounding"] == rd), np.nan) for rd in rounds]
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=2.5, color=color, capsize=1.5, elinewidth=0.6,
                        label=label if j == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(rounds))); ax.set_xticklabels([r.replace("_", "-") for r in rounds])
    ax.set_ylabel("top-1 change vs.\\ reference (pp)")
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig2_rounding.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_decomposition(); fig_rounding()
    print("wrote", ", ".join(sorted(f for f in os.listdir(OUT) if f.endswith(".pdf"))))
