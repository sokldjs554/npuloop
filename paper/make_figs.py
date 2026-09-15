"""Figures for the letter, drawn from results/*.json at IEEE column width.

    python paper/make_figs.py            # English -> paper/fig{1,2}*.pdf
    python paper/make_figs.py --lang ko  # Korean  -> paper/fig{1,2}*_ko.pdf

Matplotlib renders the labels with its own text engine, not with LaTeX, so the strings here are plain
text: a "\\%" would be drawn with the backslash visible.  The Korean figures need a Hangul font
installed (fonts-nanum or fonts-noto-cjk); without one the script stops instead of drawing tofu.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
COL = 3.45                      # IEEE two-column text width, in inches
RC = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6.5,
      "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.dpi": 300,
      "axes.linewidth": 0.6, "lines.linewidth": 1.0, "grid.linewidth": 0.3}

HANGUL_FONTS = ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR",
                "Malgun Gothic", "AppleGothic"]

T = {
    "en": {"suffix": "", "ylabel1": "codes differing (%)", "xlabel1": "node index (graph order)",
           "panels": ["ResNet-20 (CNN)", "ViT-128/6 (transformer)"],
           "propagated": "propagated", "local": "local (teacher-forced)", "local_ln": "local, LayerNorm",
           "ylabel2": "top-1 change vs. reference (pp)"},
    "ko": {"suffix": "_ko", "ylabel1": "코드 불일치 비율(%)", "xlabel1": "노드 번호(그래프 순서)",
           "panels": ["ResNet-20 (CNN)", "ViT-128/6 (트랜스포머)"],
           "propagated": "전파", "local": "국소(교사 강제)", "local_ln": "국소, LayerNorm",
           "ylabel2": "기준 대비 top-1 변화(%p)"},
}


def hangul_font():
    """Name of an installed Hangul font, or raise with the package to install."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in HANGUL_FONTS:
        if name in available:
            return name
    raise SystemExit("no Hangul font found for the Korean figures; install one, e.g.\n"
                     "    apt-get install -y fonts-nanum        # or fonts-noto-cjk\n"
                     "then delete ~/.cache/matplotlib so matplotlib rescans.")


def load(name):
    return json.load(open(os.path.join(ROOT, "results", f"{name}.json")))


SKIP = ("input", "output", "flatten", "transpose", "reshape", "const")


def fig_decomposition(t):
    """Per-node local (teacher-forced) vs propagated code mismatch for a CNN and the transformer."""
    rs = load("e15_fidelity")["records"]
    panels = list(zip(["resnet20_relu", "cust_vit"], t["panels"]))
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.7), sharex=False)
    for ax, (model, title) in zip(axes, panels):
        r = next(x for x in rs if x["model"] == model and x["scheme"] == "npu-default")
        rows = [x for x in r["per_layer_agreement"] if x["op"] not in SKIP]
        i = np.arange(len(rows))
        prop = np.array([x["mismatch_frac"] for x in rows]) * 100
        loc = np.array([x["local_mismatch_frac"] for x in rows]) * 100
        ax.plot(i, np.maximum(prop, 1e-2), "-", color="#1f77b4", label=t["propagated"])
        nz = loc > 0
        ax.plot(i[nz], loc[nz], ".", ms=3.5, color="#ff7f0e", label=t["local"])
        ln = [k for k, x in enumerate(rows) if x["op"] == "layernorm" and loc[k] > 0]
        if ln:
            ax.plot(ln, loc[ln], "s", ms=3.5, color="#d62728", label=t["local_ln"])
        ax.set_yscale("log"); ax.set_ylim(1e-2, 200); ax.set_xlim(-1, len(rows))
        # plain decade labels rather than 10^-2: mathtext would draw the exponent's minus from the
        # text font, and a Hangul font has no U+2212.
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.3, which="both")
        ax.set_ylabel(t["ylabel1"])
        ax.set_title(title, pad=2)
        ax.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    axes[-1].set_xlabel(t["xlabel1"])
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, f"fig1_decomposition{t['suffix']}.pdf"))
    plt.close(fig)


def fig_rounding(t):
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
    ax.set_ylabel(t["ylabel2"])
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, f"fig2_rounding{t['suffix']}.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(T), default="en")
    a = ap.parse_args()
    rc = dict(RC)
    if a.lang == "ko":
        # NanumGothic has no U+2212: fall back to DejaVu for it, and use the ASCII hyphen on the axes.
        rc.update({"font.family": [hangul_font(), "DejaVu Sans"], "axes.unicode_minus": False})
    plt.rcParams.update(rc)
    t = T[a.lang]
    fig_decomposition(t); fig_rounding(t)
    print("wrote", f"fig1_decomposition{t['suffix']}.pdf, fig2_rounding{t['suffix']}.pdf")


if __name__ == "__main__":
    main()
