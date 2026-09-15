"""Table bodies for the ESL letter, written from results/*.json so the paper cannot quote a stale number.

    python paper/make_tables.py      # writes paper/tab{1,2,3}_rows.tex, each \\input by npuloop_esl.tex

Each file holds a complete tabular environment: \\input inside an alignment breaks booktabs' \\bottomrule
(TeX sees the row as unterminated), so the float inputs the whole table body instead.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
LABEL = {"resnet20_relu": "ResNet-20 ReLU", "resnet20_silu": "ResNet-20 SiLU", "mnv2_050_relu6": "MobileNetV2-0.5",
         "cust_vit": "ViT-128/6", "cust_inception": "Inception-32", "imagenette_resnet20": "ResNet-20 (Imagenette)"}
SCHEME = {"npu-default": r"\textsc{pc}", "per-tensor": r"\textsc{pt}"}


def load(name):
    return json.load(open(os.path.join(ROOT, "results", f"{name}.json")))


def write(name, spec, header, rows, tabcolsep=None):
    body = [f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []
    body += [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    body += rows + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, name), "w").write("\n".join(body) + "\n")
    print(f"{name}: {len(rows)} rows")


def tab1_fidelity():
    rows = []
    for r in load("e15_fidelity")["records"]:
        p, oc = r["int_vs_fake"], r["output_codes"]
        rows.append(f"{LABEL[r['model']]} & {SCHEME[r['scheme']]} & {r['float_acc'] * 100:.2f} & {r['fake_acc'] * 100:.2f} & "
                    f"{r['int_acc'] * 100:.2f} & ${p['delta'] * 100:+.2f}\\pm{p['se'] * 100:.2f}$ & "
                    f"$[{p['ci95'][0] * 100:+.2f},{p['ci95'][1] * 100:+.2f}]$ & {p['top1_agreement'] * 100:.2f} & "
                    f"{oc['mismatch_frac'] * 100:.1f} & {oc['images_with_any_mismatch'] / r['n_test'] * 100:.0f} \\\\")
    write("tab1_rows.tex", "llcccccccc",
          "Network & Sch. & float32 & fake & integer & $\\Delta$ (pp) & $95\\%$ CI & agree (\\%) & codes (\\%) & imgs (\\%)",
          rows)


def tab2_layernorm():
    rows = []
    for r in load("e16_ln_emulation")["records"]:
        p = r["int_vs_fake"]
        ln = "float32" if r["variant"] == "float-ln" else f"integer ({r['layernorms_emulated']})"
        rows.append(f"{SCHEME[r['scheme']]} & {ln} & {r['per_op']['layernorm']['local_mean'] * 100:.2f} & "
                    f"{r['output_codes']['mismatch_frac'] * 100:.1f} & {p['top1_agreement'] * 100:.2f} & "
                    f"${p['delta'] * 100:+.2f}\\pm{p['se'] * 100:.2f}$ \\\\")
    write("tab2_rows.tex", "llcccc",
          "Sch. & LayerNorm & local (\\%) & codes (\\%) & agree (\\%) & $\\Delta$ (pp)",
          rows, tabcolsep="4pt")


def tab3_rounding():
    d = load("e14_rounding_seeds"); rs = d["records"]
    models = ["resnet20_relu", "resnet20_silu", "mnv2_050_relu6"]
    rows = []
    for rd in [x for x in d["meta"]["roundings"] if x != "tflite"]:
        cells = []
        for m in models:
            v = np.array([r["vs_reference"]["delta"] for r in rs if r["model"] == m and r["rounding"] == rd]) * 100
            cells.append(f"${v.mean():+.2f}\\pm{v.std(ddof=1):.2f}$" if len(v) > 1 else "---")
        rows.append("\\texttt{" + rd.replace("_", "\\_") + "} & " + " & ".join(cells) + " \\\\")
    write("tab3_rows.tex", "lccc",
          "Mode & ResNet-20 ReLU & ResNet-20 SiLU & MobileNetV2-0.5",
          rows, tabcolsep="4pt")


if __name__ == "__main__":
    tab1_fidelity(); tab2_layernorm(); tab3_rounding()
