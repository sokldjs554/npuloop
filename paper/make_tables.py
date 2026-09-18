"""Table bodies for the letter, written from results/*.json so the paper cannot quote a stale number.

    python paper/make_tables.py             # English -> paper/tab{1,2,3}_rows.tex
    python paper/make_tables.py --lang ko   # Korean  -> paper/tab{1,2,3}_rows_ko.tex

Each file holds a complete tabular environment: \\input inside an alignment breaks booktabs' \\bottomrule
(TeX sees the row as unterminated), so the float inputs the whole table body instead.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
LABEL = {"resnet20_relu": "ResNet-20 ReLU", "resnet20_silu": "ResNet-20 SiLU", "mnv2_050_relu6": "MobileNetV2-0.5",
         "cust_vit": "ViT-128/6", "cust_inception": "Inception-32", "imagenette_resnet20": "ResNet-20 (Imagenette)"}
SCHEME = {"npu-default": r"\textsc{pc}", "per-tensor": r"\textsc{pt}"}

T = {
    "en": {
        "suffix": "",
        "head1": "Network & Sch. & float32 & fake & integer & $\\Delta$ (pp) & $95\\%$ CI & agree (\\%) & codes (\\%) & imgs (\\%)",
        "head2": "Sch. & LayerNorm & local (\\%) & codes (\\%) & agree (\\%) & $\\Delta$ (pp)",
        "head3": "Mode & ResNet-20 ReLU & ResNet-20 SiLU & MobileNetV2-0.5",
        "head4": "Convs & Sch. & float32 & fake & integer & $\\Delta$ ($10^{-3}$\\,dB) & codes (\\%) & $\\max|\\Delta c|$",
        "float_ln": "float32", "int_ln": "integer ({n})",
    },
    "ko": {
        "suffix": "_ko",
        "head1": "신경망 & 방식 & float32 & 모의 & 정수 & $\\Delta$ (\\%p) & $95\\%$ CI & 일치(\\%) & 코드(\\%) & 이미지(\\%)",
        "head2": "방식 & LayerNorm & 국소(\\%) & 코드(\\%) & 일치(\\%) & $\\Delta$ (\\%p)",
        "head3": "반올림 & ResNet-20 ReLU & ResNet-20 SiLU & MobileNetV2-0.5",
        "head4": "합성곱 & 방식 & float32 & 모의 & 정수 & $\\Delta$ ($10^{-3}$\\,dB) & 코드(\\%) & $\\max|\\Delta c|$",
        "float_ln": "float32", "int_ln": "정수 ({n}개)",
    },
}


def load(name):
    return json.load(open(os.path.join(ROOT, "results", f"{name}.json")))


def write(name, spec, header, rows, tabcolsep=None):
    body = [f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []
    body += [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    body += rows + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, name), "w").write("\n".join(body) + "\n")
    print(f"{name}: {len(rows)} rows")


def tab1_fidelity(t):
    rows = []
    for r in load("e15_fidelity")["records"]:
        p, oc = r["int_vs_fake"], r["output_codes"]
        rows.append(f"{LABEL[r['model']]} & {SCHEME[r['scheme']]} & {r['float_acc'] * 100:.2f} & {r['fake_acc'] * 100:.2f} & "
                    f"{r['int_acc'] * 100:.2f} & ${p['delta'] * 100:+.2f}\\pm{p['se'] * 100:.2f}$ & "
                    f"$[{p['ci95'][0] * 100:+.2f},{p['ci95'][1] * 100:+.2f}]$ & {p['top1_agreement'] * 100:.2f} & "
                    f"{oc['mismatch_frac'] * 100:.1f} & {oc['images_with_any_mismatch'] / r['n_test'] * 100:.0f} \\\\")
    write(f"tab1_rows{t['suffix']}.tex", "llcccccccc", t["head1"], rows)


def tab2_layernorm(t):
    rows = []
    for r in load("e16_ln_emulation")["records"]:
        p = r["int_vs_fake"]
        ln = t["float_ln"] if r["variant"] == "float-ln" else t["int_ln"].format(n=r["layernorms_emulated"])
        rows.append(f"{SCHEME[r['scheme']]} & {ln} & {r['per_op']['layernorm']['local_mean'] * 100:.2f} & "
                    f"{r['output_codes']['mismatch_frac'] * 100:.1f} & {p['top1_agreement'] * 100:.2f} & "
                    f"${p['delta'] * 100:+.2f}\\pm{p['se'] * 100:.2f}$ \\\\")
    write(f"tab2_rows{t['suffix']}.tex", "llcccc", t["head2"], rows, tabcolsep="4pt")


def tab3_rounding(t):
    d = load("e14_rounding_seeds"); rs = d["records"]
    models = ["resnet20_relu", "resnet20_silu", "mnv2_050_relu6"]
    rows = []
    for rd in [x for x in d["meta"]["roundings"] if x != "tflite"]:
        cells = []
        for m in models:
            v = np.array([r["vs_reference"]["delta"] for r in rs if r["model"] == m and r["rounding"] == rd]) * 100
            cells.append(f"${v.mean():+.2f}\\pm{v.std(ddof=1):.2f}$" if len(v) > 1 else "---")
        rows.append("\\texttt{" + rd.replace("_", "\\_") + "} & " + " & ".join(cells) + " \\\\")
    write(f"tab3_rows{t['suffix']}.tex", "lccc", t["head3"], rows, tabcolsep="4pt")


def tab4_dense(t):
    """Two depths at the same width profile, so the propagated column can be read against depth."""
    rows, prev = [], None
    for r in sorted(load("e17_dense_output")["records"], key=lambda r: (r["conv_layers"], r["scheme"])):
        p, oc = r["int_vs_fake"], r["output_codes"]
        if prev is not None and r["conv_layers"] != prev:
            rows.append("\\midrule")
        depth = str(r["conv_layers"]) if r["conv_layers"] != prev else ""
        prev = r["conv_layers"]
        rows.append(f"{depth} & {SCHEME[r['scheme']]} & {r['float_psnr']:.3f} & {r['fake_psnr']:.3f} & {r['int_psnr']:.3f} & "
                    f"${p['delta'] * 1000:+.2f}\\pm{p['se'] * 1000:.2f}$ & {oc['mismatch_frac'] * 100:.1f} & "
                    f"{oc['max_abs_code_diff']} \\\\")
    write(f"tab4_rows{t['suffix']}.tex", "rlcccccc", t["head4"], rows, tabcolsep="4pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(T), default="en")
    t = T[ap.parse_args().lang]
    tab1_fidelity(t); tab2_layernorm(t); tab3_rounding(t); tab4_dense(t)
