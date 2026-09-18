"""표와 그림 생성기 (장문판 전용). 4쪽 원고 쪽은 make_tables.py / make_figs.py가 담당한다.

    python paper/make_thesis.py     # -> paper/th_*.tex, paper/fig3_operators_ko.pdf

장문판은 국문 전용이라 언어 옵션이 없다. 4쪽 원고와 겹치는 표(E15/E16/E14/E17)는 다시 만들지 않고
tab{1,2,3,4}_rows_ko.tex를 그대로 \\input 한다.
"""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
LABEL = {"resnet20_relu": "ResNet-20 ReLU", "resnet20_silu": "ResNet-20 SiLU",
         "mnv2_050_relu6": "MobileNetV2-0.5", "cust_vit": "ViT-128/6", "cust_inception": "Inception-32",
         "imagenette_resnet20": "ResNet-20 (Imagenette)", "espcn_x2": "ESPCN $\\times 2$"}
OPNAME = {"conv": "합성곱", "linear": "선형", "pool": "평균 풀링", "layernorm": "LayerNorm",
          "softmax": "소프트맥스", "matmul": "행렬곱", "add": "잔차 덧셈", "lut": "LUT 활성함수",
          "concat": "연결(concat)"}
STRUCTURAL = ("input", "output", "flatten", "transpose", "reshape", "const")
HANGUL_FONTS = ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR"]


def load(name):
    return json.load(open(os.path.join(ROOT, "results", f"{name}.json")))


def write(name, spec, header, rows, tabcolsep=None):
    body = [f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []
    body += [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    body += rows + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, name), "w").write("\n".join(body) + "\n")
    print(f"{name}: {len(rows)} rows")


def operator_stats():
    """연산자별 국소/전파 불일치를 E15와 E17의 모든 레코드에 걸쳐 모은다."""
    loc, prop, mx = defaultdict(list), defaultdict(list), defaultdict(int)
    for name in ("e15_fidelity", "e17_dense_output"):
        for r in load(name)["records"]:
            for row in r["per_layer_agreement"]:
                if row["op"] in STRUCTURAL:
                    continue
                loc[row["op"]].append(row["local_mismatch_frac"] * 100)
                prop[row["op"]].append(row["mismatch_frac"] * 100)
                mx[row["op"]] = max(mx[row["op"]], row["local_max_abs"])
    return loc, prop, mx


def t_baselines():
    rows = []
    for r in load("e1_baselines")["records"]:
        c = r["cost"]["tiny-1tops"]; li = r["lint"]["tiny-1tops"]["scores"]
        rows.append(f"{LABEL.get(r['model'], r['model'])} & {r['params']:,} & {r['macs'] / 1e6:.1f}M & "
                    f"{r['float_acc'] * 100:.2f} & {c['total_cycles'] / 1e3:.1f}k & "
                    f"{c['array_utilization'] * 100:.1f} & {li['overall']:.1f} \\\\")
    write("th_baselines.tex", "lrrrrrr",
          "모델 & 파라미터 & MACs & float32 top-1(\\%) & 사이클 & 배열 활용률(\\%) & lint 점수", rows, tabcolsep="5pt")


def t_operators():
    loc, prop, mx = operator_stats()
    rows = []
    for op in sorted(loc, key=lambda o: -float(np.mean(loc[o]))):
        v, p = np.array(loc[op]), np.array(prop[op])
        nz = int((v > 0).sum())
        rng = "0" if v.max() == 0 else f"{v.min():.3f}--{v.max():.3f}"
        # 잔차 덧셈은 100개 중 5개만, 그것도 원소 100만 개당 몇 개 수준이라 소수 셋째 자리에서 0이 된다.
        mean = "0" if v.mean() == 0 else (f"{v.mean():.3f}" if v.mean() >= 5e-4 else "$<$0.001")
        rows.append(f"{OPNAME.get(op, op)} & {len(v)} & {nz} & {mean} & {rng} & {mx[op]} & {p.max():.1f} \\\\")
    write("th_operators.tex", "lrrrcrr",
          "연산자 & 노드 & 불일치 노드 & 국소 평균(\\%) & 국소 범위(\\%) & 국소 최대 $|\\Delta|$ & 전파 최대(\\%)",
          rows, tabcolsep="5pt")


def t_requant():
    d = load("e7_requant_ablation"); rs = d["records"]
    models = ["resnet20_relu", "resnet20_silu", "mnv2_050_relu6"]
    base = {m: next(r for r in rs if r["model"] == m and r["config"] == "tflite-m31-acc32-b32") for m in models}
    axes = [("곱셈기 비트", ["tflite-m31-acc32-b32", "tflite-m15-acc32-b32", "tflite-m7-acc32-b32", "tflite-m3-acc32-b32"]),
            ("누산기 비트", ["tflite-m31-acc24-b32", "tflite-m31-acc20-b32", "tflite-m31-acc16-b32"]),
            ("바이어스 비트", ["tflite-m31-acc32-b16", "tflite-m31-acc32-b12"])]
    rows = []
    for axis, cfgs in axes:
        for i, cfg in enumerate(cfgs):
            cells = []
            for m in models:
                r = next((x for x in rs if x["model"] == m and x["config"] == cfg), None)
                if r is None:
                    cells.append("---"); continue
                d_pp = (r["int_acc"] - base[m]["int_acc"]) * 100
                cells.append(f"${d_pp:+.2f}$" if cfg != "tflite-m31-acc32-b32" else "기준")
            label = "\\texttt{" + cfg.replace("_", "\\_") + "}"
            rows.append((f"{axis} & " if i == 0 else " & ") + label + " & " + " & ".join(cells) + " \\\\")
    write("th_requant.tex", "llccc",
          "축 & 설정 & " + " & ".join(LABEL[m] for m in models), rows, tabcolsep="4pt")


def t_requant_seeds():
    """E18: the same axes as t_requant(), with three checkpoints so the magnitudes can be quoted at all."""
    d = load("e18_requant_axes_seeds"); rs = d["records"]
    models = ["resnet20_relu", "resnet20_silu", "mnv2_050_relu6"]
    configs = d["meta"]["configs"]; ref = configs[0]
    rows = []
    for cfg in configs:
        if cfg == ref:
            continue
        cells = []
        for m in models:
            xs = sorted([r for r in rs if r["model"] == m and r["config"] == cfg], key=lambda r: r["seed"])
            if not xs:
                cells.append("---"); continue
            v = np.array([r["vs_reference"]["delta"] for r in xs]) * 100
            sd = f"{v.std(ddof=1):.2f}" if len(v) > 1 else "---"
            cells.append(f"${v.mean():+.2f} \\pm {sd}$")
        rows.append("\\texttt{" + cfg.replace("_", "\\_").replace("tflite-", "") + "} & " + " & ".join(cells) + " \\\\")
    write("th_requant_seeds.tex", "lccc",
          "설정 & " + " & ".join(LABEL[m] for m in models), rows, tabcolsep="5pt")


def t_adaround():
    """E19: AdaRound on both paths. The gain column is the point; the reconstruction column is the control."""
    rs = load("e19_adaround")["records"]
    rows = []
    for r in rs:
        fg, ig_ = r["fake_gain"], r["int_gain"]
        rel = np.array([L["rel_after"] / max(L["rel_before"], 1e-30) for L in r["layers"]])
        sch = "\\textsc{pc}" if r["scheme"] == "npu-default" else "\\textsc{pt}"
        rows.append(f"{LABEL[r['model']]} & {sch} & {np.median(rel):.2f} & "
                    f"{r['flipped_frac'] * 100:.1f} & ${fg['delta'] * 100:+.2f} \\pm {fg['se'] * 100:.2f}$ & "
                    f"${ig_['delta'] * 100:+.2f} \\pm {ig_['se'] * 100:.2f}$ \\\\")
    write("th_adaround.tex", "llcccc",
          "신경망 & 방식 & 재구성 오차비 & 방향 반전(\\%) & 모의 이득(\\%p) & 정수 이득(\\%p)", rows, tabcolsep="5pt")


def t_tflite():
    d = load("e11_tflite_crosscheck")
    rows = []
    for r in d["records"]:
        rows.append(f"\\texttt{{{r['rounding'].replace('_', chr(92) + '_')}}} & {r['engine']} & "
                    f"{r['output_mismatch_images']} / {r['images']} & {r['output_mismatch_elems']} / {r['output_elems']} & "
                    f"{r['top1_agreement'] * 100:.1f} & {'예' if r['exact'] else '아니오'} \\\\")
    xv = d["meta"].get("xnnpack_vs_reference", {})
    write("th_tflite.tex", "llcccc",
          "반올림 조합 & 엔진 & 불일치 이미지 & 불일치 원소 & top-1 일치(\\%) & 완전 일치", rows, tabcolsep="4pt")
    return xv


def t_vela():
    rows = []
    for r in load("e13_vela")["records"]:
        v, n = r["vela"], r["npuloop"]
        rows.append(f"\\texttt{{{r['net']}}} & {v['total_macs'] / 1e6:.2f}M & {n['total_macs'] / 1e6:.2f}M & "
                    f"{v['total_cycles'] / 1e3:.1f}k & {n['total_cycles'] / 1e3:.1f}k & {r['cycle_ratio']:.3f} \\\\")
    write("th_vela.tex", "lrrrrr",
          "합성 망 & Vela MACs & npuloop MACs & Vela 사이클 & npuloop 사이클 & 비율", rows, tabcolsep="5pt")


def t_vela_kind():
    """Vela와의 사이클 격차를 연산자 종류로 분해한다. 합계 비율 하나만 보면 원인을 알 수 없다."""
    tot = defaultdict(lambda: [0.0, 0.0, 0])
    for r in load("e13_vela")["records"]:
        for L in r["per_layer"]:
            if "vela_cycles" not in L:          # 계층 수가 안 맞는 행(각 망의 pool 1개)은 건너뛴다
                continue
            t = tot[L["kind"]]
            t[0] += L["vela_cycles"]; t[1] += L["npuloop_cycles"]; t[2] += 1
    gv = sum(v[0] for v in tot.values()); gn = sum(v[1] for v in tot.values()); gap = gv - gn
    rows = []
    for k, v in sorted(tot.items(), key=lambda x: -(x[1][0] - x[1][1])):
        g = v[0] - v[1]
        rows.append(f"\\texttt{{{k}}} & {v[2]} & {v[0]:,.0f} & {v[1]:,.0f} & {v[1] / v[0]:.3f} & "
                    f"{g / gap * 100:.1f} \\\\")
    rows.append("\\midrule")
    rows.append(f"합계 & {sum(v[2] for v in tot.values())} & {gv:,.0f} & {gn:,.0f} & {gn / gv:.3f} & 100.0 \\\\")
    write("th_vela_kind.tex", "lrrrrr",
          "연산자 & 계층 & Vela 사이클 & npuloop 사이클 & npuloop/Vela & 격차 비중(\\%)", rows, tabcolsep="5pt")


def hangul_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in HANGUL_FONTS:
        if n in have:
            return n
    raise SystemExit("한글 글꼴이 없습니다: apt-get install -y fonts-nanum 후 rm -rf ~/.cache/matplotlib")


def fig_operators():
    """연산자별 국소 불일치 분포 — 3장 동기의 그림."""
    loc, _, _ = operator_stats()
    plt.rcParams.update({"font.family": [hangul_font(), "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "figure.dpi": 300, "axes.linewidth": 0.6})
    ops = sorted(loc, key=lambda o: float(np.mean(loc[o])))
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    FLOOR = 1e-4                       # 0은 로그 축에 그릴 수 없으므로 바닥에 붙여 표시한다
    for i, op in enumerate(ops):
        v = np.array(loc[op]); m = max(v.mean(), FLOOR)
        lo, hi = max(v.min(), FLOOR), max(v.max(), FLOOR)
        color = "#d62728" if v.mean() > 1 else ("#bbbbbb" if v.max() == 0 else "#1f77b4")
        ax.plot([lo, hi], [i, i], "-", color=color, lw=3, alpha=.35, solid_capstyle="butt")
        ax.plot([m], [i], "o", color=color, ms=5)
        lbl = "0%" if v.max() == 0 else (f"{v.max():.3f}%" if v.max() < 0.01 else f"{v.max():.2f}%")
        ax.text(hi * 1.6, i, lbl, va="center", fontsize=7, color="#333")
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels([f"{OPNAME.get(o, o)}  (n={len(loc[o])})" for o in ops])
    ax.set_xscale("log"); ax.set_xlim(FLOOR * 0.6, 400)
    # 10^-4 대신 평범한 숫자로: mathtext 지수의 음수 부호를 한글 글꼴에서 가져오다 두부가 된다.
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("국소(교사 강제) 코드 불일치 비율(%) — 점은 평균, 선은 노드별 최소~최대")
    ax.grid(axis="x", alpha=.3, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig3_operators_ko.pdf"))
    plt.close(fig)
    print("fig3_operators_ko.pdf")


if __name__ == "__main__":
    t_baselines(); t_operators(); t_requant(); t_requant_seeds(); t_adaround(); t_tflite(); t_vela(); t_vela_kind(); fig_operators()
