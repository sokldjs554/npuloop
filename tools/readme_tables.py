"""Generate the markdown result tables for README.md from results/*.json (run after the experiments)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
LABEL = {"resnet20_relu": "ResNet-20 ReLU", "resnet20_silu": "ResNet-20 SiLU", "resnet20_hswish": "ResNet-20 HardSwish",
         "resnet20_gelu": "ResNet-20 GELU", "mnv2_050_relu6": "MobileNetV2-0.5 ReLU6"}


def load(name):
    p = os.path.join(R, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {"records": [], "meta": {}}


def pct(v, d=2):
    return f"{v*100:.{d}f}%"


def pp(v, d=2):
    return f"{v*100:+.{d}f}%p"


def e1():
    rs = load("e1_baselines")["records"]
    if not rs: return ""
    specs = ["tiny-1tops", "edge-10tops", "pcie-80tops"]
    out = ["| 모델 | 파라미터 | MACs | FP32 acc | " + " | ".join(f"{s} cycles (util)" for s in specs) + " | lint eff / q-rob |", "|---|---|---|---|" + "---|" * len(specs) + "---|"]
    for r in rs:
        cells = [f"{r['cost'][s]['total_cycles']:,.0f} ({r['cost'][s]['array_utilization']*100:.0f}%)" for s in specs]
        sc = r["lint"]["edge-10tops"]["scores"]
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['params']:,} | {r['macs']/1e6:.1f}M | {pct(r['float_acc'])} | " + " | ".join(cells) + f" | {sc['efficiency']:.0f} / {sc['quant_robustness']:.0f} |")
    return "\n".join(out)


def e2():
    rs = load("e2_ptq_grid")["records"]
    if not rs: return ""
    models = list(dict.fromkeys(r["model"] for r in rs)); schemes = list(dict.fromkeys(r["scheme"] for r in rs))
    out = ["| 모델 (FP32) | " + " | ".join(schemes) + " |", "|---|" + "---|" * len(schemes)]
    for m in models:
        cells = []
        for s in schemes:
            r = next((x for x in rs if x["model"] == m and x["scheme"] == s), None)
            cells.append(f"{pct(r['fake_acc'])} / {pct(r['int_acc'])}" if r else "—")
        fa = next(x for x in rs if x["model"] == m)["float_acc"]
        out.append(f"| {LABEL.get(m, m)} ({pct(fa)}) | " + " | ".join(cells) + " |")
    out.append("")
    out.append("fake-quant / 비트 정확 정수 엔진 정확도. 정수 엔진은 npu-default·per-tensor는 10,000장, 나머지는 2,000장에서 측정.")
    ag = ["", "| 모델 | 스킴 | top-1 일치 (500장) | 출력 코드 불일치 | 첫 분기 레이어 | max Δlogit |", "|---|---|---|---|---|---|"]
    for r in rs:
        if r["scheme"] in ("npu-default", "per-tensor"):
            a = r["agreement"]; ag.append(f"| {LABEL.get(r['model'], r['model'])} | {r['scheme']} | {pct(a['top1_agreement'],1)} | {pct(a['output_mismatch_frac'],1)} | {a['first_divergence']} | {a['logit_max_abs_diff']:.3f} |")
    return "\n".join(out + ag)


def e7():
    rs = load("e7_requant_ablation")["records"]
    if not rs: return ""
    models = list(dict.fromkeys(r["model"] for r in rs)); cfgs = list(dict.fromkeys(r["config"] for r in rs))
    out = ["| RequantConfig | " + " | ".join(f"{LABEL.get(m, m)}: acc / 기준 대비 top-1 일치" for m in models) + " |", "|---|" + "---|" * len(models)]
    for c in cfgs:
        cells = []
        for m in models:
            r = next((x for x in rs if x["model"] == m and x["config"] == c), None)
            cells.append(f"{pct(r['int_acc'])} / {pct(r['top1_agreement_vs_reference'],1)}" if r else "—")
        out.append(f"| `{c}` | " + " | ".join(cells) + " |")
    n = rs[0].get("n_images", "?")
    out.append(f"\n{n}장 기준. 기준 구현은 `tflite-m31-acc32-b32`(gemmlowp 이중 반올림). 곱셈기 비트·누산기 폭이 줄어들 때 무엇이 먼저 무너지는지 보세요.")
    return "\n".join(out)


def e4():
    rs = load("e4_surgery")["records"]
    if not rs: return ""
    out = []
    a = [r for r in rs if r["part"] == "a"]
    if a:
        out += ["**(a) MobileNetV2-0.5, per-tensor 가중치 NPU 가정**", "", "| 단계 | fake-quant | 정수 엔진 |", "|---|---|---|"]
        for r in [x for x in a if x["scheme"] == "per-tensor"]:
            out.append(f"| {r['step']} | {pct(r['fake_acc'])} | {pct(r['int_acc'])} |")
        pc = next((x for x in a if x["scheme"] == "npu-default" and x["step"] == "ptq"), None)
        if pc: out.append(f"\n(같은 모델을 per-channel 가중치로 PTQ하면 {pct(pc['fake_acc'])} — FP32 {pct(pc['float_acc'])})")
    b = [r for r in rs if r["part"] == "b"]
    if b:
        out += ["", "**(b) ResNet-20-SiLU 활성함수 교체 + healing**", "", "| 변형 | FP32 (교체 직후 → heal 후) | INT8 fake / int | edge-10tops cycles | strict NPU cycles |", "|---|---|---|---|---|"]
        for r in b:
            fp = f"{pct(r['float_acc_after_swap'])} → {pct(r['float_acc_after_heal'])}" if "float_acc_after_swap" in r else pct(r["float_acc"])
            out.append(f"| {r['variant']} | {fp} | {pct(r['fake_acc'])} / {pct(r['int_acc'])} | {r['cycles']['edge-10tops']:,.0f} | {r['cycles']['edge-10tops-strict']:,.0f} |")
    c = [r for r in rs if r["part"] == "c"]
    if c:
        out += ["", "**(c) PTQ → QAT (3 epochs)**", "", "| 모델 | 스킴 | FP32 | PTQ | QAT fake | QAT int | QAT 이득 |", "|---|---|---|---|---|---|---|"]
        for r in c:
            out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['scheme']} | {pct(r['float_acc'])} | {pct(r['ptq_acc'])} | {pct(r['qat_acc'])} | {pct(r['int_acc'])} | {pp(r['qat_acc']-r['ptq_acc'])} |")
    return "\n".join(out)


def e5():
    rs = load("e5_calibration")["records"]
    if not rs: return ""
    import statistics
    models = list(dict.fromkeys(r["model"] for r in rs)); sizes = sorted(set(r["n"] for r in rs))
    out = ["| 모델 | 스킴 | 샘플링 | " + " | ".join(str(n) for n in sizes) + " |", "|---|---|---|" + "---|" * len(sizes)]
    for m in models:
        for s in ["npu-default", "per-tensor"]:
            for bal in (False, True):
                cells = []
                for n in sizes:
                    v = [r["drop"] for r in rs if r["model"] == m and r["scheme"] == s and r["balanced"] == bal and r["n"] == n]
                    cells.append(f"{statistics.mean(v)*100:.2f} ± {(statistics.pstdev(v))*100:.2f}" if v else "—")
                if any(c != "—" for c in cells):
                    out.append(f"| {LABEL.get(m, m)} | {s} | {'균형' if bal else '무작위'} | " + " | ".join(cells) + " |")
    out.append("\nFP32 대비 정확도 손실(%p), 시드 3개 평균 ± 표준편차. 열 = 캘리브레이션 이미지 수.")
    return "\n".join(out)


def e6():
    rs = load("e6_pruning")["records"]
    if not rs: return ""
    out = ["| 모델 | 전략 | ratio | align | 남긴 채널 | MACs | edge-10tops cycles | util | FT acc | INT8 acc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in dict.fromkeys(r["model"] for r in rs):
        base = next(r for r in rs if r["model"] == m and r["strategy"] == "none")
        for r in [x for x in rs if x["model"] == m]:
            out.append(f"| {LABEL.get(m, m)} | {r['strategy']} | {r['ratio']} | {r['align'] or ''} | {' '.join(map(str, r['keep']))} | {r['macs']/base['macs']*100:.0f}% | {r['edge-10tops']['cycles']/base['edge-10tops']['cycles']*100:.0f}% | {r['edge-10tops']['util']*100:.0f}% | {pct(r['ft_acc'])} | {pct(r['int8_acc'])} |")
    return "\n".join(out)


def e8():
    rs = load("e8_scalesim")["records"]
    if not rs: return ""
    out = ["| 모델 | 배열 | SCALE-Sim cycles | npuloop cycles | 비율 | 최악 레이어 오차 | fill/drain 없이 |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['rows']}×{r['cols']} | {r['total_scalesim']:,.0f} | {r['total_ours']:,.0f} | {r['ratio']:.4f} | {r['max_layer_abs_err']*100:.1f}% | {r['ratio_nofd']:.3f} |")
    return "\n".join(out)


def e3():
    d = load("e3_lint_vs_drop"); s = d.get("meta", {}).get("summary", {})
    if not s: return ""
    return "\n".join(f"* `{k}`: {v:.3f}" if isinstance(v, float) else f"* `{k}`: {v}" for k, v in s.items())


if __name__ == "__main__":
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5), ("E6", e6), ("E7", e7), ("E8", e8)]:
        t = fn()
        if t:
            print(f"\n<!-- {name} -->\n{t}\n")
