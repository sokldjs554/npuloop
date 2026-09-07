"""Generate the markdown result tables for README.md from results/*.json (run after the experiments)."""
import json, os, sys, statistics
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
    out = ["**(A) fake-quant 정확도 (test 10,000장)**", "", "| 모델 (FP32) | " + " | ".join(schemes) + " |", "|---|" + "---|" * len(schemes)]
    for m in models:
        cells = []
        for s in schemes:
            r = next((x for x in rs if x["model"] == m and x["scheme"] == s), None)
            cells.append(f"{pct(r['fake_acc'])} ({pp(-r['drop_fake'], 2)})" if r else "—")
        fa = next(x for x in rs if x["model"] == m)["float_acc"]
        out.append(f"| {LABEL.get(m, m)} ({pct(fa)}) | " + " | ".join(cells) + " |")
    out += ["", "**(B) 비트 정확 정수 엔진 vs fake-quant — 같은 이미지에서**", "",
            "| 모델 | 스킴 | 이미지 수 | fake-quant | 정수 엔진 | 차이 | top-1 일치 (500장) | 출력 코드 불일치 | 첫 분기 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        a = r["agreement"]; fs = r.get("fake_acc_on_int_subset", r["fake_acc"])
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['scheme']} | {r['int_eval_images']:,} | {pct(fs)} | {pct(r['int_acc'])} | {pp(r['int_acc'] - fs)} | {pct(a['top1_agreement'],1)} | {pct(a['output_mismatch_frac'],0)} | {a['first_divergence']} |")
    return "\n".join(out)


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
    specs = ["tiny-1tops", "edge-10tops", "pcie-80tops"]
    out = ["| 모델 | 전략 | ratio | align | 남긴 채널 (블록 내부) | MACs | cycles tiny / edge / pcie | edge util | FT acc | INT8 acc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in dict.fromkeys(r["model"] for r in rs):
        base = next(r for r in rs if r["model"] == m and r["strategy"] == "none")
        for r in [x for x in rs if x["model"] == m]:
            ratio = f"{r['ratio']}"
            if r["strategy"] == "cost-greedy":
                ratio += " ✓" if r.get("target_reached") else f" ✗ ({r.get('achieved_ratio', float('nan')):.2f})"
            cyc = " / ".join(f"{r[sp]['cycles']/base[sp]['cycles']*100:.0f}%" for sp in specs if sp in r and sp in base)
            out.append(f"| {LABEL.get(m, m)} | {r['strategy']} | {ratio} | {r['align'] or ''} | {' '.join(map(str, r['keep']))} | {r['macs']/base['macs']*100:.0f}% | {cyc} | {r['edge-10tops']['util']*100:.0f}% | {pct(r['ft_acc'])} | {pct(r['int8_acc'])} |")
    out.append("\nMACs·cycles는 프루닝 전 대비. cost-greedy의 ratio는 edge-10tops 사이클 목표이며 ✓ = 도달, ✗ = 최소 채널 폭(8)에서 멈춤(괄호는 실제 달성 비율). FT acc = 3 epoch fine-tune 후 FP32, INT8 acc = npu-default PTQ fake-quant.")
    return "\n".join(out)


def e8():
    rs = load("e8_scalesim")["records"]
    if not rs: return ""
    out = ["| 모델 | 배열 | SCALE-Sim cycles | npuloop cycles | 비율 | 최악 레이어 오차 | fill/drain 없이 |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['rows']}×{r['cols']} | {r['total_scalesim']:,.0f} | {r['total_ours']:,.0f} | {r['ratio']:.4f} | {r['max_layer_abs_err']*100:.1f}% | {r['ratio_nofd']:.3f} |")
    return "\n".join(out)


def e3():
    d = load("e3_lint_vs_drop"); s = d.get("meta", {}).get("summary", {}); rs = d.get("records", [])
    if not s: return ""
    out = ["**(a) 레이어 수준** — 한 레이어의 가중치만 양자화했을 때의 Δloss vs lint의 정적 채널 범위 비율(BN folding 후 max/median)", "",
           "| 가중치 스킴 | 레이어 수 | Spearman ρ (범위 비율 vs Δloss) | Spearman ρ (범위 비율 vs −SQNR) |", "|---|---|---|---|"]
    for sch in ["per-tensor", "npu-default"]:
        if f"n_layers_{sch}" in s:
            out.append(f"| {sch} | {s[f'n_layers_{sch}']} | {s[f'layer_spearman_range_vs_dloss_{sch}']:.2f} | {s[f'layer_spearman_range_vs_sqnr_{sch}']:.2f} |")
    mp = [r for r in rs if r.get("kind") == "model"]
    if mp:
        out += ["", "**(b) 모델 수준** — lint 점수(edge-10tops) vs E2에서 측정한 FP32 대비 손실(%p; 양수 = 손실)", "",
                "| 모델 | lint q-rob | lint eff | 스킴 수 | 평균 손실 fake / int | 최악 int 손실 |", "|---|---|---|---|---|---|"]
        for m in dict.fromkeys(r["model"] for r in mp):
            xs = [r for r in mp if r["model"] == m]
            out.append(f"| {LABEL.get(m, m)} | {xs[0]['quant_robustness']:.0f} | {xs[0]['efficiency']:.0f} | {len(xs)} | "
                       f"{statistics.mean(r['drop_fake'] for r in xs)*100:+.2f} / {statistics.mean(r['drop_int'] for r in xs)*100:+.2f} | {max(r['drop_int'] for r in xs)*100:+.2f} |")
        for sch in ["per-tensor", "npu-default"]:
            k = f"model_spearman_robustness_vs_drop_{sch}"
            if k in s:
                out.append(f"\n모델 수준 Spearman(−q-rob vs fake 손실, {sch}, n = {len(set(r['model'] for r in mp))}): {s[k]:.2f}")
    return "\n".join(out)


TABLES = [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5), ("E6", e6), ("E7", e7), ("E8", e8)]


def inject(readme_path: str) -> int:
    """Replace the text between <!-- TABLE:Ex --> and <!-- /TABLE:Ex --> markers in README with fresh tables."""
    import re
    src = open(readme_path, encoding="utf-8").read()
    n = 0
    for name, fn in TABLES:
        t = fn()
        pat = re.compile(rf"(<!-- TABLE:{name} -->)(.*?)(<!-- /TABLE:{name} -->)", re.S)
        if pat.search(src):
            src = pat.sub(lambda m: f"{m.group(1)}\n{t if t else '_(아직 실행되지 않음)_'}\n{m.group(3)}", src); n += 1
    open(readme_path, "w", encoding="utf-8").write(src)
    return n


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--inject":
        print(f"injected {inject(sys.argv[2])} tables")
    else:
        for name, fn in TABLES:
            t = fn()
            if t:
                print(f"\n<!-- {name} -->\n{t}\n")
