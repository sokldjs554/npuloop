"""Generate the markdown result tables for README.md from results/*.json (run after the experiments)."""
import json, os, sys, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
LABEL = {"resnet20_relu": "ResNet-20 ReLU", "resnet20_silu": "ResNet-20 SiLU", "resnet20_hswish": "ResNet-20 HardSwish",
         "resnet20_gelu": "ResNet-20 GELU", "mnv2_050_relu6": "MobileNetV2-0.5 ReLU6",
         "cust_vit": "고객 A · ViT-128/6", "cust_inception": "고객 B · Inception-32"}


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
    out = ["| 모델 | 파라미터 | MACs | val acc (선택 epoch) | test acc | " + " | ".join(f"{s} cycles (util)" for s in specs) + " | edge-10tops µJ/장 (simulated) | lint eff / q-rob |", "|---|---|---|---|---|" + "---|" * len(specs) + "---|---|"]
    for r in rs:
        cells = [f"{r['cost'][s]['total_cycles']:,.0f} ({r['cost'][s]['array_utilization']*100:.0f}%)" for s in specs]
        sc = r["lint"]["edge-10tops"]["scores"]
        val = f"{pct(r['val_acc'])} (ep {r['selected_epoch']})" if "val_acc" in r else "—"
        e = r["cost"]["edge-10tops"].get("energy_uj")
        energy = f"{e:.1f}" if e is not None else "—"
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['params']:,} | {r['macs']/1e6:.1f}M | {val} | {pct(r['float_acc'])} | " + " | ".join(cells) + f" | {energy} | {sc['efficiency']:.0f} / {sc['quant_robustness']:.0f} |")
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
        out += ["", "**(c) PTQ → QAT (2 epochs × 250 steps, lr 0.002)**", "", "| 모델 | 스킴 | FP32 | PTQ | QAT fake | QAT int | QAT 이득 |", "|---|---|---|---|---|---|---|"]
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


def e9():
    rs = load("e9_customer_intake")["records"]
    if not rs: return ""
    rs = sorted(rs, key=lambda r: (not r.get("customer"), r["model"]))
    out = ["**(a) 인테이크 요약** — `npuloop intake`가 낸 값 (사이클은 비용 모델, INT8은 정수 엔진 실측)", "",
           "| 모델 | 파라미터 | MACs | edge-10tops cycles (활용률) | strict cycles | 온칩 실행 | lint eff / q-rob |",
           "|---|---|---|---|---|---|---|"]
    for r in rs:
        d = r["intake"]["diagnose"]; ds = r["intake_strict"]["diagnose"]
        ok = "예" if r["intake"]["receive"]["runs_on_chip"] else "**아니오**"
        out.append(f"| {r['label']} | {r['params']:,} | {r['macs']/1e6:.1f}M | {d['cycles']:,.0f} ({d['array_utilization']*100:.1f}%) | "
                   f"{ds['cycles']:,.0f} | {ok} / strict {'예' if r['intake_strict']['receive']['runs_on_chip'] else '아니오'} | "
                   f"{d['lint']['scores']['efficiency']:.0f} / {d['lint']['scores']['quant_robustness']:.0f} |")
    done = [r for r in rs if r.get("after")]
    if done:
        out += ["", "**(b) 처방을 실제로 적용한 결과**", "",
                "| 모델 | 처방 | FP32 | INT8 정수 엔진 | edge-10tops cycles | strict cycles |", "|---|---|---|---|---|---|"]
        for r in done:
            a = r["after"]; d = r["intake"]["diagnose"]; ds = r["intake_strict"]["diagnose"]
            out.append(f"| {r['label']} | {a['action']} | {pct(r['float_acc'])} → {pct(a['float_acc'])} | "
                       f"{pct(r['ptq']['int_acc'])} → {pct(a['int_acc'])} | {d['cycles']:,.0f} → {a['cycles']:,.0f} | "
                       f"{ds['cycles']:,.0f} → {a['cycles_strict']:,.0f} |")
    out.append("\n온칩 실행 = 모든 op가 NPU에서 실행됨(호스트 폴백 없음). strict = LUT·softmax·layernorm 지원이 없는 프리셋.")
    return "\n".join(out)


def e10():
    rs = load("e10_engine_timing")["records"]
    if not rs: return ""
    meta = load("e10_engine_timing").get("meta", {})
    out = [f"이 호스트({rs[0]['cpu']}, 스레드 1개)에서 {meta.get('batch', 64)}장 배치를 {meta.get('repeats', 5)}회 돌린 중앙값입니다. "
           "**검증 엔진의 실측이지 NPU 지연이 아닙니다** — 마지막 열은 같은 모델의 비용 모델 값이며 둘은 다른 질문에 답합니다.", "",
           "| 모델 | 노드 | NumPy 엔진 ms/장 | C++ 엔진 ms/장 | C++/NumPy | 가장 비싼 op (C++ 기준) | 두 엔진 출력 코드 동일 | 비용 모델 edge-10tops (simulated) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rs:
        top = sorted(r["by_op"].items(), key=lambda kv: -kv[1]["cpp_ms"])[:2]
        top_txt = " · ".join(f"{op} {v['cpp_ms'] / r['cpp']['total_ms'] * 100:.0f}%" for op, v in top)
        eq = r.get("output_equality")
        eq_txt = (f"{eq['images'] - eq['images_with_any_mismatch']:,}/{eq['images']:,}장" if eq else "—")
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['nodes']} | {r['numpy']['per_image_ms']:.1f} | {r['cpp']['per_image_ms']:.1f} | "
                   f"{r['speedup']:.1f}× | {top_txt} | {eq_txt} | {r['modelled_cycles']:,.0f} cycles ({r['modelled_latency_ms']:.3f} ms) |")
    return "\n".join(out)


def e11():
    d = load("e11_tflite_crosscheck"); rs = d["records"]
    if not rs: return ""
    meta = d.get("meta", {}); xnn = meta.get("xnnpack_vs_reference", {})
    out = [f"TensorFlow {meta.get('tensorflow', '?')}, TFLite full-integer PTQ (int8 in/out), 연산: {' → '.join(meta.get('ops', []))}. "
           f"TFLite가 정한 스케일·zero-point·int8 가중치·int32 바이어스를 그대로 읽어 npuloop IntGraph를 만들고, 같은 int8 입력 {meta.get('images', '?'):,}장을 두 런타임에 넣었습니다.", "",
           "| 반올림 (conv·pool / fc) | 엔진 | 출력 코드가 다른 이미지 | 다른 원소 | top-1 일치 | conv1 / conv2 / conv3 / pool / fc 국소 불일치 |",
           "|---|---|---|---|---|---|"]
    for r in rs:
        pn = list(r["per_node"].values())
        cells = " / ".join(f"{v['mismatch_frac'] * 100:.2f}%" for v in pn)
        out.append(f"| {r['graph_rounding']} / {r['fc_rounding']} | {r['engine']} | {r['output_mismatch_images']}/{r['images']} | "
                   f"{r['output_mismatch_elems']}/{r['output_elems']} | {pct(r['top1_agreement'], 1)} | {cells} |")
    if xnn:
        out.append(f"\nTFLite 자신의 XNNPACK 델리게이트(최적화 경로)와 reference 커널은 같은 모델·입력에서 출력 코드가 **{xnn['output_mismatch_images']}/{xnn['images']}장** 다릅니다.")
    return "\n".join(out)


def e12():
    d = load("e12_imagenette"); rs = d["records"]
    if not rs: return ""
    base = next((r for r in rs if r["kind"] == "baseline"), None)
    out = []
    if base:
        c = base["cost"]
        out += [f"ResNet-20(stem stride 2), 128×128 입력, 학습 {d['meta']['train']:,}장 / 검증 {d['meta']['val']:,}장 / test {d['meta']['test']:,}장, "
                f"{base['epochs']} epoch({base['train_minutes']:.0f}분, CPU). val {pct(base['val_acc'])} (ep {base['selected_epoch']}) → **test {pct(base['test_acc'])}**, "
                f"MACs {base['macs'] / 1e6:.0f}M (CIFAR ResNet-20의 4배).", "",
                "| 프리셋 | cycles | 지연 (simulated) | 배열 활용률 | DRAM | 에너지 µJ/장 (simulated) | lint eff / q-rob |", "|---|---|---|---|---|---|---|"]
        for spec, v in c.items():
            sc = base["lint"][spec]["scores"]
            out.append(f"| {spec} | {v['total_cycles']:,.0f} | {v['latency_ms']:.3f} ms | {v['array_utilization'] * 100:.1f}% | {v['dram_bytes'] / 1024:.0f} KB | {v['energy_uj']:.1f} | {sc['efficiency']:.0f} / {sc['quant_robustness']:.0f} |")
    ptq = [r for r in rs if r["kind"] == "ptq"]
    if ptq:
        out += ["", f"PTQ (test {ptq[0]['int_eval_images']:,}장 전부, 정수 정확도는 C++ 엔진):", "",
                "| 스킴 | FP32 | fake-quant | 정수 엔진 | 차이 |", "|---|---|---|---|---|"]
        for r in ptq:
            out.append(f"| {r['scheme']} | {pct(r['float_acc'])} | {pct(r['fake_acc'])} | {pct(r['int_acc'])} | {pp(r['int_acc'] - r['fake_acc'])} |")
        eq = next((r.get("output_equality") for r in ptq if r.get("output_equality")), None)
        b = next((r.get("bench") for r in ptq if r.get("bench")), None)
        if eq:
            out.append(f"\nNumPy·C++ 엔진 출력 코드 동일: {eq['images'] - eq['images_with_any_mismatch']:,}/{eq['images']:,}장.")
        if b:
            out.append(f"실측(검증 엔진, 1스레드, 128px): NumPy {b['engines']['numpy']['per_image_ms']:.0f} ms/장, C++ {b['engines']['cpp']['per_image_ms']:.0f} ms/장 ({b['speedup']:.1f}×).")
    return "\n".join(out)


TABLES = [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5), ("E6", e6), ("E7", e7), ("E8", e8), ("E9", e9), ("E10", e10), ("E11", e11), ("E12", e12)]


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
