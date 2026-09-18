"""Generate the markdown result tables for README.md from results/*.json (run after the experiments)."""
import json, os, sys, statistics
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
LABEL = {"espcn_x2": "ESPCN ×2 (초해상)", "resnet20_relu": "ResNet-20 ReLU", "resnet20_silu": "ResNet-20 SiLU", "resnet20_hswish": "ResNet-20 HardSwish",
         "resnet20_gelu": "ResNet-20 GELU", "mnv2_050_relu6": "MobileNetV2-0.5 ReLU6",
         "cust_vit": "고객 A · ViT-128/6", "cust_inception": "고객 B · Inception-32", "imagenette_resnet20": "ResNet-20 (Imagenette-128)"}


def load(name):
    p = os.path.join(R, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {"records": [], "meta": {}}


def pct(v, d=2):
    return f"{v*100:.{d}f}%"


def pp(v, d=2):
    return f"{v*100:+.{d}f}%p"


def headline():
    """The one table the README leads with: what each question is answered by, and what is NOT verified.

    Every number comes from results/*.json; the caveat in the third column is a literal that lives in the same
    row as the number it qualifies, so the two cannot drift apart through an edit. A row whose result file is
    missing or malformed degrades to a dash instead of taking the whole injection down with it.
    """
    def row(q, fn, caveat):
        try:
            return f"| {q} | {fn()} | {caveat} |"
        except Exception:
            return f"| {q} | — | {caveat} |"

    def e11_answer():
        rs = load("e11_tflite_crosscheck")
        best = next(r for r in rs["records"] if r["engine"] == "numpy" and r["output_mismatch_elems"] == 0)
        return (f"TFLite reference 커널과 **{best['images']:,}장 × 모든 텐서 0 불일치**"
                f"(conv·pool은 gemmlowp 이중 반올림, fc는 단일 반올림)")
    def e8_answer():
        rs = load("e8_scalesim")["records"]
        return (f"SCALE-Sim v3와 {len(rs)}개 (모델, 배열) 조합에서 합계 오차 ≤ {max(abs(r['ratio'] - 1) for r in rs) * 100:.2f}%, "
                f"최악 레이어 {max(r['max_layer_abs_err'] for r in rs) * 100:.1f}%")
    def e13_answer():
        rs = load("e13_vela")["records"]
        v = [r["cycle_ratio"] for r in rs]
        return f"Arm Vela 대비 **{len(v)}/{len(v)} 낙관적**(사이클 비 {min(v):.2f}~{max(v):.2f})"
    def e2_answer():
        try:
            from .submission_quality import fidelity_summary
        except ImportError:
            from submission_quality import fidelity_summary
        return fidelity_summary(load("e15_fidelity")["records"])
    def e15_answer():
        rs = load("e15_fidelity")["records"]
        c = [r["output_codes"]["mismatch_frac"] for r in rs]
        e16 = load("e16_ln_emulation")["records"]
        a = next(r for r in e16 if r["scheme"] == "npu-default" and r["variant"] == "float-ln")
        b = next(r for r in e16 if r["scheme"] == "npu-default" and r["variant"] == "int-ln")
        return (f"그러나 출력 코드는 {min(c) * 100:.1f}~{max(c) * 100:.1f}%가 다릅니다. 가장 큰 국소 원천(ViT의 LayerNorm)을 "
                f"정수 산술로 바꿔 국소 불일치를 {pct(a['per_op']['layernorm']['local_mean'], 2)} → {pct(b['per_op']['layernorm']['local_mean'], 2)}로 없애도 "
                f"출력 코드 불일치는 {pct(a['output_codes']['mismatch_frac'], 1)} → {pct(b['output_codes']['mismatch_frac'], 1)}까지만 내려갑니다")
    def e14_answer():
        rs = [r for r in load("e14_rounding_seeds")["records"] if r["rounding"] in ("truncate", "floor")]
        d = [r["vs_reference"]["delta"] * 100 for r in rs]
        z = [abs(r["vs_reference"]["delta"] / r["vs_reference"]["se"]) for r in rs if r["vs_reference"]["se"]]
        return (f"requant에서 반올림 대신 시프트를 쓰면 {len(rs)}개 (모델, 시드, 모드) 조합 "
                f"**{sum(1 for x in d if x < 0)}개 전부**에서 손해입니다({min(d):+.2f}~{max(d):+.2f}%p, \\|Δ\\|/SE {min(z):.1f}~{max(z):.1f})")
    def e7_answer():
        rs = [r for r in load("e7_requant_ablation")["records"] if r["model"] == "resnet20_relu"]
        lossless = [r for r in rs if r["top1_agreement_vs_reference"] > 0.98]
        collapse = [r for r in rs if r["int_acc"] < 0.5]
        return (f"{len(rs)}개 구현 구성 중 {len(lossless)}개는 무손실, {len(collapse)}개는 모델을 무너뜨립니다 "
                f"(곱셈기 7비트·누산기 20비트까지는 공짜, 바이어스 12비트·누산기 16비트는 붕괴)")

    rows = [
        ("**정수 엔진을 믿어도 되는가** ([E11](docs/EXPERIMENTS.md#e11))", e11_answer,
         "대조 범위는 conv(stride 1·2)·MEAN·fully-connected. depthwise·add·softmax는 미대조이고, TFLite 자신도 XNNPACK 경로와 reference가 155/1,000장 다릅니다"),
        ("**비용 모델이 맞는가** ([E8](docs/EXPERIMENTS.md#e8))", e8_answer,
         "검증된 것은 dense conv·linear의 연산 사이클(단일 코어, 메모리 스톨 없음). depthwise·벡터 패스·멀티코어·DRAM roofline은 검증되지 않았습니다"),
        ("**그럼 실리콘에 가까운가** ([E13](docs/EXPERIMENTS.md#e13))", e13_answer,
         "**아니오.** 둘 다 해석적 추정기이고 Vela는 컴파일된 스케줄(fusion·타일링)을, 이쪽은 레이어를 하나씩 셉니다. E8의 일치는 하드웨어 충실도가 아니라 같은 이상화를 공유하는 두 모델이 같은 식을 같게 구현했다는 확인입니다"),
        ("**fake-quant 정확도를 믿어도 되는가** ([E2](docs/EXPERIMENTS.md#e2)·[E15](docs/EXPERIMENTS.md#e15))", e2_answer,
         "E15·E16의 16개 비교를 다중비교 보정하지 않은 값입니다(E16까지 합치면 최대 \\|Δ\\|/SE 2.2). 모델 5개 + Imagenette 1개, 스킴 2개 범위"),
        ("**텐서도 같은가** ([E15](docs/EXPERIMENTS.md#e15)·[E16](docs/EXPERIMENTS.md#e16))", e15_answer,
         "출력 코드 불일치는 test 전체, 국소 불일치는 125/250장 배치의 teacher-forced 값입니다 — 두 수를 같은 문장에서 섞지 마세요"),
        ("**싸구려 반올림의 값** ([E14](docs/EXPERIMENTS.md#e14))", e14_answer,
         "강건한 것은 부호와 유의성이고 **크기는 아닙니다**: SiLU의 truncate는 시드에 따라 −1.36~−2.98%p(시드 SD가 효과의 37%). 한 시드 숫자를 그 모드의 비용으로 인용하면 안 됩니다"),
        ("**그 밖의 정수 구현 세부** ([E7](docs/EXPERIMENTS.md#e7))", e7_answer,
         "바이어스 폭 축은 지수 조정 없는 **클리핑**을 재고(시판 NPU는 더 넓습니다 — Vela는 40비트), 누산기 포화는 부분합이 아니라 최종합에 한 번만 걸리며, 결과는 K ≤ 1,280인 이 모델들에 한정됩니다"),
    ]
    out = ["| 질문 | `results/*.json`이 답하는 것 | 단, 검증되지 않은 것 |", "|---|---|---|"]
    out += [row(q, fn, c) for q, fn, c in rows]
    return "\n".join(out)


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
    out.append("\n온칩 실행 = 가상 프리셋에서 모든 op 지원으로 판정됨(호스트 폴백 없음, 실측 아님). strict = LUT·softmax·layernorm 지원이 없는 프리셋.")
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


def _pm(delta, se):
    return f"{delta*100:+.2f} ± {se*100:.2f}%p"


def e15():
    rs = load("e15_fidelity")["records"]
    if not rs: return ""
    out = ["| 모델 | 데이터셋 | 스킴 | test 장수 | FP32 | fake-quant | 정수 엔진 | 정수 − fake (쌍 SE) | 95% CI | 정답 여부가 갈린 장수 | top-1 일치 | 출력 코드 불일치 (전체) | 첫 분기 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        p = r["int_vs_fake"]; oc = r["output_codes"]
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['dataset']} | {r['scheme']} | {r['n_test']:,} | {pct(r['float_acc'])} | {pct(r['fake_acc'])} | {pct(r['int_acc'])} | "
                   f"{_pm(p['delta'], p['se'])} | [{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}] | {p['n_correctness_disagree']:,} | {pct(p['top1_agreement'], 2)} | "
                   f"{pct(oc['mismatch_frac'], 1)} ({oc['images_with_any_mismatch']:,}장) | {r['agreement_batch'].get('first_divergence')} |")
    batches = sorted({r["agreement_batch"]["images"] for r in rs})
    nb = "/".join(str(b) for b in batches) + "장"
    out.append(f"\n쌍 SE = 같은 이미지에서 잰 (정수 정답 − fake 정답)의 표본 표준편차 / √n. 정확도·출력 코드 열은 test 전체(로짓 코드 = 클래스 × 이미지)에서, "
               f"첫 분기 열과 아래의 국소/전파 표는 {nb} 배치의 전파·teacher-forced 비교에서 나온 값입니다"
               + (f" (CIFAR-10 {max(batches)}장, Imagenette {min(batches)}장)." if len(batches) > 1 else "."))
    out += ["", "| 모델 | 스킴 | conv/linear 국소 불일치 최대 | 국소 불일치가 가장 큰 노드 | 마지막 compute 노드의 전파 불일치 |", "|---|---|---|---|---|"]
    for r in rs:
        rows = [x for x in r["per_layer_agreement"] if x["op"] not in ("input", "output", "flatten", "transpose", "reshape")]
        if not rows: continue
        cl = [x for x in rows if x["op"] in ("conv", "linear")]
        worst = max(rows, key=lambda x: x["local_mismatch_frac"])
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['scheme']} | {pct(max(x['local_mismatch_frac'] for x in cl), 3) if cl else '—'} | "
                   f"{worst['name']} ({worst['op']}, {pct(worst['local_mismatch_frac'], 2)}) | {pct(rows[-1]['mismatch_frac'], 1)} |")
    out += [""] + e15_summary(rs)
    return "\n".join(out)


def e15_summary(rs):
    """The numeric claims of the E15 section, derived from the records."""
    p = [r["int_vs_fake"] for r in rs]
    z = [abs(x["delta"] / x["se"]) for x in p if x["se"]]
    codes = [r["output_codes"]["mismatch_frac"] for r in rs]
    anyimg = [r["output_codes"]["images_with_any_mismatch"] / r["n_test"] for r in rs]
    dis = [x["n_correctness_disagree"] for x in p]
    ci_zero = sum(1 for x in p if x["ci95"][0] <= 0 <= x["ci95"][1])
    conv = [x["local_mismatch_frac"] for r in rs for x in r["per_layer_agreement"] if x["op"] == "conv"]
    lin = [x["local_mismatch_frac"] for r in rs for x in r["per_layer_agreement"] if x["op"] == "linear"]
    first = sorted({r["agreement_batch"].get("first_divergence") for r in rs})
    out = [f"* **정확도 차이**: {len(rs)}행 모두 |정수 − fake| ≤ {max(abs(x['delta']) for x in p) * 100:.2f}%p, |Δ|/SE ≤ {max(z):.1f}, "
           f"95% 신뢰구간이 0을 품는 행 {ci_zero}/{len(p)}. 정답 여부가 갈린 이미지는 {min(dis):,}~{max(dis):,}장.",
           f"* **출력 코드**: test 전체에서 코드의 {min(codes) * 100:.1f}~{max(codes) * 100:.1f}%가 다르고, 코드가 하나라도 다른 이미지는 "
           f"{min(anyimg) * 100:.0f}~{max(anyimg) * 100:.0f}%.",
           f"* **국소 원천**: conv {min(conv) * 100:.2f}~{max(conv) * 100:.2f}%, linear {min(lin) * 100:.2f}~{max(lin) * 100:.2f}%, "
           f"첫 분기 노드는 {', '.join(x for x in first if x)}."]
    return out


def e16():
    rs = load("e16_ln_emulation")["records"]
    if not rs: return ""
    out = ["| 모델 | 스킴 | fake-quant의 LayerNorm | fake-quant | 정수 엔진 | 정수 − fake (쌍 SE) | top-1 일치 | 출력 코드 불일치 (test 전체) | 로짓 평균 |차| | LN 국소 불일치 | softmax 국소 | matmul 국소 | linear 국소 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        p = r["int_vs_fake"]; po = r["per_op"]
        def loc(op):
            return pct(po[op]["local_mean"], 2) if op in po else "—"
        ln = "float32 (기존)" if r["variant"] == "float-ln" else f"정수 에뮬레이션 ({r['layernorms_emulated']}개)"
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {r['scheme']} | {ln} | {pct(r['fake_acc'])} | {pct(r['int_acc'])} | "
                   f"{_pm(p['delta'], p['se'])} | {pct(p['top1_agreement'], 2)} | {pct(r['output_codes']['mismatch_frac'], 1)} | {r['logit_mean_abs_diff']:.3f} | "
                   f"{loc('layernorm')} | {loc('softmax')} | {loc('matmul')} | {loc('linear')} |")
    out.append(f"\ntest {rs[0]['n_test']:,}장 전체. 국소 불일치는 {rs[0]['agreement_batch']['images']}장 배치에서 연산 종류별 평균(teacher-forced). 정수 엔진 쪽은 두 행에서 같은 정수 프로그램이다(export가 바뀌지 않음을 실험이 assert).")
    return "\n".join(out)


def e17():
    """Two depths at one width profile: the depth column is what separates depth from task."""
    rs = load("e17_dense_output")["records"]
    if not rs: return ""
    out = ["| conv 층수 | 스킴 | float32 | fake-quant | 정수 엔진 | 정수 − fake (쌍 SE) | 양자화 비용 (fake − float) | 출력 픽셀 코드 불일치 | 코드가 다른 이미지 | 최대 \\|Δ코드\\| | conv 국소 불일치 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rs, key=lambda r: (r.get("conv_layers", 0), r["scheme"])):
        p_, oc = r["int_vs_fake"], r["output_codes"]
        loc = [row["local_mismatch_frac"] * 100 for row in r["per_layer_agreement"] if row["op"] == "conv"]
        out.append(f"| {r.get('conv_layers', '—')} | {r['scheme']} | {r['float_psnr']:.3f} dB | {r['fake_psnr']:.3f} dB | {r['int_psnr']:.3f} dB | "
                   f"{p_['delta']:+.5f} ± {p_['se']:.5f} dB | {r['fake_vs_float']['delta']:+.3f} dB | "
                   f"{pct(oc['mismatch_frac'], 1)} | {oc['images_with_any_mismatch']:,} / {oc['images']:,} | "
                   f"{oc['max_abs_code_diff']} | {min(loc):.2f}–{max(loc):.2f}% |")
    r0 = rs[0]
    out.append(f"\ntest {r0['n_test']:,}장 전체, 이미지당 출력값 {r0['output_codes']['values_per_image']:,}개(3×128×128 픽셀). "
               f"PSNR은 [0,1] 범위에서 이미지별로 재고 배포와 똑같이 출력을 clip한 뒤 평균한 값이다. "
               f"국소 불일치는 {r0['agreement_batch']['images']}장 배치의 teacher-forced 평균. "
               f"두 깊이는 같은 폭 구성이며, 깊은 쪽이 더 좋은 모델은 아니다(float PSNR이 0.12 dB 낮다).")
    return "\n".join(out)


def e14():
    d = load("e14_rounding_seeds"); rs = d["records"]
    if not rs: return ""
    models = list(dict.fromkeys(r["model"] for r in rs)); rounds = d["meta"].get("roundings") or list(dict.fromkeys(r["rounding"] for r in rs))
    out = ["| 반올림 (2단계) | " + " | ".join(f"{LABEL.get(m, m)}: 기준 대비 Δacc, 시드 평균 ± 시드 표준편차 (시드별)" for m in models) + " |", "|---|" + "---|" * len(models)]
    for rd in rounds:
        if rd == "tflite": continue
        cells = []
        for m in models:
            xs = sorted([r for r in rs if r["model"] == m and r["rounding"] == rd], key=lambda r: r["seed"])
            if not xs: cells.append("—"); continue
            ds_ = np.array([r["vs_reference"]["delta"] for r in xs]) * 100
            sd = f"{ds_.std(ddof=1):.2f}" if len(ds_) > 1 else "—"
            cells.append(f"{ds_.mean():+.2f} ± {sd}%p ({', '.join(f'{v:+.2f}' for v in ds_)}; n={len(ds_)})")
        out.append(f"| `{rd}` | " + " | ".join(cells) + " |")
    out += ["", "| 모델 | 시드 | FP32 | fake-quant | 기준 정수 (`tflite`) | `single` | `half_even` | `truncate` | `floor` |", "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        for seed in sorted({r["seed"] for r in rs if r["model"] == m}):
            xs = {r["rounding"]: r for r in rs if r["model"] == m and r["seed"] == seed}
            ref = xs.get("tflite")
            if not ref: continue
            def cell(rd):
                r = xs.get(rd)
                return f"{pct(r['int_acc'])} ({_pm(r['vs_reference']['delta'], r['vs_reference']['se'])}, 일치 {pct(r['vs_reference']['top1_agreement'], 1)})" if r else "—"
            out.append(f"| {LABEL.get(m, m)} | {seed} | {pct(ref['float_acc'])} | {pct(ref['fake_acc'])} | {pct(ref['int_acc'])} | {cell('single')} | {cell('half_even')} | {cell('truncate')} | {cell('floor')} |")
    out.append(f"\ntest {d['meta'].get('n_test', 10000):,}장 전체, 스킴 npu-default, 캘리브레이션 512장(seed 0)은 시드마다 그 시드의 체크포인트로 다시 수집. Δacc의 쌍 SE는 같은 이미지에서 기준 구현과 비교한 값.")
    out += [""] + e14_summary(rs, models)
    return "\n".join(out)


def e14_summary(rs, models):
    """The numeric claims of the E14 section, derived from the records so they cannot go stale."""
    LOSSY = ("truncate", "floor")
    lossy = [r for r in rs if r["rounding"] in LOSSY]
    out = []
    if lossy:
        z = [abs(r["vs_reference"]["delta"] / r["vs_reference"]["se"]) for r in lossy if r["vs_reference"]["se"]]
        dl = [r["vs_reference"]["delta"] * 100 for r in lossy]
        n_worse = sum(1 for v in dl if v < 0)
        out.append(f"* **부호와 유의성**: truncate·floor는 측정한 {len(lossy)}개 (모델, 시드, 모드) 조합 중 {n_worse}개에서 기준보다 나쁩니다 — "
                   f"{min(dl):+.2f}~{max(dl):+.2f}%p, |Δ|/SE {min(z):.1f}~{max(z):.1f}.")
    spread = []
    for m in models:
        parts = []
        for rd in LOSSY:
            xs = sorted([r for r in rs if r["model"] == m and r["rounding"] == rd], key=lambda r: r["seed"])
            if len(xs) < 2:
                continue
            v = np.array([r["vs_reference"]["delta"] for r in xs]) * 100
            parts.append(f"{rd} {v.mean():+.2f} ± {v.std(ddof=1):.2f}%p (시드별 폭 {v.max() - v.min():.2f}%p, n={len(v)})")
        if parts:
            spread.append(f"{LABEL.get(m, m)} — " + ", ".join(parts))
    if spread:
        out.append("* **크기의 시드 분산**: " + " / ".join(spread) + ".")
    pairs = [(m, sd) for m in models for sd in sorted({r["seed"] for r in rs if r["model"] == m})]
    cmp_rows = []
    for m, sd in pairs:
        f = next((r for r in rs if r["model"] == m and r["seed"] == sd and r["rounding"] == "floor"), None)
        t = next((r for r in rs if r["model"] == m and r["seed"] == sd and r["rounding"] == "truncate"), None)
        if f and t:
            cmp_rows.append((m, sd, f["vs_reference"]["delta"] - t["vs_reference"]["delta"], f["vs_reference"]["se"] + t["vs_reference"]["se"]))
    if cmp_rows:
        n_floor_worse = sum(1 for _, _, diff, _ in cmp_rows if diff < 0)
        n_sep = sum(1 for _, _, diff, se in cmp_rows if abs(diff) > se)
        out.append(f"* **floor vs truncate**: {len(cmp_rows)}개 (모델, 시드) 중 floor가 더 나쁜 경우 {n_floor_worse}개, "
                   f"두 모드의 차이가 각자의 쌍 SE 합보다 큰 경우 {n_sep}개.")
    lossless = [r for r in rs if r["rounding"] in ("single", "half_even") and r["vs_reference"]["se"]]
    if lossless:
        zz = [abs(r["vs_reference"]["delta"] / r["vs_reference"]["se"]) for r in lossless]
        dd = [abs(r["vs_reference"]["delta"]) * 100 for r in lossless]
        out.append(f"* **single / half_even**: {len(lossless)}개 조합 모두 |Δ| ≤ {max(dd):.2f}%p, |Δ|/SE ≤ {max(zz):.1f}, 기준과 이미지 단위 일치 "
                   f"{min(r['vs_reference']['top1_agreement'] for r in lossless) * 100:.1f}~{max(r['vs_reference']['top1_agreement'] for r in lossless) * 100:.1f}%.")
    return out


TABLES = [("HEADLINE", headline), ("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5), ("E6", e6), ("E7", e7), ("E8", e8), ("E9", e9), ("E10", e10), ("E11", e11), ("E12", e12), ("E14", e14), ("E15", e15), ("E16", e16), ("E17", e17)]


def inject(path: str) -> tuple[int, set]:
    """Replace the text between <!-- TABLE:Ex --> and <!-- /TABLE:Ex --> markers with fresh tables.

    Returns (how many were replaced, which marker names this file carried). The caller uses the names to check
    that no table lost its home — a marker that exists in no file would silently freeze on the day it moved.
    """
    import re
    src = open(path, encoding="utf-8").read()
    n, names = 0, set()
    for name, fn in TABLES:
        t = fn()
        pat = re.compile(rf"(<!-- TABLE:{name} -->)(.*?)(<!-- /TABLE:{name} -->)", re.S)
        if pat.search(src):
            src = pat.sub(lambda m: f"{m.group(1)}\n{t if t else '_(아직 실행되지 않음)_'}\n{m.group(3)}", src)
            n += 1; names.add(name)
    open(path, "w", encoding="utf-8").write(src)
    return n, names


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--inject":
        seen = set()
        for path in sys.argv[2:]:
            n, names = inject(path); seen |= names
            print(f"injected {n} tables into {path}")
        homeless = [name for name, fn in TABLES if name not in seen and fn()]
        if homeless:
            sys.exit(f"no marker found for: {', '.join(homeless)} — a generated table lost its home and would freeze")
    else:
        for name, fn in TABLES:
            t = fn()
            if t:
                print(f"\n<!-- {name} -->\n{t}\n")
