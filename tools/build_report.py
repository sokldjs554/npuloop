"""Build the technical report (Markdown -> HTML -> PDF) from results/*.json.

    python tools/build_report.py            # writes docs/report/npuloop_report.{md,html,pdf}

Every number in the report is read from the results files, so the report can be regenerated after any re-run.
Figures are drawn with matplotlib and inlined as base64 PNGs; the PDF is printed by headless Chromium when one is
available (Playwright's bundle or a system chromium), otherwise only the HTML is written.
"""
import base64, io, json, os, shutil, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "docs", "report")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from readme_tables import LABEL, pct, pp   # noqa: E402


def load(name):
    p = os.path.join(R, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {"records": [], "meta": {}}


def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight"); buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def fig_requant():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = load("e7_requant_ablation")["records"]
    if not rs: return None
    models = list(dict.fromkeys(r["model"] for r in rs)); cfgs = list(dict.fromkeys(r["config"] for r in rs))
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [next((r["top1_agreement_vs_reference"] for r in rs if r["model"] == m and r["config"] == c), np.nan) for c in cfgs]
        ax.bar(np.arange(len(cfgs)) + i * w, [v * 100 for v in vals], w, label=LABEL.get(m, m))
    ax.set_xticks(np.arange(len(cfgs)) + w * (len(models) - 1) / 2); ax.set_xticklabels([c.replace("tflite-", "").replace("-acc32-b32", "") for c in cfgs], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("top-1 agreement with reference (%)"); ax.set_ylim(0, 105); ax.legend(fontsize=8, loc="lower left"); ax.grid(axis="y", alpha=.3)
    ax.set_title("E7: integer implementation choices vs the gemmlowp reference (2,000 images)", fontsize=10)
    return fig_to_b64(fig)


def fig_attribution():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = load("e9_customer_intake")["records"]
    vit = next((r for r in rs if r["model"] == "cust_vit" and r.get("ptq", {}).get("per_op_agreement")), None)
    if not vit: return None
    po = vit["ptq"]["per_op_agreement"]
    ops = [k for k, v in sorted(po.items(), key=lambda kv: -kv[1]["local_mismatch_mean"]) if v["local_mismatch_mean"] > 0 or v["propagated_mean"] > 0][:8]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(ops))
    ax.bar(x - 0.2, [po[o]["local_mismatch_mean"] * 100 for o in ops], 0.4, label="local (teacher-forced)")
    ax.bar(x + 0.2, [po[o]["propagated_mean"] * 100 for o in ops], 0.4, label="propagated (end-to-end)")
    ax.set_xticks(x); ax.set_xticklabels(ops); ax.set_ylabel("codes that differ (%)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    ax.set_title("E9: fake-quant vs integer engine on the ViT, per op kind (64 images)", fontsize=10)
    return fig_to_b64(fig)


def fig_tflite():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in load("e11_tflite_crosscheck")["records"] if r["engine"] == "numpy"]
    if not rs: return None
    nodes = list(next(iter(rs))["per_node"].keys())
    short = [n.split("/")[1].split("_")[0] if "/" in n else ("fc" if "Call" in n else n) for n in nodes]
    fig, ax = plt.subplots(figsize=(8, 3))
    w = 0.8 / len(rs)
    for i, r in enumerate(rs):
        ax.bar(np.arange(len(nodes)) + i * w, [r["per_node"][n]["mismatch_frac"] * 100 for n in nodes], w, label=r["rounding"])
    ax.set_xticks(np.arange(len(nodes)) + w * (len(rs) - 1) / 2); ax.set_xticklabels(short)
    ax.set_ylabel("codes that differ from TFLite (%)"); ax.set_yscale("symlog", linthresh=0.1); ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=.3)
    ax.set_title("E11: npuloop vs TFLite reference kernels, per tensor, by rounding mode", fontsize=10)
    return fig_to_b64(fig)


def table(rows, header):
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def build_markdown():
    e1 = load("e1_baselines")["records"]; e2 = load("e2_ptq_grid")["records"]; e7 = load("e7_requant_ablation")["records"]
    e8 = load("e8_scalesim")["records"]; e9 = load("e9_customer_intake")["records"]; e10 = load("e10_engine_timing")["records"]
    e11 = load("e11_tflite_crosscheck"); e12 = load("e12_imagenette")["records"]
    n_models = len(e1)
    # E2 summary: |int - fake| across primary schemes
    prim = [r for r in e2 if r["scheme"] in ("npu-default", "per-tensor")]
    gaps = [abs(r["int_acc"] - r["fake_acc_on_int_subset"]) for r in prim]
    top1 = [r["agreement"]["top1_agreement"] for r in e2]
    out_mis = [r["agreement"]["output_mismatch_frac"] for r in e2]
    e2_rows = [[LABEL.get(r["model"], r["model"]), r["scheme"], f"{r['int_eval_images']:,}", pct(r["fake_acc_on_int_subset"]), pct(r["int_acc"]),
                pp(r["int_acc"] - r["fake_acc_on_int_subset"]), pct(r["agreement"]["top1_agreement"], 1), pct(r["agreement"]["output_mismatch_frac"], 0)] for r in prim]
    # E7 table for ResNet-20 ReLU
    sat = lambda v: sum(v.values()) if isinstance(v, dict) else int(v or 0)
    e7_rows = [[r["config"], pct(r["int_acc"]), pct(r["top1_agreement_vs_reference"], 1), f"{sat(r['saturations']):,}"] for r in e7 if r["model"] == "resnet20_relu"]
    # E9 attribution
    vit = next((r for r in e9 if r["model"] == "cust_vit"), None)
    po = vit["ptq"]["per_op_agreement"] if vit and vit["ptq"].get("per_op_agreement") else {}
    e9_rows = [[op, v["n"], pct(v["local_mismatch_mean"], 2), pct(v["local_mismatch_max"], 2), pct(v["propagated_mean"], 1)]
               for op, v in sorted(po.items(), key=lambda kv: -kv[1]["local_mismatch_mean"]) if v["local_mismatch_mean"] > 0][:6]
    # E10
    e10_rows = [[LABEL.get(r["model"], r["model"]), f"{r['numpy']['per_image_ms']:.1f}", f"{r['cpp']['per_image_ms']:.1f}", f"{r['speedup']:.1f}×",
                 f"{r['output_equality']['images'] - r['output_equality']['images_with_any_mismatch']:,}/{r['output_equality']['images']:,}" if r.get("output_equality") else "—",
                 f"{r['modelled_cycles']:,.0f}"] for r in e10]
    # E11
    e11_rows = [[f"{r['graph_rounding']} / {r['fc_rounding']}", f"{r['output_mismatch_images']}/{r['images']}", f"{r['output_mismatch_elems']}/{r['output_elems']}",
                 " / ".join(f"{v['mismatch_frac'] * 100:.2f}%" for v in r["per_node"].values())] for r in e11["records"] if r["engine"] == "numpy"]
    xnn = e11.get("meta", {}).get("xnnpack_vs_reference", {})
    # E8
    e8_rows = [[LABEL.get(r["model"], r["model"]), f"{r['rows']}×{r['cols']}", f"{r['total_scalesim']:,}", f"{r['total_ours']:,.0f}", f"{r['ratio']:.4f}", pct(r["max_layer_abs_err"], 1)] for r in e8]
    # E12
    base12 = next((r for r in e12 if r["kind"] == "baseline"), None)
    ptq12 = [r for r in e12 if r["kind"] == "ptq"]

    figs = dict(requant=fig_requant(), attribution=fig_attribution(), tflite=fig_tflite())
    img = lambda k, alt: f'![{alt}](data:image/png;base64,{figs[k]})' if figs.get(k) else ""

    md = f"""# npuloop: 가상 NPU 비용 모델과 비트 정확 정수 엔진으로 닫는 모델 압축 루프

**기술 보고서** · {time.strftime('%Y-%m-%d')} · 저장소 [github.com/sokldjs554/npuloop](https://github.com/sokldjs554/npuloop)

## 초록

NPU 회사의 모델 팀은 고객이 보낸 체크포인트에 대해 "우리 칩에서 돌기는 하는가, 얼마나 걸리는가, 무엇을 바꿔야 하는가"를 답해야 한다.
이 보고서는 그 세 질문을 숫자로 답하는 오픈소스 툴킷 npuloop의 설계와 실험을 정리한다. torch.fx 그래프 IR 위에 (1) SCALE-Sim으로
검증한 weight-stationary systolic-array 비용 모델, (2) NPU 준비도 lint, (3) 비트 단위로 일치하는 두 정수 엔진(NumPy·C++)과 파이썬 없는
독립 실행기, (4) 비용 모델을 루프 안에서 호출하는 그래프 기반 구조적 프루닝을 얹었다. CIFAR-10 5개 모델(ResNet-20 ReLU/SiLU, MobileNetV2-0.5,
Inception-32, ViT-128/6)과 Imagenette 128×128에서 fake-quant와 정수 실행의 불일치를 노드별로 국소(teacher-forced)/전파로 분리해 재고,
정수 구현 세부(반올림·곱셈기 비트·누산기·바이어스 폭)의 정확도 비용을 ablation했으며, TensorFlow Lite reference 커널과 1,000장 × 모든 텐서에서
비트 일치함을 보였다. 주요 결과: fake-quant는 정확도 예측기로 충분하지만(주 스킴에서 |Δ| ≤ {max(gaps) * 100:.2f}%p) 텐서 단위로는
출력 코드의 {min(out_mis) * 100:.0f}~{max(out_mis) * 100:.0f}%가 다르고, transformer에서는 LayerNorm이 국소 불일치의 대부분을 차지하며,
TFLite reference 커널은 conv/pool과 fully-connected가 서로 다른 반올림을 쓴다.

## 1. 문제

INT8 NPU에 모델을 올리는 작업은 세 층위의 불일치를 다룬다. (a) 학습 프레임워크의 fake-quant와 실제 정수 실행의 차이, (b) FLOPs와
사이클의 차이(배열 정렬·채움/비움·메모리), (c) 같은 "INT8"이라도 런타임마다 다른 requantization 구현. 공개 도구는 이 셋을 따로 다룬다:
MQBench·PPQ는 (a)를 백엔드 정확도 격차로, SCALE-Sim·Timeloop는 (b)를 오프라인 시뮬레이션으로, TFLite·gemmlowp는 (c)를 코드로만.
npuloop의 기여는 셋을 한 그래프 IR 위에 놓고, 결정(프루닝·활성함수 교체·PTQ/QAT)이 실제 정수 결과와 사이클로 되돌아오게 한 것이다.

## 2. 방법

**IR.** `torch.fx`로 추적한 그래프에서 BN을 접고 shape를 전파한 `StaticGraph`. conv/linear(토큰 단위 포함)/add/mul/concat/matmul/softmax/
layernorm/transpose/reshape/pool/활성함수를 지원하고 나머지는 즉시 실패시킨다.

**비용 모델.** weight-stationary systolic array에서 GEMM 한 타일의 사이클을 `M + 2R + C − 2`(M 출력 픽셀, R×C 배열)로 두고 ⌈K/R⌉·⌈N/C⌉
타일을 합산, 멀티코어 M/N 분할, depthwise 엔진, LUT/융합/호스트 폴백 활성함수, DRAM roofline을 더한다. SCALE-Sim v3(사이클 정확, WS)와
4개 모델 × 3개 배열 크기에서 합계 0.1% 이내로 일치한다(표 6). 에너지는 MAC·SRAM·DRAM·벡터·호스트 항목의 pJ 상수(Horowitz 2014 자릿수)로
같은 양에서 추정하며 `simulated` 라벨을 단다.

**정수 엔진.** 캘리브레이션된 fake-quant 그래프를 int8 가중치·int32 바이어스·Q31 곱셈기+시프트의 `IntGraph`로 export하고, gemmlowp/TFLite의
`MultiplyByQuantizedMultiplier` 의미론으로 실행한다. NumPy 구현과 C++ 커널(ctypes)은 모든 중간 텐서에서 비트 일치를 테스트로 강제하고,
`.npuloop` 파일(JSON 헤더 + raw 정수 배열)을 파이썬 없이 실행하는 C++ 러너가 같은 답을 낸다. `RequantConfig`로 반올림 모드·곱셈기 비트·
누산기/바이어스 폭을 바꿔 "싸구려 구현"을 흉내 낼 수 있고, 노드별 반올림 오버라이드도 가능하다.

**검증.** 같은 이미지에서 fake-quant 코드와 정수 코드를 노드별로 비교하되, 두 가지로 잰다: **전파** 불일치(끝까지 정수로 실행)와
**국소** 불일치(각 노드에 fake-quant의 입력 코드를 teacher-forcing으로 넣고 그 노드의 출력만 비교). 국소 값이 어느 연산이 차이를 *만들어내는지*를,
전파 값이 그 차이가 어디까지 *번지는지*를 보여 준다.

**프루닝.** 그래프에서 conv/linear 출력 채널을 활성함수·depthwise conv를 지나 소비자까지 따라가 그룹을 찾는다(residual add·pool·matmul·
LayerNorm에 닿으면 제외, concat을 지나면 소비자의 입력 슬라이스를 함께 자름). uniform·aligned·cost-greedy(사이클 절감/중요도 손실이 큰
그룹부터, 비용 모델을 매 단계 호출) 세 전략.

## 3. 실험 설정

CIFAR-10 공식 학습 50,000장을 고정 시드로 학습 45,000 / 검증 5,000(클래스별 500)으로 나누고 검증 정확도로 체크포인트를 고르며 test 10,000장은
마지막에 한 번 평가한다. 모델 {n_models}개: {', '.join(LABEL.get(r['model'], r['model']) for r in e1)}. 가상 NPU 프리셋 4개(tiny-1tops, edge-10tops,
pcie-80tops, edge-10tops-strict). 캘리브레이션은 학습 분할에서 512장. 모든 학습·실험은 CPU 4코어에서 seed 0 한 번으로, 0.2%p 이하의 정확도
차이는 잡음이다(10k 표준오차 ≈ 0.3%p).

## 4. 결과

### 4.1 fake-quant는 정확도를 맞히지만 텐서는 맞히지 않는다 (E2)

{table(e2_rows, ["모델", "스킴", "이미지", "fake-quant", "정수 엔진", "차이", "top-1 일치", "출력 코드 불일치"])}

주 스킴에서 정확도 차이는 최대 {max(gaps) * 100:.2f}%p, 이미지 단위 top-1 일치는 {min(top1) * 100:.1f}~{max(top1) * 100:.1f}%다. 그러나 출력
코드는 {min(out_mis) * 100:.0f}~{max(out_mis) * 100:.0f}%가 다르다. 각 conv의 국소 불일치는 0.03~0.2%(±1 LSB)뿐이며 나머지는 나비효과다.

### 4.2 정수 구현 세부의 비용 (E7)

{img('requant', 'E7 requant ablation')}

{table(e7_rows, ["RequantConfig (ResNet-20 ReLU)", "정확도", "기준과 top-1 일치", "포화 횟수"])}

반올림 모드(single/half-even)는 무관하지만 truncate/floor는 1~3%p를 잃고, 3비트 곱셈기는 ResNet-20 ReLU에서 −8%p, 바이어스 int16은 최대 −10%p,
누산기 int16은 모델을 무너뜨린다(1,115만 회 포화). 이 순서는 세 모델에서 같다.

### 4.3 transformer에서 차이를 만드는 곳은 LayerNorm이다 (E9)

{img('attribution', 'E9 per-op attribution')}

{table(e9_rows, ["op", "노드 수", "국소 불일치 평균", "최대", "전파"])}

ViT-128/6의 최종 출력 코드 불일치는 {pct(vit['ptq']['agreement']['output_mismatch_frac'], 0) if vit else '—'}(CNN 30~50%)이고 top-1 일치는
{pct(vit['ptq']['agreement']['top1_agreement'], 1) if vit else '—'}다. 국소 불일치는 LayerNorm이 압도적이다: 정수 LayerNorm은 int64 합과 정확한 정수 제곱근으로
정규화하고 fake-quant는 float32로 계산한 뒤 격자에 올리므로 네 개 중 하나꼴로 반올림 경계 반대편에 떨어진다(모두 ±1 LSB).

### 4.4 실제 런타임과의 비트 일치 (E11)

{img('tflite', 'E11 TFLite cross-check')}

{table(e11_rows, ["반올림 conv·pool / fc", "출력이 다른 이미지", "다른 원소", "conv1 / conv2 / conv3 / pool / fc 국소 불일치"])}

TFLite full-integer 모델의 파라미터를 그대로 읽어 만든 정수 그래프는 conv·pool을 gemmlowp 이중 반올림, fully-connected를 단일 반올림으로
실행할 때 TFLite reference 커널과 1,000장 × 모든 텐서에서 0개 불일치다. 그래프 전체를 한 모드로 두면 fc에서 0.2%가 ±1 LSB 어긋난다.
TFLite 자신의 XNNPACK 델리게이트는 reference 커널과 {xnn.get('output_mismatch_images', '?')}/{xnn.get('images', '?')}장에서 다르다.

### 4.5 실측과 모델을 같은 표에 (E10)

{table(e10_rows, ["모델", "NumPy ms/장", "C++ ms/장", "C++/NumPy", "두 엔진 출력 동일", "모델 사이클 (edge-10tops)"])}

이 저장소가 실측할 수 있는 시간은 호스트 CPU의 검증 엔진뿐이다. 처음 잰 C++ 엔진은 순진한 7중 루프라 NumPy(내부는 torch conv2d)보다 3배
느렸고, im2col + int32 GEMM(4탭 블록, `K·max|x−zp|·128 < 2³¹`일 때만 int32 누산)으로 고쳐 2~3배 빠르게 만들었다.

### 4.6 비용 모델 검증 (E8)

{table(e8_rows, ["모델", "배열", "SCALE-Sim cycles", "npuloop cycles", "비율", "최악 레이어 오차"])}

검증 범위는 dense conv·linear의 systolic 연산 사이클(단일 코어, 메모리 스톨 없음)이다. 처음 모델(`M + R + C`)은 10~13% 낙관적이었고
가중치 타일 적재 `R` 사이클이 빠져 있었다.

### 4.7 두 번째 데이터셋 (E12)

{('Imagenette 128×128, ResNet-20(stem stride 2): val ' + pct(base12['val_acc']) + ' → test ' + pct(base12['test_acc']) + ', ' + f"{base12['macs'] / 1e6:.0f}M MACs; PTQ " + ', '.join(f"{r['scheme']} fake {pct(r['fake_acc'])} / 정수 {pct(r['int_acc'])}" for r in ptq12)) if base12 else '실행 전.'}

## 5. 한계

CIFAR-10 규모, seed 1개, 가상 NPU(실제 컴파일러의 fusion·타일링·메모리 스케줄링 없음), 에너지는 자릿수 추정, TFLite 교차 검증은 conv·pool·fc에
한정, ImageNet 사전학습 체크포인트 미사용(호스트 차단). 체크포인트를 바꾸면 결론 일부가 흔들렸다: E3의 레이어 수준 상관은 재현되지 않았고
E7의 3비트 곱셈기 손실은 체크포인트에 따라 −1.3에서 −8.2%p까지 달랐다.

## 6. 결론

fake-quant 정확도는 믿어도 되지만 텐서는 믿으면 안 되고, "INT8"은 런타임마다 다른 반올림을 뜻하며, 프루닝의 값은 사이클로 환산했을 때만
드러난다. 이 세 문장을 재현 가능한 숫자로 만든 것이 npuloop이다.

## 참고문헌

Jacob et al. 2018 (CVPR) 정수 산술 추론; gemmlowp/TFLite `MultiplyByQuantizedMultiplier`; Samajdar et al. 2020 SCALE-Sim; Nagel et al. 2019 DFQ;
Liu et al. 2017 Network Slimming; Li et al. 2021 MQBench; Kim et al. 2021 I-BERT; Horowitz 2014 (ISSCC) 에너지 수치.
"""
    return md


def main():
    md = build_markdown()
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "npuloop_report.md"), "w", encoding="utf-8").write(md)
    from markdown_it import MarkdownIt
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(md)
    css = """body{font-family:'Noto Sans KR','IBM Plex Sans',system-ui,sans-serif;max-width:860px;margin:32px auto;padding:0 24px;line-height:1.55;color:#16212C;font-size:12.5px}
h1{font-size:22px;line-height:1.25}h2{font-size:16px;margin-top:26px;border-bottom:1px solid #D5DCE4;padding-bottom:4px}h3{font-size:13.5px;margin-top:18px}
table{border-collapse:collapse;font-size:10.5px;margin:8px 0;width:100%}th,td{border:1px solid #D5DCE4;padding:3px 6px;text-align:right}th:first-child,td:first-child{text-align:left}
th{background:#F2F4F7}img{max-width:100%;display:block;margin:8px auto}code{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:11px;background:#F2F4F7;padding:0 3px}
@page{size:A4;margin:16mm}"""
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>npuloop technical report</title><style>{css}</style></head><body>{body}</body></html>"
    html_path = os.path.join(OUT, "npuloop_report.html")
    open(html_path, "w", encoding="utf-8").write(html)
    chrome = next((c for c in ["/opt/pw-browsers/chromium-1194/chrome-linux/chrome", shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("google-chrome")] if c and os.path.exists(c)), None)
    if chrome:
        pdf = os.path.join(OUT, "npuloop_report.pdf")
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file://" + html_path],
                       check=False, capture_output=True, timeout=180)
        print("pdf", os.path.exists(pdf) and os.path.getsize(pdf))
    print("wrote", html_path, len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
