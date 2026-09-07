# npuloop — NPU 비용 모델과 비트 정확 정수 엔진으로 닫는 모델 압축 루프

> **"INT8 NPU에 올릴 모델의 양자화·프루닝·활성함수 선택을, FLOPs와 fake-quant가 아니라
> 가상 NPU 비용 모델과 비트 정확(bit-exact) 정수 추론 엔진에 대고 검증한다."**
>
> 해석적 systolic-array 비용 모델(`npu`) → 정적 준비도 lint(`lint`) → NPU 방식 양자화·QAT·CLE·활성함수 교체(`quant`)
> → PE-array 정렬 구조적 프루닝(`prune`) → int8 가중치/int32 바이어스/고정소수점 requant로 export한 정수 그래프를
> NumPy와 C++ 커널로 **비트 단위로 같게** 실행하고 fake-quant와 비교(`intengine`)하는 한 저장소입니다.
> CIFAR-10에서 ResNet-20 계열 4종과 MobileNetV2로 7개 실험(E1–E7)을 돌려 결과를 JSON으로 남겼고, 그 JSON을 읽는
> [인터랙티브 데모](#데모)가 있습니다.

<!-- RESULTS_SUMMARY -->

## 왜 이 프로젝트를 했나

이전 프로젝트들([ondevice-pcb-inspection](https://github.com/sokldjs554/ondevice-pcb-inspection), [edgesight-soc](https://github.com/sokldjs554/edgesight-soc))에서
PTQ INT8 배포와 RTL NPU를 각각 만들어 보고 나니, 두 세계 사이에 구멍이 하나 보였습니다.

* **모델 쪽**은 `torch.round`로 흉내 낸 fake-quant 정확도와 FLOPs로 결정을 내리고,
* **하드웨어 쪽**은 int32 누산기·고정소수점 requant·배열 타일링으로 실제 사이클과 실제 정확도를 만듭니다.

둘 사이의 차이를 개인 프로젝트 수준에서 정량화한 저장소는 찾지 못했습니다(조사 내용은 [docs/RELATED.md](docs/RELATED.md)).
"fake-quant가 90%라고 하면 NPU에서도 90%인가?", "FLOPs를 반으로 줄이면 사이클도 반이 되는가?",
"SiLU를 ReLU로 바꾸면 무엇을 얻고 잃는가?"에 **숫자로** 답하는 것이 목표였습니다.

## 폐루프 구조

```mermaid
flowchart LR
    A[PyTorch 모델<br>ResNet-20 · MobileNetV2] --> B[graph.trace<br>fx IR · BN folding]
    B --> C[npu.estimate<br>cycles · util · roofline]
    B --> D[lint<br>준비도 점수·findings]
    C --> E[prune / surgery<br>array-aligned · act swap]
    D --> E
    E --> F[quant.prepare<br>PTQ · QAT · CLE · BC]
    F --> G[intengine.export<br>int8 W · int32 b · M0,shift]
    G --> H[NumPy ⇄ C++ 정수 엔진<br>bit-exact]
    H --> I[verify<br>fake-quant vs integer<br>국소 vs 전파]
    I -. 결과가 다시 결정으로 .-> E
```


## 실험 결과 (CIFAR-10, seed 0, CPU 4코어)

숫자는 전부 `results/*.json`에서 `tools/readme_tables.py --inject README.md`로 생성한 것입니다. 정확도는 test 10,000장 기준이며,
`simulated`로 표시한 사이클·활용률은 가상 NPU 비용 모델 값입니다.

### E1. 베이스라인과 정적 분석

같은 학습 레시피(30 epoch OneCycle SGD)로 훈련한 5개 모델을 4개 가상 NPU 프리셋에 올렸을 때의 사이클과 lint 점수입니다.
ResNet-20의 16/32 채널은 64폭 배열의 열을 1/4~1/2밖에 채우지 못해 배열 활용률이 20% 아래로 떨어지고, 32×32 배열(tiny-1tops)에서는 50%대로 올라갑니다.
"큰 NPU가 항상 유리하지 않다"는 것이 첫 번째 관찰입니다.

<!-- TABLE:E1 -->
| 모델 | 파라미터 | MACs | FP32 acc | tiny-1tops cycles (util) | edge-10tops cycles (util) | pcie-80tops cycles (util) | lint eff / q-rob |
|---|---|---|---|---|---|---|---|
| ResNet-20 ReLU | 272,474 | 40.8M | 90.47% | 87,730 (45%) | 34,417 (14%) | 22,791 (5%) | 80 / 94 |
<!-- /TABLE:E1 -->

### E8. 비용 모델은 믿을 만한가 — SCALE-Sim 대조

해석적 모델의 연산 사이클을 SCALE-Sim v3(사이클 정확 systolic 시뮬레이터, WS dataflow, 대역폭 제한 없음)와 레이어별로 비교했습니다.
처음 만든 모델(타일당 `M + R + C`)은 10~13% 낙관적이었고, SCALE-Sim의 per-fold 사이클을 뜯어보니 가중치 타일을 배열에 싣는 `R` 사이클이 빠져 있었습니다.
`M + 2R + C − 2`로 고친 뒤에는 세 배열 크기 모두에서 합계 0.03%, 최악 레이어 0.5%(FC의 off-by-one) 이내입니다.

<!-- TABLE:E8 -->
| 모델 | 배열 | SCALE-Sim cycles | npuloop cycles | 비율 | 최악 레이어 오차 | fill/drain 없이 |
|---|---|---|---|---|---|---|
| ResNet-20 ReLU | 32×32 | 84,276 | 84,298 | 1.0003 | 0.5% | 0.683 |
| ResNet-20 ReLU | 64×64 | 49,123 | 49,145 | 1.0004 | 0.5% | 0.614 |
| ResNet-20 ReLU | 16×16 | 208,486 | 208,508 | 1.0001 | 0.5% | 0.766 |
<!-- /TABLE:E8 -->

### E2. PTQ 스킴 그리드 — fake-quant는 정수 엔진을 얼마나 잘 예측하는가

같은 512장 캘리브레이션으로 7개 스킴을 적용한 뒤 (a) PyTorch fake-quant 정확도, (b) 같은 파라미터를 int8 가중치·int32 바이어스·고정소수점 requant로 export해 **비트 정확 정수 엔진**으로 돌린 정확도를 따로 쟀습니다.

<!-- TABLE:E2 -->
| 모델 (FP32) | npu-default | npu-percentile | npu-mse | per-tensor | per-tensor-mse | pow2 |
|---|---|---|---|---|---|---|
| ResNet-20 ReLU (90.47%) | 90.51% / 90.48% | 90.52% / 90.85% | 90.50% / 90.70% | 90.52% / 90.52% | 90.50% / 90.90% | 90.53% / 90.90% |

fake-quant / 비트 정확 정수 엔진 정확도. 정수 엔진은 npu-default·per-tensor는 10,000장, 나머지는 2,000장에서 측정.

| 모델 | 스킴 | top-1 일치 (500장) | 출력 코드 불일치 | 첫 분기 레이어 | max Δlogit |
|---|---|---|---|---|---|
| ResNet-20 ReLU | npu-default | 99.2% | 45.9% | stem_conv | 0.459 |
| ResNet-20 ReLU | per-tensor | 99.4% | 45.1% | stem_conv | 0.612 |
<!-- /TABLE:E2 -->

관찰:

* **fake-quant는 정확도 예측기로는 충분히 정확합니다.** ResNet-20 ReLU에서 fake 90.51% vs 정수 90.48%, 이미지 단위 top-1 일치 99%대.
* **그러나 텐서 단위로는 전혀 같지 않습니다.** 각 conv의 국소(teacher-forced) 불일치는 0.03~0.2%(±1 LSB)뿐인데, 끝까지 정수로 실행하면 깊은 레이어에서 코드의 15~30%, 최종 로짓의 45%가 달라집니다. ±1 LSB가 다음 레이어의 반올림 결정을 바꾸는 나비효과이고, 분류 결과는 그래도 거의 바뀌지 않습니다. "fake-quant와 하드웨어 결과가 다르다"는 보고를 받으면 먼저 *어느 레이어에서, 국소로, 몇 LSB* 인지 물어야 하는 이유입니다.
* **국소 불일치의 출처**: conv/linear(고정소수점 곱셈기 + 바이어스 반올림), avgpool(half-away vs half-even 타이), pow2 스킴의 add(정확한 .5 타이). TFLite식 add(20비트 left-shift)는 일반 스킴에서 국소 불일치가 0입니다.
* **`sym-act`(대칭 int8 활성값)는 처음 실행에서 정수 엔진 정확도가 9%로 무너졌습니다.** "requant clamp가 곧 ReLU"라는 가정이 `qmin=-127`에서는 틀리기 때문입니다. fake-quant만 보면 절대 안 보이는 버그를 정수 엔진이 잡았고, export 단계에서 노드별 clamp 하한을 `zp`로 명시하도록 고쳤습니다 ([docs/INTEGER_DATAPATH.md](docs/INTEGER_DATAPATH.md)).

### E7. 정수 구현 세부의 정확도 비용

양자화 파라미터는 그대로 두고 정수 엔진의 구현만 바꿨습니다. 기준 구현(gemmlowp 이중 반올림, Q31 곱셈기, int32 누산기·바이어스)과의 **이미지 단위 top-1 일치율**은 시드 잡음이 없는 지표입니다.

<!-- TABLE:E7 -->
| RequantConfig | ResNet-20 ReLU: acc / 기준 대비 top-1 일치 |
|---|---|
| `tflite-m31-acc32-b32` | 91.00% / 100.0% |
| `single-m31-acc32-b32` | 91.00% / 99.4% |
| `half_even-m31-acc32-b32` | 90.85% / 99.4% |
| `truncate-m31-acc32-b32` | 89.30% / 95.2% |
| `floor-m31-acc32-b32` | 88.65% / 94.0% |
| `tflite-m15-acc32-b32` | 90.85% / 99.4% |
| `tflite-m7-acc32-b32` | 90.80% / 99.0% |
| `tflite-m3-acc32-b32` | 89.70% / 95.6% |

2000장 기준. 기준 구현은 `tflite-m31-acc32-b32`(gemmlowp 이중 반올림). 곱셈기 비트·누산기 폭이 줄어들 때 무엇이 먼저 무너지는지 보세요.
<!-- /TABLE:E7 -->

* 반올림 모드: TFLite single rounding과 half-even은 기준과 99.4% 일치하지만, **truncate/floor는 95%로 떨어지고 정확도도 1.6~2.3%p 잃습니다.** requant에서 "그냥 시프트"는 공짜가 아닙니다.
* 곱셈기 비트: 15비트까지는 거의 손실이 없고, 7비트에서 99.0%, **3비트(사실상 2의 거듭제곱 곱셈기)에서 95.6%** — 스케일을 2의 거듭제곱으로 제한하는 NPU라면 QAT 때 그 제약을 같이 학습시켜야 하는 근거입니다.
* 누산기/바이어스 폭: 표를 보세요 — 포화(saturation) 횟수와 함께 기록했습니다.

### E3. 정적 lint는 실제 손실을 예측하는가

<!-- TABLE:E3 -->
_(아직 실행되지 않음)_
<!-- /TABLE:E3 -->

### E4. 수술: CLE · 바이어스 보정 · 활성함수 교체 · QAT

<!-- TABLE:E4 -->
_(아직 실행되지 않음)_
<!-- /TABLE:E4 -->

### E5. 캘리브레이션 세트는 몇 장이면 되는가

<!-- TABLE:E5 -->
_(아직 실행되지 않음)_
<!-- /TABLE:E5 -->

### E6. PE-array 정렬 프루닝 — MACs와 사이클은 다르게 움직인다

<!-- TABLE:E6 -->
_(아직 실행되지 않음)_
<!-- /TABLE:E6 -->

## 데모

`docs/index.html`(= `demo/index.html`)은 위 JSON을 인라인한 정적 페이지입니다. 비용 모델을 JavaScript로 그대로 포팅해서
배열 크기·코어 수·DRAM 대역폭·depthwise 엔진 유무·비-ReLU 활성함수 실행 방식을 바꾸면 레이어별 사이클과 활용률이 즉시 다시 계산됩니다.
lint 리포트, PTQ 그리드와 레이어별 일치도, requant ablation, 수술, 캘리브레이션, 프루닝 Pareto, lint-vs-drop 산점도를 모두 담았습니다.

**라이브 데모:** https://claude.ai/code/artifact/8112e532-a441-4118-9cc1-fb939e3dab49 (같은 페이지가 저장소의 `docs/index.html`에도 들어 있어 GitHub Pages로 그대로 서비스할 수 있습니다.)


## 저장소 구조

```
npuloop/
├── graph/ir.py          torch.fx → StaticGraph (BN folding, shape 전파, 미지원 op 즉시 실패)
├── npu/spec.py, cost.py 가상 NPU 프리셋 4종 + weight-stationary systolic 비용 모델 (멀티코어 M/N 분할, DW 엔진, LUT/폴백, DRAM roofline)
├── lint/checks.py       정적·동적 준비도 점검 → efficiency / quant-robustness 점수
├── quant/               관측기(minmax·percentile·MSE), fake-quant(STE·LSQ), NPU식 삽입(prepare), 캘리브레이션, CLE, 바이어스 보정, 민감도, 활성함수 교체, QAT
├── intengine/           IntGraph export, gemmlowp/TFLite 동일 requant, NumPy 엔진, C++ 커널(ctypes), fake-vs-int 검증
├── prune/structured.py  채널 프루닝: uniform · aligned · cost-greedy(비용 모델 in-the-loop)
├── zoo/                 CIFAR-10 npz 로더, 모델, 재현 가능한 트레이너(resume)
└── cli.py               npuloop cost | lint | quantize
experiments/             E1–E7 스크립트 (재개 가능, results/*.json에 provenance 라벨과 함께 저장)
results/                 실험 결과 JSON
demo/                    build.py + index.template.html → 인라인 JSON 데모 페이지 (docs/index.html)
tests/                   pytest 44개 (참조 구현 대조, 비트 동일성, 정확성 회귀)
docs/                    DESIGN.md · INTEGER_DATAPATH.md · RELATED.md
```

## 빠른 시작

```bash
pip install -e .[dev]           # torch(CPU), numpy, pytest
python -m pytest -q             # 44 tests, ~5 s (C++ 커널은 첫 실행 때 g++로 컴파일)

# CIFAR-10 (npz 한 파일) 준비: tools/prepare_cifar10.py 참고
python -m npuloop.zoo.train --arch resnet --act relu --epochs 30 --out runs/resnet20_relu --data data/cifar10.npz

npuloop cost runs/resnet20_relu/best.pt --spec edge-10tops          # 레이어별 사이클/활용률 표
npuloop lint runs/resnet20_relu/best.pt --spec edge-10tops --data data/cifar10.npz
npuloop quantize runs/resnet20_relu/best.pt --data data/cifar10.npz --scheme npu-default --verify 500 --int-eval 2000

python examples/walkthrough.py --ckpt runs/resnet20_relu/best.pt --data data/cifar10.npz   # 1~2분짜리 전체 흐름 데모
bash experiments/run_all.sh     # E1–E7 전부 (CPU 4코어 기준 수 시간), results/*.json
python experiments/e8_scalesim.py   # 비용 모델 vs SCALE-Sim (pip install scalesim)
python demo/build.py            # results → demo/index.html, docs/index.html
```

파이썬 API 한 줄 요약:

```python
from npuloop.zoo import load_checkpoint, CIFAR10NPZ
from npuloop.graph import trace
from npuloop.npu import estimate
from npuloop.lint import lint
from npuloop.quant import prepare, calibrate, evaluate, QScheme
from npuloop.intengine import export_int_graph, NumpyEngine
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.verify import compare, agreement_table

m = load_checkpoint("runs/resnet20_relu/best.pt"); ds = CIFAR10NPZ("data/cifar10.npz")
print(estimate(trace(m), "edge-10tops").table())                     # simulated cycles
print(lint(trace(m), "edge-10tops", ds.calib_batch(64).numpy()).markdown())
qm = prepare(m, QScheme()); calibrate(qm, [ds.calib_batch(256)])
ig = export_int_graph(qm)                                             # int8/int32/M0,shift
rows, summary = compare(qm, ig, ds.calib_batch(200)); print(agreement_table(rows), summary)
print(NumpyEngine(ig).evaluate(ds, limit=2000), CppEngine(ig).evaluate(ds, limit=2000))
```


## 설계에서 신경 쓴 것들

* **양자화 지점이 NPU 데이터패스와 1:1.** ReLU 계열은 requant clamp에 융합되므로 활성함수 뒤에만 양자화기가 있고, SiLU/GELU/HardSwish는 int8 LUT라서 **활성함수 앞에 양자화기가 하나 더** 들어갑니다. avgpool 출력은 입력 스케일에 묶입니다(TFLite와 동일). "ReLU 모델이 INT8에 강하다"는 통념의 기계적 이유가 코드에 그대로 있습니다 ([docs/INTEGER_DATAPATH.md](docs/INTEGER_DATAPATH.md)).
* **정수 산술은 참조 구현과 비트 동일.** `SaturatingRoundingDoublingHighMul`, `RoundingDivideByPOT`, TFLite add의 20비트 left-shift, avgpool의 half-away 반올림을 그대로 구현하고, 스칼라 gemmlowp 참조와 랜덤 테스트로 대조합니다. NumPy 엔진과 C++ 커널은 **모든 중간 텐서**가 같아야 테스트가 통과합니다.
* **비용 모델은 시뮬레이터로 검증.** weight-stationary 타일당 `M + 2R + C − 2` 사이클 모델이 SCALE-Sim v3의 사이클 정확 결과와 레이어당 0.5% 이내(합계 0.03%)로 일치합니다(E8). 처음 만든 `M + R + C` 모델은 10–13% 낙관적이었고, 이 차이를 SCALE-Sim으로 찾아 고쳤습니다.
* **불일치를 두 관점으로 분리.** fake-quant와 정수 엔진의 차이를 "국소(각 op에 fake-quant 코드를 먹였을 때)"와 "전파(끝까지 정수로 실행)"로 나눠 재서, ±1 LSB의 국소 오차가 어떻게 누적되고 최종 정확도에는 왜 거의 영향이 없는지 보입니다.
* **재현 가능성과 출처 라벨.** 학습·프루닝·QAT는 시드 고정·재개 가능(`state.pt`)이고, 실험 JSON의 모든 레코드에 `provenance: measured | simulated` 라벨이 붙습니다. 프리셋 NPU는 공개 헤드라인 수치에 맞춘 **가정**이며 특정 벤더의 실제 구조가 아님을 코드와 문서에 명시했습니다.
* **테스트가 실제 버그를 잡았습니다.** 프루닝으로 새로 만든 BatchNorm이 eval 모드를 물려받지 않아 배치 통계로 평가되던 버그, half-even 반올림의 shift=0 예외, per-tensor 서브셋 평가가 클래스 순서로 정렬된 테스트셋 때문에 편향되던 문제를 모두 테스트/실험 단계에서 발견해 고쳤습니다(커밋 이력 참고).

## 한계와 다음 단계

* **CIFAR-10, 30 epoch, seed 1개.** 정확도 차이 0.2%p 이하는 잡음입니다(10k 이미지 표준오차 ≈ 0.3%p). 결론은 "방향"이지 소수점 둘째 자리가 아닙니다.
* **가상 NPU.** 실제 칩의 컴파일러(fusion, 타일링, 메모리 스케줄링)와 다릅니다. 비용 모델은 연산 사이클은 검증했지만 DRAM/SRAM 모델은 1차 근사(roofline)입니다. 실제 NPU 보드가 생기면 같은 IntGraph를 올려 정확도·지연을 대조하는 것이 첫 번째 할 일입니다.
* **지원 op가 좁습니다.** conv/dw-conv/linear/add/global-avgpool/elementwise 활성함수만. concat·upsample·attention은 `UnsupportedOpError`로 즉시 실패시킵니다(조용히 넘어가지 않기 위해). 검출 헤드(DFL, NMS)까지 정수로 옮기는 것이 다음 단계입니다.
* **프루닝 대상이 residual 밖의 내부 채널뿐**이라 절감 폭에 상한이 있습니다. residual stream 채널을 같이 자르려면 의존성 그래프가 필요합니다.
* **혼합 정밀도 없음.** 이 프로젝트의 NPU는 INT8 고정이라 비트 폭 탐색 대신 정수 구현 세부(E7)에 집중했습니다.

## 라이선스

[Apache License 2.0](LICENSE). 데이터: CIFAR-10 (Krizhevsky, 2009). SCALE-Sim은 검증 실험에서만 사용합니다(MIT).

