# npuloop 설계 문서

## 한 줄 요약

**"NPU에 올릴 모델의 압축 결정(양자화·프루닝·활성함수 교체)을 FLOPs와 fake-quant가 아니라,
가상 NPU 비용 모델과 비트 정확(bit-exact) 정수 엔진에 대고 내린다."**

```
        ┌──────────────────────┐   shapes/weights   ┌────────────────────┐
model ─►│ graph.trace (fx IR)  │───────────────────►│ npu.estimate       │ cycles, util, roofline
        │  BN folding          │                    │ lint               │ readiness findings/score
        └──────────┬───────────┘                    └─────────┬──────────┘
                   │                                          │ cost-in-the-loop
                   ▼                                          ▼
        ┌──────────────────────┐   prune / swap act   ┌────────────────────┐
        │ quant.prepare        │◄─────────────────────│ prune / surgery    │
        │  PTQ · QAT · CLE · BC│                      └────────────────────┘
        └──────────┬───────────┘
                   │ export
                   ▼
        ┌──────────────────────┐   bit-exact   ┌────────────────────┐
        │ intengine (IntGraph) │◄─────────────►│ C++ kernels        │
        │ NumPy reference      │               │ (ctypes)           │
        └──────────┬───────────┘               └────────────────────┘
                   │ verify: fake-quant vs integer, local vs propagated mismatch
                   ▼
              experiments / demo
```

## 모듈

| 모듈 | 역할 | 핵심 결정 |
|---|---|---|
| `graph/ir.py` | torch.fx → `StaticGraph` (input/conv/linear/add/act/pool/flatten/output) | BN은 IR 생성 시점에 conv로 접는다. NPU는 BN을 모른다. 지원하지 않는 op은 `UnsupportedOpError`로 즉시 실패시킨다(조용히 넘어가지 않음). |
| `npu/spec.py` | 가상 NPU 파라미터(`NPUSpec`)와 프리셋 | 프리셋은 공개된 헤드라인 수치(TOPS, 대역폭)에 맞춘 **가정**이며 특정 벤더의 마이크로아키텍처가 아니다. 모든 결과의 출처 라벨은 `simulated`. |
| `npu/cost.py` | 해석적 비용 모델 | weight-stationary systolic array 타일 모델(SCALE-Sim WS 연산 사이클과 타일당 1사이클 이내로 일치, E8), 멀티코어 M/N 분할 선택, depthwise 엔진, LUT/fused/fallback 활성함수, DRAM roofline. 같은 양(MAC·활성값/가중치 바이트·DRAM 바이트·벡터/호스트 원소)에 pJ 상수(Horowitz 2014 자릿수)를 곱한 에너지 추정(`simulated`). |
| `lint/checks.py` | 정적(가중치·shape)/동적(캘리브레이션 배치) 점검 → 효율성·양자화 강건성 점수 | 점수는 **레이어 비율** 기반이라 깊이에 따라 포화되지 않는다. 점수의 예측력은 실험으로 검증한다. |
| `quant/` | NPU 방식의 fake-quant 삽입(`prepare`), 관측기, 캘리브레이션, QAT, CLE, 바이어스 보정, 민감도, 활성함수 교체 | 양자화 지점이 NPU 데이터패스와 1:1: ReLU 계열은 requant clamp에 융합, 비-ReLU는 LUT라서 pre-activation을 int8로 한 번 더 양자화. avgpool 출력은 입력 스케일에 묶임. |
| `intengine/` | `IntGraph` export, NumPy 정수 엔진, C++ 커널(im2col + int32 GEMM), 검증, 노드별 실측(`bench`), `.npuloop` 파일(`serialize`), 파이썬 없는 독립 실행기(`cpp/int8_runner.cpp`) | requant는 gemmlowp/TFLite와 비트 동일(`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`) — TFLite reference 커널과 1,000장 × 모든 텐서에서 확인(E11; conv/pool 이중 반올림, fc 단일 반올림은 노드별 `attrs["rounding"]`로). `RequantConfig`로 일부러 "싸구려" 구현(곱셈기 비트 수, 반올림 모드, 누산기 폭, 바이어스 폭)을 흉내 내 정확도 비용을 측정. |
| `prune/structured.py` | fx 그래프에서 찾은 채널 그룹에 uniform / aligned / cost-greedy 프루닝 | 그룹 = conv/linear 출력 채널을 활성함수·depthwise conv를 지나 소비자까지 따라간 것(체인, concat 브랜치, MLP 은닉). residual add·pool·matmul·LayerNorm에 닿는 채널은 제외. `cost-greedy`는 비용 모델을 루프 안에서 호출해 "사이클 절감 / 중요도 손실"이 가장 큰 그룹부터 PE-array 폭 단위로 제거. |
| `zoo/` | npz 데이터(CIFAR-10·Imagenette, 층화 검증 분할), 모델(ResNet-20 계열, MobileNetV2-CIFAR, ViT, Inception), 재현 가능한 트레이너 | 모든 활성함수는 `nn.Module`이라 교체·추적이 기계적으로 가능. 트레이너는 시드 고정·재개 가능하며 체크포인트는 검증 정확도로만 고른다. |

## 정수 데이터패스 정의 (가상 NPU의 "ISA")

* 활성값: per-tensor asymmetric **uint8** (zp ∈ [0,255]); 가중치: per-output-channel symmetric **int8** ([-127,127]); 바이어스: int32, 스케일 `s_in·s_w[c]`.
* conv/linear: int32 누산 → `+bias` → `MultiplyByQuantizedMultiplier(acc, M0[c], shift[c])` → `+zp_out` → clamp. clamp가 곧 ReLU/ReLU6.
* add: TFLite와 동일(입력 두 개를 20비트 left-shift 도메인으로 rescale → 합 → 한 번만 requant).
* 비-ReLU 활성함수: 256-entry LUT (입력 int8 코드 → 출력 코드).
* global avgpool: int32 합 → 반올림 나눗셈(round half away), 스케일 유지.
* 입력: 호스트가 float → uint8 (round half to even).

## 실험 계획 (E1–E16)

결과 전문은 [EXPERIMENTS.md](EXPERIMENTS.md), 각 행의 링크는 그 실험의 절로 갑니다.

| # | 질문 | 방법 |
|---|---|---|
| [E1](EXPERIMENTS.md#e1) | 베이스라인 | ResNet-20 {ReLU, SiLU}, MobileNetV2-0.5 (ReLU6), 30 epochs OneCycle, seed 0 (계획했던 HardSwish·GELU 베이스라인은 CPU 시간 때문에 제외) |
| [E2](EXPERIMENTS.md#e2) | PTQ 스킴별 INT8 정확도와 fake-quant↔정수 엔진 일치 | 스킴 × 모델, fake-quant 정확도 + 비트 정확 정수 엔진 정확도 |
| [E3](EXPERIMENTS.md#e3) | 정적 lint 점수가 실제 INT8 손실을 예측하는가 | 모델/스킴별 quant-robustness 점수 vs 측정된 정확도 손실 |
| [E4](EXPERIMENTS.md#e4) | 수술(CLE·BC·활성함수 교체·QAT)이 손실을 얼마나 회복하는가 | per-tensor MobileNetV2에 CLE+BC, SiLU→ReLU/HardSwish 교체 후 healing, QAT |
| [E5](EXPERIMENTS.md#e5) | 캘리브레이션 세트 크기/구성 | N ∈ {8…2048}, 무작위 vs 클래스 균형, 시드 3개 |
| [E6](EXPERIMENTS.md#e6) | PE-array 정렬 프루닝 | uniform vs aligned vs cost-greedy: 정확도(짧은 fine-tune) vs 비용 모델 사이클 vs FLOPs |
| [E7](EXPERIMENTS.md#e7) | 정수 구현 선택의 정확도 비용 | RequantConfig ablation: 반올림 모드, 곱셈기 비트, 누산기/바이어스 폭 |
| [E8](EXPERIMENTS.md#e8) | 비용 모델은 믿을 만한가 | SCALE-Sim v3(weight-stationary, 사이클 정확) 대조: 총 사이클과 레이어별 오차 |
| [E9](EXPERIMENTS.md#e9) | 고객 모델 인테이크 | 가상 고객 2곳(ViT, concat 분기 CNN)을 접수 → 진단 → 처방 → 적용까지 돌려 예측 절감과 실측 정확도를 나란히 |
| [E10](EXPERIMENTS.md#e10) | 검증 엔진은 얼마나 걸리는가 | NumPy·C++ 비트 정확 엔진의 노드별 wall-clock을 호스트 CPU에서 실측해 같은 노드의 비용 모델 사이클 옆에 둠(실측 vs 모델을 섞지 않기 위해) |
| [E11](EXPERIMENTS.md#e11) | 실제 런타임과 비트가 맞는가 | TFLite full-integer 모델의 스케일·가중치·바이어스를 그대로 읽어 IntGraph를 만들고 reference 커널과 모든 텐서를 대조 — conv·pool은 이중 반올림, fc는 단일 반올림으로 1,000장 완전 일치 |
| [E12](EXPERIMENTS.md#e12) | 데이터셋·입력 크기 축 | Imagenette 128×128에서 ResNet-20(stem stride 2)을 학습해 인테이크·PTQ·정수 엔진·실측을 같은 파이프라인으로 |
| [E13](EXPERIMENTS.md#e13) | 비용 모델 vs 벤더 추정기 | 같은 아키텍처에서 INT8 TFLite를 만들어 Arm Vela(Ethos-U55-256)와 `npu.estimate`를 대조. 연산자별 사이클·MAC 활용률·CPU 폴백. 다섯 망 모두에서 이 비용 모델이 낙관적(비율 0.29–0.84) | `experiments/e13_vela.py` |
| [E14](EXPERIMENTS.md#e14) | E7의 반올림 축은 시드에 강건한가 | ResNet-20 ReLU/SiLU, MobileNetV2-0.5를 시드 1·2로 다시 학습(`experiments/run_seeds.sh`)해 2단계 반올림 5모드를 3시드 × test 10,000장에서, 기준 구현·fake-quant 대비 **같은 이미지의 쌍 표준오차**와 함께 | `experiments/e14_rounding_seeds.py` |
| [E15](EXPERIMENTS.md#e15) | fake-quant ↔ 정수 차이에 오차 막대를 붙이면 | 5개 CIFAR 모델 + Imagenette-128 ResNet-20을 test 전체에서 FP32·fake·정수로 세 번 평가해 (정수 정답 − fake 정답)의 표본 표준편차로 쌍 SE·95% CI, test 전체의 출력 코드 불일치, 국소/전파 분해 | `experiments/e15_fidelity.py` |
| [E16](EXPERIMENTS.md#e16) | LayerNorm을 정수로 에뮬레이션하면 transformer 격차가 닫히는가 | `npuloop.quant.emulate`로 fake-quant의 LayerNorm을 정수 엔진 산술로 바꾸고(export 불변) ViT-128/6에서 국소·전파 불일치·top-1 일치·정확도 차이를 재측정 — 국소 24.8% → 0이지만 출력 코드 불일치는 72.7% → 65.7% | `experiments/e16_ln_emulation.py` |

## 검증 원칙

* 모든 정수 프리미티브는 스칼라 gemmlowp 참조 구현과 랜덤 테스트로 비교한다.
* NumPy 엔진과 C++ 커널은 **모든 중간 텐서**가 비트 단위로 같아야 한다(테스트로 강제).
* fake-quant ↔ 정수 엔진 불일치는 "국소(teacher-forced)"와 "전파" 두 관점으로 나눠 보고한다.
* 실험 JSON에는 `provenance: measured | simulated` 라벨을 붙인다.
* 테스트는 통과 여부를 적는 자리가 아니라 버그를 잡는 자리다. 실제로 잡은 것들: 프루닝으로 새로 만든
  BatchNorm이 eval 모드를 물려받지 않던 버그, half-even 반올림의 shift=0 예외, 서브셋 평가가 클래스
  순서로 편향되던 문제.

## 저장소 구조

```
npuloop/
├── graph/ir.py          torch.fx → StaticGraph (BN folding, shape 전파)
├── npu/                 가상 NPU 프리셋 4종 + weight-stationary systolic 비용 모델
├── lint/                정적·동적 준비도 점검 → efficiency / quant-robustness 점수
├── quant/               관측기 · fake-quant · NPU식 양자화기 삽입 · CLE · 바이어스 보정 · QAT · 정수 LayerNorm 에뮬레이션
├── intengine/           IntGraph export · gemmlowp/TFLite 동일 requant · NumPy 엔진 · C++ 커널 · 검증 · `.npuloop` 직렬화
├── prune/               fx 그래프 채널 그룹에 uniform · aligned · cost-greedy(비용 모델 in-the-loop) 프루닝
├── zoo/                 npz 로더(CIFAR-10·Imagenette) · 모델 · 재현 가능한 트레이너
└── cli.py
experiments/             E1–E20 스크립트 (재개 가능)
results/                 실험 결과 JSON (provenance 라벨 포함)
demo/                    build.py + 템플릿 → docs/index.html
tests/                   pytest 스위트 (개수는 실행 로그 기준)
examples/                walkthrough.py · quickstart/ (번들 체크포인트 + CIFAR-10 샘플)
tools/                   표 생성 · 데이터셋 준비 · 보고서 빌드 · oneDNN 재현 스크립트
docs/                    EXPERIMENTS · USAGE · DESIGN · INTEGER_DATAPATH · RELATED · report/ · upstream/
```
