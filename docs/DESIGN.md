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
| `npu/cost.py` | 해석적 비용 모델 | weight-stationary systolic array 타일 모델(SCALE-Sim의 1차 모델과 동일), 멀티코어 M/N 분할 선택, depthwise 엔진, LUT/fused/fallback 활성함수, DRAM roofline. |
| `lint/checks.py` | 정적(가중치·shape)/동적(캘리브레이션 배치) 점검 → 효율성·양자화 강건성 점수 | 점수는 **레이어 비율** 기반이라 깊이에 따라 포화되지 않는다. 점수의 예측력은 실험으로 검증한다. |
| `quant/` | NPU 방식의 fake-quant 삽입(`prepare`), 관측기, 캘리브레이션, QAT, CLE, 바이어스 보정, 민감도, 활성함수 교체 | 양자화 지점이 NPU 데이터패스와 1:1: ReLU 계열은 requant clamp에 융합, 비-ReLU는 LUT라서 pre-activation을 int8로 한 번 더 양자화. avgpool 출력은 입력 스케일에 묶임. |
| `intengine/` | `IntGraph` export, NumPy 정수 엔진, C++ 커널, 검증 | requant는 gemmlowp/TFLite와 비트 동일(`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`). `RequantConfig`로 일부러 "싸구려" 구현(곱셈기 비트 수, 반올림 모드, 누산기 폭, 바이어스 폭)을 흉내 내 정확도 비용을 측정. |
| `prune/structured.py` | 채널 프루닝(uniform / aligned / cost-greedy) | 프루닝 대상은 residual stream 밖의 내부 채널. `cost-greedy`는 비용 모델을 루프 안에서 호출해 "사이클 절감 / 중요도 손실"이 가장 큰 그룹부터 PE-array 폭 단위로 제거. |
| `zoo/` | CIFAR-10 데이터(npz), 모델(ResNet-20 계열, MobileNetV2-CIFAR), 재현 가능한 트레이너 | 모든 활성함수는 `nn.Module`이라 교체·추적이 기계적으로 가능. 트레이너는 시드 고정·재개 가능. |

## 정수 데이터패스 정의 (가상 NPU의 "ISA")

* 활성값: per-tensor asymmetric **uint8** (zp ∈ [0,255]); 가중치: per-output-channel symmetric **int8** ([-127,127]); 바이어스: int32, 스케일 `s_in·s_w[c]`.
* conv/linear: int32 누산 → `+bias` → `MultiplyByQuantizedMultiplier(acc, M0[c], shift[c])` → `+zp_out` → clamp. clamp가 곧 ReLU/ReLU6.
* add: TFLite와 동일(입력 두 개를 20비트 left-shift 도메인으로 rescale → 합 → 한 번만 requant).
* 비-ReLU 활성함수: 256-entry LUT (입력 int8 코드 → 출력 코드).
* global avgpool: int32 합 → 반올림 나눗셈(round half away), 스케일 유지.
* 입력: 호스트가 float → uint8 (round half to even).

## 실험 계획 (E1–E7)

| # | 질문 | 방법 |
|---|---|---|
| E1 | 베이스라인 | ResNet-20 {ReLU, SiLU, HardSwish, GELU}, MobileNetV2-0.5 (ReLU6), 30 epochs OneCycle, seed 0 |
| E2 | PTQ 스킴별 INT8 정확도와 fake-quant↔정수 엔진 일치 | 스킴 × 모델, fake-quant 정확도 + 비트 정확 정수 엔진 정확도 |
| E3 | 정적 lint 점수가 실제 INT8 손실을 예측하는가 | 모델/스킴별 quant-robustness 점수 vs 측정된 정확도 손실 |
| E4 | 수술(CLE·BC·활성함수 교체·QAT)이 손실을 얼마나 회복하는가 | per-tensor MobileNetV2에 CLE+BC, SiLU→ReLU/HardSwish 교체 후 healing, QAT |
| E5 | 캘리브레이션 세트 크기/구성 | N ∈ {8…2048}, 무작위 vs 클래스 균형, 시드 3개 |
| E6 | PE-array 정렬 프루닝 | uniform vs aligned vs cost-greedy: 정확도(짧은 fine-tune) vs 비용 모델 사이클 vs FLOPs |
| E7 | 정수 구현 선택의 정확도 비용 | RequantConfig ablation: 반올림 모드, 곱셈기 비트, 누산기/바이어스 폭 |

## 검증 원칙

* 모든 정수 프리미티브는 스칼라 gemmlowp 참조 구현과 랜덤 테스트로 비교한다.
* NumPy 엔진과 C++ 커널은 **모든 중간 텐서**가 비트 단위로 같아야 한다(테스트로 강제).
* fake-quant ↔ 정수 엔진 불일치는 "국소(teacher-forced)"와 "전파" 두 관점으로 나눠 보고한다.
* 실험 JSON에는 `provenance: measured | simulated` 라벨을 붙인다.
