# npuloop — 가상 NPU 제약에 맞춘 모델 변경·학습·경량화 실험

[![tests](https://github.com/sokldjs554/npuloop/actions/workflows/ci.yml/badge.svg)](https://github.com/sokldjs554/npuloop/actions/workflows/ci.yml)

> **모델을 가상 NPU의 연산 제약에 맞게 변경하고, 재학습·양자화·프루닝 전후의 정확도와 추정 실행 비용을 비교한 연구·개발 프로젝트입니다.**
> PyTorch 학습·실험 스크립트와 Python CLI를 연결하고, 내보낸 정수 그래프는 NumPy/C++로 검증합니다.
> 정확도는 저장된 실험의 측정값이고, NPU 사이클은 가상 프리셋의 추정값입니다. 실제 칩에서 측정한 값은 없습니다.

## ▶ [공개 워크벤치 — 설치·서버·API 키 없이 바로 열립니다](https://sokldjs554.github.io/npuloop/)

모델 변경 전후 비교, NPU 비용 분석, 정수 검증, E1–E20 기록, 그리고 **화면의 모든 숫자에 대한 원본 JSON**을 브라우저에서 엽니다.
로컬에서는 `docs/index.html`을 그냥 열어도 같습니다.

![npuloop 워크벤치 둘러보기](docs/model_study_walkthrough.gif)

52초 · 컷 12장 — 저장된 실험 기록을 실제로 조작해 캡처했습니다(`python tools/record_walkthrough.py`로 다시 만듭니다).
활성함수 교체 직후와 회복 학습 후 → 변경안 비교 → 호스트 폴백 97.4% → 출력 코드 불일치 72.7% → LayerNorm 정수 교체와 ImageNet 규모 → 그 숫자의 원본 JSON 순서입니다.

## 30초 요약

- **모델 변경/학습:** ResNet-20의 SiLU→ReLU 교체로 FP32가 90.34%→40.36%로 하락했고, 3 epoch 재학습 후 **89.74%**까지 회복했습니다.
- **경량화:** 구조적 프루닝에서 MACs **40.81M→20.76M**, FP32 **89.59%→87.22%**를 함께 기록해 비용 절감과 정확도 손실을 같이 봤습니다.
- **NPU 조건:** ViT GELU→ReLU 변경은 FP32를 **80.98%→80.95%**로 유지했지만, 추정 비용 효과는 지원 연산 조건에 따라 크게 달랐습니다.
- **핵심 결과:** 정확도 차이는 최대 **0.20%p**였지만 출력 정수 코드의 **28.5–72.7%**가 달랐습니다. 가장 큰 국소 원천(ViT의 LayerNorm)을 정수 산술로 바꿔 국소 불일치를 24.77%→0.00%로 없애도 출력 불일치는 **72.7%→65.7%**까지만 내려갑니다.
- **범위:** 가상 NPU 비용과 호스트 정수 실행을 다루며, 실제 NPU 성능 측정은 아닙니다.

정확도 유사성과 비트 일치는 구분합니다. 국소 불일치는 고정 배치, 정확도와 출력 코드는 전체 테스트셋의 측정이고, 신뢰구간의 0 포함은 동등성 입증이 아닙니다.

## 이 숫자들을 어디까지 믿어도 되는가

<!-- TABLE:HEADLINE -->
| 질문 | `results/*.json`이 답하는 것 | 단, 검증되지 않은 것 |
|---|---|---|
| **정수 엔진을 믿어도 되는가** ([E11](docs/EXPERIMENTS.md#e11)) | TFLite reference 커널과 **1,000장 × 모든 텐서 0 불일치**(conv·pool은 gemmlowp 이중 반올림, fc는 단일 반올림) | 대조 범위는 conv(stride 1·2)·MEAN·fully-connected. depthwise·add·softmax는 미대조이고, TFLite 자신도 XNNPACK 경로와 reference가 155/1,000장 다릅니다 |
| **비용 모델이 맞는가** ([E8](docs/EXPERIMENTS.md#e8)) | SCALE-Sim v3와 12개 (모델, 배열) 조합에서 합계 오차 ≤ 0.08%, 최악 레이어 0.5% | 검증된 것은 dense conv·linear의 연산 사이클(단일 코어, 메모리 스톨 없음). depthwise·벡터 패스·멀티코어·DRAM roofline은 검증되지 않았습니다 |
| **그럼 실리콘에 가까운가** ([E13](docs/EXPERIMENTS.md#e13)) | Arm Vela 대비 **5/5 낙관적**(사이클 비 0.29~0.84) | **아니오.** 둘 다 해석적 추정기이고 Vela는 컴파일된 스케줄(fusion·타일링)을, 이쪽은 레이어를 하나씩 셉니다. E8의 일치는 하드웨어 충실도가 아니라 같은 이상화를 공유하는 두 모델이 같은 식을 같게 구현했다는 확인입니다 |
| **fake-quant 정확도를 믿어도 되는가** ([E2](docs/EXPERIMENTS.md#e2)·[E15](docs/EXPERIMENTS.md#e15)) | 95% Wald 구간의 0 포함 12/12; 최대 관측 정확도 차이 0.20%p (정수 − 모의, 전체 테스트셋). 허용 오차 내 동등성 입증은 아님 | E15·E16의 16개 비교를 다중비교 보정하지 않은 값입니다(E16까지 합치면 최대 \|Δ\|/SE 2.2). 모델 5개 + Imagenette 1개, 스킴 2개 범위 |
| **텐서도 같은가** ([E15](docs/EXPERIMENTS.md#e15)·[E16](docs/EXPERIMENTS.md#e16)) | 그러나 출력 코드는 28.5~72.7%가 다릅니다. 가장 큰 국소 원천(ViT의 LayerNorm)을 정수 산술로 바꿔 국소 불일치를 24.77% → 0.00%로 없애도 출력 코드 불일치는 72.7% → 65.7%까지만 내려갑니다 | 출력 코드 불일치는 test 전체, 국소 불일치는 125/250장 배치의 teacher-forced 값입니다 — 두 수를 같은 문장에서 섞지 마세요 |
| **싸구려 반올림의 값** ([E14](docs/EXPERIMENTS.md#e14)) | requant에서 반올림 대신 시프트를 쓰면 18개 (모델, 시드, 모드) 조합 **18개 전부**에서 손해입니다(-3.37~-0.90%p, \|Δ\|/SE 4.4~11.8) | 강건한 것은 부호와 유의성이고 **크기는 아닙니다**: SiLU의 truncate는 시드에 따라 −1.36~−2.98%p(시드 SD가 효과의 37%). 한 시드 숫자를 그 모드의 비용으로 인용하면 안 됩니다 |
| **그 밖의 정수 구현 세부** ([E7](docs/EXPERIMENTS.md#e7)) | 13개 구현 구성 중 7개는 무손실, 2개는 모델을 무너뜨립니다 (곱셈기 7비트·누산기 20비트까지는 공짜, 바이어스 12비트·누산기 16비트는 붕괴) | 바이어스 폭 축은 지수 조정 없는 **클리핑**을 재고(시판 NPU는 더 넓습니다 — Vela는 40비트), 누산기 포화는 부분합이 아니라 최종합에 한 번만 걸리며, 결과는 K ≤ 1,280인 이 모델들에 한정됩니다 |
<!-- /TABLE:HEADLINE -->

**세 번째 열이 이 저장소의 성격입니다.** 가상 NPU 비용 모델은 벤더 추정기와 대면시키면 낙관적이고(E13), 그 사실을 지우지 않고 적었습니다.

CPU 학습 중 만난 [oneDNN의 1×1 conv backward-weights 오버플로](docs/upstream/onednn_1x1_bwd_weights_rtus_overflow.md)는 benchdnn으로 단독 재현하고
2줄 패치까지 검증해 [uxlfoundation/oneDNN#6035](https://github.com/uxlfoundation/oneDNN/issues/6035)로 보고했습니다. 이 저장소의 실험 모델들은 stride > 1인 1×1 conv의 입력 채널이 모두 16 이상이라 결과와 무관하며,
CI는 `ONEDNN_MAX_CPU_ISA=AVX2`로도 전체 스위트를 돌립니다.

## 모델을 어떻게 바꾸고 검증했는가

| 실험 | 변경과 학습 | 결과 |
|---|---|---|
| **E4 · 활성함수 변경과 복구 학습** | ResNet-20의 SiLU를 ReLU로 교체하고 3 epoch 재학습 | FP32 **90.34% → 교체 직후 40.36% → 재학습 후 89.74%**. 같은 실험의 INT8 정수 정확도는 90.26% → 89.81% |
| **E6 · 구조적 프루닝** | 균일·배열 정렬·비용 기반 채널 선택을 비교하고 각각 3 epoch 미세조정 | 균일 0.5 조건에서 MACs **40.81M → 20.76M**, `edge-10tops` 추정 사이클 **34,417 → 27,739**, FP32 **89.59% → 87.22%**. 연산량 감소와 추정 사이클 감소가 비례하지 않습니다 |
| **E9 · NPU 조건에 따른 모델 변경** | ViT의 GELU를 ReLU로 교체하고 3 epoch 재학습 | FP32 **80.98% → 80.95%**. strict 프리셋의 추정 사이클은 **2,503,878 → 1,708,230**, LUT 지원 프리셋에서는 **52,246 → 52,054** |

근거: [E4](results/e4_surgery.json) · [E6](results/e6_pruning.json) · [E9](results/e9_customer_intake.json). 이 표는 기존 결과의 요약이며 서로 다른 실험 사이를 전후 비교로 연결하지 않습니다.
전체 흐름은 **데이터 준비·분할 → PyTorch 학습 → 구조 변경·미세조정 → 보정·양자화 → 정수 export → 정확도·비용·출력 비교**이고, 모듈 구조와 정수 데이터패스 정의는 [docs/DESIGN.md](docs/DESIGN.md)에 있습니다.

PTQ·QAT·보정은 개선 여부를 **비교하는** 실험으로 다뤘습니다. E4의 SiLU 모델은 PTQ 정수 90.26%에서 2 epoch QAT 후 90.01%로 낮아졌습니다. 적용했다는 사실만으로 개선을 주장하지 않습니다.

## 고객이 체크포인트를 보내오면 (E9)

가상 고객 두 곳을 가정했습니다. **고객 A는 ViT**(Transformer 블록 6개, LayerNorm 13개, GELU), **고객 B는 concat 분기 CNN**.
둘 다 이 저장소가 원래 거부하던 구조라 `concat`·`matmul`·`softmax`·`layernorm`·`transpose`를 IR·양자화·정수 엔진(NumPy와 C++ 모두)·비용 모델·lint에 새로 넣었습니다.
표의 사이클과 온칩 실행 여부는 가상 프리셋의 추정·판정이지 실측이 아닙니다.

<!-- TABLE:E9 -->
**(a) 인테이크 요약** — `npuloop intake`가 낸 값 (사이클은 비용 모델, INT8은 정수 엔진 실측)

| 모델 | 파라미터 | MACs | edge-10tops cycles (활용률) | strict cycles | 온칩 실행 | lint eff / q-rob |
|---|---|---|---|---|---|---|
| 고객 B · Inception-32 | 423,066 | 54.0M | 31,224 (21.1%) | 31,224 | 예 / strict 예 | 91 / 97 |
| 고객 A · ViT-128/6 | 810,890 | 57.0M | 52,246 (13.3%) | 2,503,878 | 예 / strict 아니오 | 78 / 75 |
| 기존 · MobileNetV2-0.5 | 700,490 | 28.0M | 87,056 (3.5%) | 569,619 | 예 / strict 예 | 77 / 98 |
| 기존 · ResNet-20 ReLU | 272,474 | 40.8M | 34,417 (14.5%) | 34,417 | 예 / strict 예 | 80 / 97 |
| 기존 · ResNet-20 SiLU | 272,474 | 40.8M | 34,785 (14.3%) | 1,554,865 | 예 / strict 아니오 | 72 / 75 |

**(b) 처방을 실제로 적용한 결과**

| 모델 | 처방 | FP32 | INT8 정수 엔진 | edge-10tops cycles | strict cycles |
|---|---|---|---|---|---|
| 고객 A · ViT-128/6 | swap {'gelu': 'relu'} + heal 3 epochs | 80.98% → 80.95% | 80.80% → 81.00% | 52,246 → 52,054 | 2,503,878 → 1,708,230 |

온칩 실행 = 가상 프리셋에서 모든 op 지원으로 판정됨(호스트 폴백 없음, 실측 아님). strict = LUT·softmax·layernorm 지원이 없는 프리셋.
<!-- /TABLE:E9 -->

* **같은 처방도 칩이 다르면 값이 달라집니다.** GELU→ReLU 처방은 LUT가 있는 `edge-10tops`에서 **0.37% 절감**, LUT가 없는 strict 프리셋에서는 **−32%**입니다. 비용 모델을 루프 안에 두면 "이 칩에서 이 수술은 의미가 없다"를 **재학습 전에** 압니다.
* **부족하면 부족하다고 말합니다.** 고객 A는 strict NPU에서 사이클의 **97%가 호스트 폴백**이라, 활성함수만 바꿔서는 layernorm 13개와 softmax 6개가 남습니다. 리포트의 결론은 모델 수술이 아니라 **벡터 유닛이 있는 프리셋**입니다.
* **attention이 모의·정수를 가장 크게 가릅니다.** 출력 코드 불일치가 CNN의 30~50%에서 74%로 오르고, 가장 큰 국소 원천은 LayerNorm(국소 24.7%)이었습니다. 원인을 제거해 보는 후속 실험이 [E16](docs/EXPERIMENTS.md#e16)입니다.

E9의 FP32는 전체 테스트셋, INT8은 고정 2,000장 부분집합입니다. 각 열의 변경 전후만 비교하며 FP32와 INT8의 차이를 양자화 손실로 계산하지 않습니다.

## 한계

* **주요 분류 평가는 CIFAR-10 모델 5종과 Imagenette 1종이며 E17은 초해상 2종입니다.** E14의 반올림 축만 3개 학습 반복을 씁니다. 작은 차이의 해석은 대응 신뢰구간을 따라야 하고, 0.2%p 이하를 일괄 잡음이라 판정하지 않습니다.
* **체크포인트를 바꾸면 결론 일부가 흔들립니다** — E3의 레이어 수준 상관과 E7의 3비트 곱셈기 손실이 재학습 후 달라졌습니다.
* **가상 NPU입니다.** 실제 칩의 컴파일러(fusion·타일링·메모리 스케줄링)와 다르고, 비용 모델은 dense GEMM 사이클만 검증했습니다(E8). 벤더 추정기 대조는 E13.
* **실측한 시간은 호스트 CPU의 검증 엔진뿐입니다**(E10). NPU 사이클·에너지는 전부 `simulated`이고 에너지 상수는 자릿수 추정입니다.
* **지원 op**: conv/dw-conv/grouped-conv · linear · add · mul · concat · matmul · softmax · layernorm · transpose/reshape · pool · elementwise 활성함수. upsample·detection 헤드·KV 캐시는 아직입니다.
* **혼합 정밀도 없음**(INT8 고정), **프루닝 대상은 residual 밖의 내부 채널뿐**, **CLE의 이득을 보이지 못했습니다**(이 체크포인트들에는 CLE가 고칠 불균형이 없습니다).
* 각 항목의 근거와 나머지 한계는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)의 해당 실험 절에 있습니다.

## 직접 돌려보기

데이터셋 다운로드도 학습도 없이 도는 경로를 넣어 두었습니다 — 작은 체크포인트 2개와 CIFAR-10 샘플 1,012장(`examples/quickstart/`, 6 MB).
CPU 4코어 기준 **1분 안팎**(C++ 커널 컴파일 포함).

```bash
git clone https://github.com/sokldjs554/npuloop && cd npuloop && pip install -e .[dev]
make quickstart      # intake → quantize/verify → export + 파이썬 없는 C++ 실행기
python -m pytest -q  # C++ 커널은 첫 실행 때 g++로 컴파일됩니다
make tables          # results/*.json → README·docs/EXPERIMENTS.md의 표 재생성
```

CLI는 여섯 동사입니다: `npuloop cost | lint | intake | quantize | export | bench`.
그중 `npuloop quantize`가 **fake-quant와 비트 정확 정수 엔진을 노드별로 대조**하는 명령입니다.

모델 수정·회복 학습을 **새로** 돌려 그 결과를 데모에 얹으려면 `python -m npuloop.study_runner`를 씁니다
(생성된 `study.json`을 워크벤치의 *내 모델로 실험 실행*에 넣으면 학습 곡선까지 그대로 표시됩니다 — [실험 실행기 안내](docs/STUDY_RUNNER.md)).
데이터셋 준비·전체 학습·실험 재현·파이썬 API는 [docs/USAGE.md](docs/USAGE.md)에 있습니다.

<details>
<summary><b>실험 20개 (E1–E20)</b> — 각 실험이 답하는 질문</summary>

| # | 질문 | |
|---|---|---|
| E1 | 베이스라인 5개와 정적 분석(사이클·활용률·lint 점수) | [→](docs/EXPERIMENTS.md#e1) |
| E2 | PTQ 스킴 7개 × 모델 5개, fake-quant ↔ 비트 정확 정수 엔진 일치 | [→](docs/EXPERIMENTS.md#e2) |
| E3 | 정적 lint 점수가 실제 INT8 손실을 예측하는가 | [→](docs/EXPERIMENTS.md#e3) |
| E4 | 수술(CLE·바이어스 보정·활성함수 교체·QAT)이 손실을 얼마나 회복하는가 | [→](docs/EXPERIMENTS.md#e4) |
| E5 | 캘리브레이션은 몇 장이면 되는가 | [→](docs/EXPERIMENTS.md#e5) |
| E6 | PE-array 정렬 프루닝 — MACs와 사이클은 다르게 움직인다 | [→](docs/EXPERIMENTS.md#e6) |
| E7 | 정수 구현 세부(반올림·곱셈기 비트·누산기/바이어스 폭)의 정확도 비용 | [→](docs/EXPERIMENTS.md#e7) |
| E8 | 비용 모델은 믿을 만한가 — SCALE-Sim v3 대조 | [→](docs/EXPERIMENTS.md#e8) |
| E9 | 고객 모델 인테이크 — 접수 → 진단 → 처방 → 적용 | [→](docs/EXPERIMENTS.md#e9) |
| E10 | 검증 엔진은 얼마나 걸리는가 — 실측과 모델 사이클을 같은 표에 | [→](docs/EXPERIMENTS.md#e10) |
| E11 | 실제 런타임과 비트가 맞는가 — TensorFlow Lite reference 커널 교차 검증 | [→](docs/EXPERIMENTS.md#e11) |
| E12 | 데이터셋·입력 크기 축 — Imagenette 128×128 | [→](docs/EXPERIMENTS.md#e12) |
| E13 | 벤더 추정기와 대면 — Arm Vela(Ethos-U55-256) 대조 | [→](docs/EXPERIMENTS.md#e13) |
| E14 | 반올림 모드의 비용은 체크포인트에 강건한가 — 시드 3개 × 10,000장 | [→](docs/EXPERIMENTS.md#e14) |
| E15 | fake-quant ↔ 정수 차이에 쌍 표준오차를 붙이면 | [→](docs/EXPERIMENTS.md#e15) |
| E16 | LayerNorm을 정수로 에뮬레이션하면 transformer 격차가 닫히는가 | [→](docs/EXPERIMENTS.md#e16) |
| E17 | 출력 텐서가 결과물인 과제(초해상)에서는 격차가 어떻게 보이는가 — conv 3개 vs 9개 | [→](docs/EXPERIMENTS.md#e17) |
| E18 | E7의 곱셈기·누산기·바이어스 폭도 시드 3개 × 10,000장으로 — 등급은 옮겨 가고 크기는 안 옮겨 감 | [→](docs/EXPERIMENTS.md#e18) |
| E19 | AdaRound의 이득은 비트 정확한 정수 실행에서도 남는가 — 8비트에서는 어느 쪽에도 없음 | [→](docs/EXPERIMENTS.md#e19) |
| E20 | ImageNet 규모 그래프·ImageNet 가중치(ResNet-50)에서도 같은 격차가 보이는가 | [→](docs/EXPERIMENTS.md#e20) |

</details>

## 더 읽을 것

| 문서 | 내용 |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 실험 E1–E20 전문과 생성된 표 |
| [장문 연구 원고](paper/npuloop_thesis.pdf) | 32쪽 개인 연구 원고 (게재된 논문이 아닙니다) |
| [docs/VALIDATION_SCOPE.md](docs/VALIDATION_SCOPE.md) | 무엇을 검증했고 무엇을 검증하지 않았는가 |
| [docs/DESIGN.md](docs/DESIGN.md) | 아키텍처, 모듈, 정수 데이터패스 정의, 저장소 구조, 검증 원칙 |
| [docs/INTEGER_DATAPATH.md](docs/INTEGER_DATAPATH.md) | 양자화 지점과 정수 데이터패스의 대응 |
| [docs/RELATED.md](docs/RELATED.md) | 선행 연구와 이 저장소의 위치 — **자기 주장을 정정한 기록 포함** |
| [docs/USAGE.md](docs/USAGE.md) | 설치·학습·CLI·실험 재현·파이썬 API |
| [docs/REPRODUCTION_REQUIREMENTS.md](docs/REPRODUCTION_REQUIREMENTS.md) | 재현에 필요한 자원과 시간 |
| [docs/INTEGRATION_REPORT.md](docs/INTEGRATION_REPORT.md) | 통합 검증 기록과 체크포인트 읽기 정책 |
| [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) · [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 화면 사용 안내 · 공개 사이트 게시 기준 |
| [docs/upstream/](docs/upstream/) | oneDNN 버그 보고서와 패치 |
| [paper/](paper/) | 장문판과 4쪽 단문판(영문·국문)의 상태와 빌드 방법 |

## 데이터와 외부 도구

데이터: CIFAR-10 (Krizhevsky, 2009), Imagenette (fast.ai).
SCALE-Sim은 E8, ethos-u-vela는 E13 검증 실험에서만 씁니다.
체크포인트는 `config`와 텐서 `state_dict` 형식만 `weights_only=True, map_location="cpu"`로 읽으며, 임의 객체 역직렬화로 재시도하지 않습니다([정책](docs/INTEGRATION_REPORT.md)).
