# 관련 작업 조사 — "남들이 하지 않은 주제"인지 확인하기

프로젝트 주제를 정하기 전에 GitHub 저장소, 시뮬레이터/비용 모델, 논문, 한국어 포트폴리오를 네 갈래로 훑었습니다
(2026-09-06 기준, 검색 도구 기반이라 완전하지는 않습니다). 요지만 적습니다.

## 이미 아주 많은 것

* **CIFAR-10 압축 템플릿**: ResNet-18/MobileNetV2 + magnitude/채널 프루닝 + `torch.ao` PTQ/QAT (+KD) → 정확도/크기/지연 표.
  `pruning cifar10` 82개, `quantization cifar10` 34개, `quantization-aware-training` 토픽 115개 저장소. 대부분 과제·튜토리얼 복제.
* **PyTorch 양자화 튜토리얼 복제** (motokimura / leimao 스타일): FX graph mode PTQ/QAT, ONNX export, ORT 지연 측정, Streamlit.
* **논문 재현**: PACT, DoReFa, LSQ, AdaRound, BRECQ, DFQ(CLE+BC), HAWQ — 참조 구현은 2019–2023년, 새 개인 저장소는 과제 수준.
* **벤더 툴체인 배포 로그**: YOLO → ONNX → RKNN/QNN/TensorRT/TFLite/Hailo/Ethos-U (`rknn yolov8` 123개). 양자화 내부는 블랙박스.
* **from-scratch INT8 엔진**: 53개, 대부분 LLM(GPT-2/Qwen) 대상. FPGA 쪽 INT8 CNN 가속기(29개)는 MNIST + 골든모델 비트 일치가 표준.
* **한국어**: PTQ vs QAT 표(yugwangyeol), 벤더 SDK YOLO 배포 캡스톤, RTL systolic array 풀스택, 코디세이 부트캠프의 "mini-npu-simulator" 무리(~30개, MAC 패턴 매처 수준), LLM 양자화 해커톤.

## 비어 있던 자리 (이 프로젝트가 채우려는 것)

| 갭 | 조사 결과 | npuloop |
|---|---|---|
| fake-quant ↔ 정수 실행 불일치를 **코드(LSB) 단위로, 노드별 국소/전파로 분리**해 측정 | 처음 조사 때는 검색 0건이라고 적었으나 **정정**: MQBench(NeurIPS 2021 D&B)는 학술 fake-quant와 실제 백엔드(TensorRT·SNPE·TVM·ACL·FBGEMM)의 배포 격차를 벤치마크했고, PPQ(OpenPPL)는 레이어별·그래프별 양자화 오차 분석 도구를, HAWQ-V3는 TVM 배포까지 포함한 정수 전용 추론을 이미 제공합니다. 그들이 재는 것은 float↔양자화 오차 또는 백엔드 정확도 격차이고, 이 저장소가 재는 것은 **같은 스케일의 정수 코드가 몇 LSB 어긋나는지를 노드마다 teacher-forcing으로 분리한 값**입니다(E2/E7). KEA는 로짓 ±1 드리프트를 언급하는 정도 | `intengine.verify`: 국소(teacher-forced) vs 전파 불일치, `RequantConfig` ablation |
| torch.fx 그래프를 먹는 가벼운 순수 파이썬 NPU 비용 모델 | branes-ai/graphs(0 star, roofline CLI), Vela(TFLite/Ethos-U 전용), SCALE-Sim/Timeloop/MAESTRO/ZigZag는 오프라인 도구 | `npu.estimate` + SCALE-Sim 대조(E8) |
| 비용 모델을 프루닝 루프 안에서 호출 | **정정(2026-09-13)**: 처음에 "U-Boost는 코드 없음"이라고 적었으나 틀렸습니다. U-Boost NAS(Yüzügüler et al., ECCV 2022)는 배열 기반 가속기의 **미분 가능한 활용도 해석 모델**을 NAS 목적함수 안에 넣고 128×128 systolic array RTL로 검증했으며, 코드도 공개되어 있습니다(github.com/yuezuegu/UBoostNAS). 해석 모델을 탐색 루프에 넣는 계열은 그 뒤로도 이어집니다 — HASS(FPL 2024)는 하드웨어 성능·자원 추정을 비정형 희소성 탐색에, SASP(GLSVLSI 2025, arXiv 2411.10285)는 systolic 배열 크기에 맞춘 블록 희소성을 트랜스포머 FFN 공동 설계에 씁니다. HALP/NetAdapt/AMC는 여전히 **측정 LUT**(실리콘 필요), Torch-Pruning(DepGraph)은 MACs·파라미터만 봅니다. 남는 자리는 "해석 모델 + 의존성 그래프 위의 구조적 채널 프루닝 + 그 결정을 정수 실행 정확도까지 되돌려 확인"의 조합이지 "해석 모델을 루프에 넣은 최초"가 아닙니다 | `prune.prune_cost_greedy`, E6 |
| NPU 친화 활성함수 교체 + 재학습의 정확도·사이클 트레이드오프 | `quantization-friendly training`, `hardware-friendly activation quantization replace` 0건; 얼굴인식 저장소 하나가 PReLU→LeakyReLU를 손으로 교체 | `quant.swap_activations` + healing, E4(b); LUT 활성함수의 이중 양자화 지점을 lint가 설명 |
| 캘리브레이션 세트 크기/구성 연구 (CV PTQ) | quantscope의 오염 테스트 정도 | E5 |
| 정수 구현 선택(반올림·곱셈기 비트·누산기 폭)의 정확도 비용 | 축마다 사정이 다릅니다. **누산기 폭에는 선행 연구가 있습니다** — A2Q(Colbert et al., ICCV 2023)와 A2Q+(2024)는 목표 누산기 비트 폭에서 유도한 한계로 가중치 L1 노름을 제약해 오버플로를 원천 차단하며 Brevitas·FINN에 통합돼 있습니다. 다만 그쪽은 *좁은 누산기에 맞게 학습시키는* 문제를 풀고, 이 저장소는 *평범하게 학습된 모델을 그냥 좁혔을 때 무엇이 깨지는지*를 잽니다. **반올림 모드**는 선택지 자체가 널리 알려져 있고(gemmlowp 이중 반올림 vs `TFLITE_SINGLE_ROUNDING`, PPQ는 플랫폼별 반올림 정책을 인코딩) TFLite 이슈·PR로도 논의됐지만, 그 선택의 **정확도 비용을 표로 낸 것**은 찾지 못했습니다. **곱셈기 비트 폭** ablation은 어떤 규모에서도 찾지 못했습니다 | `RequantConfig`, E7 |
| 검증 문화: PyTorch 쪽 압축 코드에 비트 정확 골든모델 | FPGA 저장소에는 흔하지만 PyTorch 쪽 압축 저장소에는 거의 없음 | NumPy ⇄ C++ 비트 동일 테스트, gemmlowp 참조 대조 |

## 이 프로젝트에 가장 가까운 것들 (참고하고 차별화한 대상)

* **riskywindow/KEA** — 16×16 systolic MXU + DW 유닛 + 스크래치패드, 기능/사이클 근사 시뮬레이터, MLIR 컴파일러, MobileNetV2 int8 bit-exact. *압축 결정을 하드웨어에 맞춰 바꾸는 단계는 없음.*
* **akshatkumbhat/quantscope** — FX PTQ, 관측기 4종, W4A4 QAT, 256-config 혼합정밀도 sweep, 사전등록 실험·CI 신뢰구간·provenance 라벨. *하드웨어 모델·정수 엔진 없음, 토이 데이터.*
* **UllasP0707/Edge-Opt** — JSON 하드웨어 프로파일 + roofline, STE QAT, 프루닝. *합성 데모 수준, 배열 타일링 모델 없음.*
* **bccha/npu-from-scratch (한국어)** — fx tracing → C 코드 생성, 8×8 MAC 타일에 맞춘 INT8 패딩, MNIST, PyTorch–HW 일치. *MNIST, 비용 모델·프루닝·QAT 없음.*
* **Arm Vela** — 순수 파이썬 성능 추정기(`max(대역폭 사이클, 연산 사이클)`)의 산업 표본. *TFLite/Ethos-U 전용, 학습 루프와 무관.*

## 정정과 추가 조사 (2026-09-10)

처음 조사(2026-09-06)의 "검색 0건" 표현 중 두 곳은 과장이었습니다. 위 표에 정정을 적었고 요지는 다음과 같습니다.

* **MQBench**(Li et al., NeurIPS 2021 Datasets & Benchmarks) — 여러 PTQ/QAT 알고리즘을 실제 하드웨어 백엔드 설정으로 재현하고 fake-quant와 배포 결과의 격차를 벤치마크. **PPQ**(OpenPPL) — 그래프 스케줄링, 레이어별/그래프별 오차 분석, 여러 타깃 플랫폼 export. **HAWQ-V3**(Yao et al., ICML 2021) — dyadic 정수 전용 추론과 TVM 배포. 이 저장소의 차별점은 "가상 NPU의 비용 모델을 압축 결정 루프에 넣고, 정수 엔진 두 개를 모든 텐서에서 비트 일치시킨 뒤, 정수 구현 세부를 ablation한다"는 조합이지 "fake-quant와 정수 실행의 차이를 잰 최초"가 아닙니다.
* **정수 softmax/LayerNorm은 새 설계가 아닙니다.** I-BERT(Kim et al., ICML 2021)의 정수 sqrt·다항 근사, FQ-ViT(Lin et al., IJCAI 2022)의 log-int-softmax와 PTF LayerNorm, PTQ4ViT(Yuan et al., ECCV 2022), I-ViT(Li & Gu, ICCV 2023)의 Shiftmax·I-LayerNorm이 앞서 있습니다. 이 저장소의 정수 softmax(Q15 exp LUT + 정확한 정수 나눗셈)와 LayerNorm(int64 합, 정확한 정수 sqrt)은 그 계열의 단순한 변형이며, 기여는 그것을 NumPy와 C++에서 비트 일치시키고 fake-quant와의 국소 불일치(LayerNorm 24.9%)를 수치로 낸 것입니다.
* **LSQ**: `quant/fake.py`의 학습 가능한 스케일은 LSQ의 그래디언트 식(STE로 자연히 나오는 `round(x/s) − x/s`)은 따르지만 LSQ의 그래디언트 스케일 `1/√(N·Q_P)`는 구현하지 않았고, 실험(E4의 QAT)에서는 스케일을 고정한 채 학습했습니다. 따라서 이 저장소는 LSQ를 재현했다고 말하지 않습니다.
* 검색 기반 조사라 여전히 완전하지 않습니다. 비슷한 것을 알고 계시면 이슈로 알려 주시면 표를 고치겠습니다.

## 정정과 추가 조사 (2026-09-13)

지원 준비 과정에서 이 문서를 다시 검증했고, 사실 오류 하나와 누락 넷을 고쳤습니다.

* **U-Boost NAS는 코드가 있습니다.** 위 표의 "코드 없음"은 틀렸고, 하필 이 저장소의 중심 아이디어(해석적 활용도 모델을 탐색 루프에)에
  가장 가까운 선행 연구입니다. 차별점은 NAS가 아니라 구조적 프루닝이라는 것, 그리고 결정을 정수 실행 정확도까지 되돌려 확인한다는 것입니다.
* **Torch2Chip**(MLSys 2024) — 정수 엔진 쪽의 최근접 비교 대상이고 문제 의식이 이 저장소와 같습니다:
  현재 알고리즘들이 양자화된 정수를 중간 결과로만 취급하고 최종 출력은 "이산화된" 부동소수점이라, 하드웨어 설계자에게 정수 파라미터 추출과
  레이어 fusion 부담을 떠넘긴다는 것입니다. 커스텀 압축 + 자동 fusion + 정수 파라미터 추출 + 연산자별 정수 텐서 관측을 제공합니다.
  **없는 것**: 하드웨어 비용 모델, 비트 일치 대조용 두 번째 엔진, fake-quant와 정수 실행의 격차 수치.
* **HASS**(FPL 2024) — dataflow 가속기를 위한 하드웨어 인지 프루닝을 표방하며 하드웨어 성능·자원 추정을 희소성 탐색과 공동 최적화합니다.
  FPGA dataflow에 비정형 희소성이라 이 저장소(systolic, 구조적 채널)와는 다른 자리입니다.
* **SASP**(GLSVLSI 2025, arXiv 2411.10285) — 프루닝 블록 크기를 systolic 배열 차원에 맞춰 타일 단위로 건너뛰게 하고,
  (배열 크기 × 희소율) 공간을 함께 훑습니다. E6의 "정렬 프루닝" 아이디어에 가장 가까운 2025년 연구입니다.
* **A2Q / A2Q+**(ICCV 2023 / 2024) — 누산기 비트 폭의 정확도·자원 결과를 다루는 선행 연구. E7의 누산기 폭 축은 이 계열을 인용해야 합니다.

덧붙여 **E8의 표현을 낮췄습니다.** SCALE-Sim은 그 자체가 이상화 모델이고, 실제 TPU와의 대조는 2026년에야
(그것도 선형 상관까지) 확인되었으며 elementwise 연산에서는 따로 모델이 필요하다고 보고되었습니다.
따라서 0.5% 일치는 "닫힌 식을 올바로 구현했다"는 교차 확인이지 하드웨어 충실도의 증거가 아닙니다.
**E13**(Arm Vela 대조)을 추가한 이유가 이것입니다. 벤더가 실제 판매되는 칩을 위해 출하하는 추정기와 대면시키자
이 저장소의 모델이 체계적으로 낙관적이라는 것이 드러났습니다.

## 추가 조사 (2026-09-18) — 검색을 못 한 채로 적는 항목

**이 세션에서는 arxiv·Semantic Scholar·OpenReview·Crossref가 모두 프록시에서 막혔습니다.** 아래 세 항목은
따라서 **조사한 결과가 아니라 조사하지 못했다는 기록**입니다. 위 표의 항목들과 같은 근거를 갖고 있지 않습니다.

* **E19 (AdaRound를 정수 경로에서 재기).** AdaRound(ICML 2020)와 그 계열(BRECQ 등)이 보고하는 이득은 전부
  모의 양자화 그래프 위의 정확도입니다. 그 이득을 **비트 정확한 정수 프로그램에서 다시 재는** 것이 이 저장소가
  던질 수 있는 질문이고, 실제로 8비트에서는 어느 쪽에서도 이득이 없다는 음성 결과가 나왔습니다. 다만
  **이것이 남들이 하지 않은 일인지는 확인하지 못했습니다.** MQBench와 Torch2Chip은 실제 백엔드까지 배포하는
  계열이므로 그 안에서 이미 다뤄졌을 가능성이 충분히 있습니다. 면접이나 원고에서 이 항목을 "최초"로 말하면
  안 됩니다. 말할 수 있는 것은 "이 저장소에서는 두 경로를 같은 이미지에서 짝지어 쟀고 결과는 이렇다"까지입니다.
* **E20 (ImageNet 규모).** 이건 애초에 신규성 주장이 아니라 **적용 범위** 항목입니다. 기존 한계("CIFAR-10
  규모라 일반화를 말할 수 없다")를 없애기 위한 측정이지, 새로운 것을 했다는 주장이 아닙니다. 다만 E18의
  누산기 폭 결과가 K ≤ 1,280에서만 측정됐다는 단서는 이제 K = 4,608 그래프가 붙었습니다 — A2Q/A2Q+가 푸는
  문제(좁은 누산기에 맞게 **학습**)와는 여전히 다른 질문(평범하게 학습된 모델을 그냥 좁히면 무엇이 깨지는가)입니다.
* **max pooling 지원.** 신규성과 무관한 **기본 커버리지**입니다. 없었다는 것이 오히려 이상한 쪽이고, 채운 것이
  자랑거리는 아닙니다. 기록해 둘 값어치가 있는 부분은 검증 방식입니다 — 패딩 의미론을 논증으로 정당화하지 않고
  TFLite의 참조 커널과 코드 단위로 대조했습니다(`tests/test_tflite_maxpool.py`).

## 참고한 논문

* Nagel et al., *Data-Free Quantization through Weight Equalization and Bias Correction* (ICCV 2019) — CLE, BC
* Esser et al., *Learned Step Size Quantization* (ICLR 2020) — 학습 가능한 스케일(그래디언트 스케일링은 미구현, 실험에서는 미사용)
* Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (CVPR 2018) — 정수 requant 체계
* gemmlowp `FixedPoint` / TFLite `MultiplyByQuantizedMultiplier`, `TFLITE_SINGLE_ROUNDING`
* Samajdar et al., *SCALE-Sim* (ISPASS 2020) — weight-stationary 사이클 모델
* Liu et al., *Learning Efficient Convolutional Networks through Network Slimming* (ICCV 2017) — BN-γ 채널 중요도
* Shen et al., *HALP: Hardware-Aware Latency Pruning* (NeurIPS 2022) — 지연 기반 프루닝 (측정 LUT)
* Gupta & Akin, *Accelerator-aware Neural Network Design using AutoML* (2020) — EdgeTPU에서 depthwise가 systolic array를 못 채우는 문제
* Yüzügüler et al., *U-Boost NAS: Utilization-Boosted Differentiable Neural Architecture Search* (ECCV 2022) — 해석적 활용도 모델을 탐색 루프에 (코드 공개)
* Colbert et al., *A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance* (ICCV 2023), *A2Q+* (2024) — 누산기 폭
* Nagel et al., *Up or Down? Adaptive Rounding for Post-Training Quantization* (ICML 2020) — AdaRound (E19에서 구현·측정)
* Li et al., *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction* (ICLR 2021) — 블록 단위 재구성 (미구현)
* *Torch2Chip: An End-to-end Customizable Deep Neural Network Compression and Deployment Toolkit* (MLSys 2024)
* *HASS: Hardware-Aware Sparsity Search for Dataflow DNN Accelerator* (FPL 2024)
* Palacios et al., *Systolic Arrays and Structured Pruning Co-design for Efficient Transformers in Edge Systems* (GLSVLSI 2025)
* Li et al., *MQBench: Towards Reproducible and Deployable Model Quantization Benchmark* (NeurIPS 2021 D&B) — fake-quant와 실제 백엔드의 배포 격차
* Yao et al., *HAWQ-V3: Dyadic Neural Network Quantization* (ICML 2021) — 정수 전용 추론, dyadic requant
* Kim et al., *I-BERT: Integer-only BERT Quantization* (ICML 2021) — 정수 softmax/GELU/LayerNorm
* Lin et al., *FQ-ViT* (IJCAI 2022) · Yuan et al., *PTQ4ViT* (ECCV 2022) · Li & Gu, *I-ViT* (ICCV 2023) — ViT의 PTQ와 정수 전용 추론
