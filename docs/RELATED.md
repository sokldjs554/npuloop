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
| 비용 모델을 프루닝 루프 안에서 호출 | HALP/NetAdapt는 측정 LUT, Torch-Pruning은 HW 모델 없음, U-Boost(NAS)는 코드 없음, `channel pruning alignment hardware` 0건 | `prune.prune_cost_greedy`, E6 |
| NPU 친화 활성함수 교체 + 재학습의 정확도·사이클 트레이드오프 | `quantization-friendly training`, `hardware-friendly activation quantization replace` 0건; 얼굴인식 저장소 하나가 PReLU→LeakyReLU를 손으로 교체 | `quant.swap_activations` + healing, E4(b); LUT 활성함수의 이중 양자화 지점을 lint가 설명 |
| 캘리브레이션 세트 크기/구성 연구 (CV PTQ) | quantscope의 오염 테스트 정도 | E5 |
| 정수 구현 선택(반올림·곱셈기 비트·누산기 폭)의 정확도 비용 | deterministic-int8-llm-inference(LLM/GPU)만. TFLite는 `TFLITE_SINGLE_ROUNDING` 같은 선택지를 코드로 갖고 있지만 정확도 비용을 표로 낸 곳은 찾지 못함 | `RequantConfig`, E7 |
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

## 참고한 논문

* Nagel et al., *Data-Free Quantization through Weight Equalization and Bias Correction* (ICCV 2019) — CLE, BC
* Esser et al., *Learned Step Size Quantization* (ICLR 2020) — 학습 가능한 스케일(그래디언트 스케일링은 미구현, 실험에서는 미사용)
* Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (CVPR 2018) — 정수 requant 체계
* gemmlowp `FixedPoint` / TFLite `MultiplyByQuantizedMultiplier`, `TFLITE_SINGLE_ROUNDING`
* Samajdar et al., *SCALE-Sim* (ISPASS 2020) — weight-stationary 사이클 모델
* Liu et al., *Learning Efficient Convolutional Networks through Network Slimming* (ICCV 2017) — BN-γ 채널 중요도
* Shen et al., *HALP: Hardware-Aware Latency Pruning* (NeurIPS 2022) — 지연 기반 프루닝 (측정 LUT)
* Gupta & Akin, *Accelerator-aware Neural Network Design using AutoML* (2020) — EdgeTPU에서 depthwise가 systolic array를 못 채우는 문제
* Li et al., *MQBench: Towards Reproducible and Deployable Model Quantization Benchmark* (NeurIPS 2021 D&B) — fake-quant와 실제 백엔드의 배포 격차
* Yao et al., *HAWQ-V3: Dyadic Neural Network Quantization* (ICML 2021) — 정수 전용 추론, dyadic requant
* Kim et al., *I-BERT: Integer-only BERT Quantization* (ICML 2021) — 정수 softmax/GELU/LayerNorm
* Lin et al., *FQ-ViT* (IJCAI 2022) · Yuan et al., *PTQ4ViT* (ECCV 2022) · Li & Gu, *I-ViT* (ICCV 2023) — ViT의 PTQ와 정수 전용 추론
