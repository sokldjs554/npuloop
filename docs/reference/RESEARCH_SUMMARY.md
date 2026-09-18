# npuloop — 모델 변경·재학습·경량화와 정수 실행 검증

윤기혁 · 개인 연구·개발 프로젝트 · 모빌린트 Deep Learning Research Engineer 지원용 프로젝트 설명

## 해결하려던 문제

같은 모델 변경도 대상 NPU의 지원 연산에 따라 효과가 달라집니다. 지원하지 않는 활성함수를 교체하면 실행 비용을 줄일 수 있지만 정확도를 잃을 수 있고, 채널을 줄여도 배열 활용률 때문에 연산량과 실행 비용이 같은 비율로 감소하지 않을 수 있습니다. npuloop에서는 가상 NPU 조건을 바꿔 모델 변경 후보를 비교하고, 실제 재학습·양자화 결과의 정확도를 함께 확인했습니다.

모델 변경과 학습은 PyTorch 코드와 실험 스크립트로 수행합니다. 통합 [실험 실행기](STUDY_RUNNER.md)는 체크포인트·데이터를 받아 구조 수정 직후, 회복 학습 후, PTQ·정수 경로를 동일 시험 이미지로 평가하고 출처를 저장합니다. 브라우저 Workbench의 첫 화면에서는 모델 실험·학습 곡선·경량화 후보를 비교하고 실행기의 결과 JSON을 가져올 수 있습니다. 실제 모빌린트 칩이나 SDK에서 측정한 성능은 포함하지 않습니다.

## 모델 설계·학습에서 확인한 결과

다음 수치는 저장소의 기존 E1–E20 실험 결과입니다. 새 실행기의 번들 샘플 학습 검증은 별도 기록이며 이 표의 결과를 대체하지 않습니다.

| 실험 | 문제와 개입 | 결과와 판단 |
|---|---|---|
| **E4: 활성함수 변경 후 정확도 회복** | ResNet-20 SiLU→ReLU 교체 후 3 epoch 재학습 | FP32 90.34% → 교체 직후 40.36% → 재학습 후 89.74%. 정수 정확도는 기존 90.26% → 변경·재학습 후 89.81%. 지원 연산에 맞추는 과정에서 발생한 손실과 학습으로 회복한 범위를 함께 확인 |
| **E6: 구조적 프루닝과 비용 비교** | 균일·배열 정렬·비용 기반 채널 선택 및 3 epoch 미세조정 | ResNet-20 균일 0.5 조건: MACs 40,813,184 → 20,759,168, edge 추정 사이클 34,417 → 27,739, FP32 89.59% → 87.22%. 연산량 감소만으로 속도 향상을 주장하지 않고 정확도 손실도 비교 |
| **E9: NPU 지원 조건에 따른 선택** | ViT GELU→ReLU 교체 후 3 epoch 재학습 | FP32 80.98% → 80.95%. strict 추정 사이클 2,503,878 → 1,708,230; LUT 지원 조건에서는 52,246 → 52,054. LayerNorm·softmax 폴백이 남아 활성함수 변경만으로 strict 조건을 해결하지 못함 |

원본: [E4](https://github.com/sokldjs554/npuloop/blob/master/results/e4_surgery.json), [E6](https://github.com/sokldjs554/npuloop/blob/master/results/e6_pruning.json), [E9](https://github.com/sokldjs554/npuloop/blob/master/results/e9_customer_intake.json).
구현: [활성함수 변경·QAT](https://github.com/sokldjs554/npuloop/blob/master/experiments/e4_surgery.py), [프루닝·미세조정](https://github.com/sokldjs554/npuloop/blob/master/experiments/e6_pruning.py), [ViT 변경·재학습](https://github.com/sokldjs554/npuloop/blob/master/experiments/e9_customer_intake.py).

E4·E6·E9 사이의 수치를 하나의 전후 비교로 합치지 않습니다. E4의 정수 평가는 10,000장입니다. E6의 `int8_acc`는 모의 양자화 정확도이며 위 표는 FP32 `ft_acc`를 사용했습니다. E9는 FP32가 전체 테스트셋, 정수 정확도가 2,000장 부분집합이므로 해당 두 값의 차이를 곧바로 양자화 손실로 계산하지 않습니다.

PTQ·QAT·보정도 효과를 검증하는 대상으로 다뤘습니다. E4의 SiLU 모델은 PTQ 정수 정확도 90.26%에서 2 epoch QAT 후 90.01%로 낮아졌습니다. 따라서 QAT를 적용했다는 이유만으로 개선됐다고 설명하지 않습니다. E6에서도 배열 정렬이 모든 정확도·비용 조건에서 우월하다고 결론 내리지 않습니다.

## 데이터부터 평가까지 구현한 과정

1. 데이터 준비·분할·전처리: CIFAR-10 및 Imagenette 준비 도구와 데이터 로더를 구성했습니다.
2. 모델 학습: PyTorch 학습기에서 학습·검증 곡선, 설정과 체크포인트를 저장합니다. 기본 학습기는 검증셋 정확도로 체크포인트를 선택하고 시험셋을 평가합니다. E4·E6·E9의 복구 학습은 정해진 epoch 후의 모델을 비교합니다.
3. 구조 분석·변경: `torch.fx` 추적과 BatchNorm folding으로 그래프를 만들고, 활성함수 교체·프루닝 후보를 가상 NPU 조건별로 평가합니다.
4. 양자화·정수 export: 보정 데이터를 이용한 PTQ, QAT, 보정 실험을 수행하고 int8 가중치·int32 바이어스·재양자화 파라미터를 내보냅니다.
5. 결과 검증: 정확도, 가상 비용, 계층별 정수 출력 차이를 비교합니다. NumPy·C++ 실행 경로와 파일 export 후 독립 C++ 실행기를 대조합니다.

근거: [데이터 준비](https://github.com/sokldjs554/npuloop/blob/master/tools/prepare_cifar10.py), [Imagenette 준비](https://github.com/sokldjs554/npuloop/blob/master/tools/prepare_imagenette.py), [학습기](https://github.com/sokldjs554/npuloop/blob/master/npuloop/zoo/train.py), [실험 공통 코드](https://github.com/sokldjs554/npuloop/blob/master/experiments/common.py), [사용 방법](USAGE.md). 원래 학습 반복은 가중치 초기화 시드가 완전히 고정되지 않았으며, 현재 시드 처리가 과거 가중치의 동일 재생성을 보장하지는 않습니다.

## 모델 결과를 신뢰하기 위해 조사한 정수 실행의 차이

E15에서 분류 모델·데이터셋 구성 6종과 양자화 방식 2종의 전체 테스트 평가를 비교했습니다. 관측 정확도 차이는 최대 0.20%p였지만 출력 정수 코드의 28.5–72.7%는 달랐습니다. 정확도 유사성으로 출력의 비트 일치를 보장할 수 없다는 결과입니다. 신뢰구간의 0 포함은 정확도 동등성의 입증이 아닙니다. **근거: 원고 표 5.2.**

E16에서 LayerNorm만 정수 연산과 맞추자 pc 조건의 국소 차이는 24.77%에서 0%가 됐지만 출력 코드 불일치는 72.7%에서 65.7%로만 줄었습니다. LayerNorm 단독 개입의 한계를 확인한 결과이며 모든 연산자의 누적 교체가 불가능하다는 뜻은 아닙니다. **근거: 원고 표 5.3.**

E17에서는 초해상 입력의 float64·float32 나눗셈과 반올림 경계 차이를 입력 양자화기 결함으로 추적했습니다. 틀리는 입력과 수정 경로를 회귀 테스트로 확인했습니다. **근거: 원고 6.2절 및 [입력 양자화 테스트](https://github.com/sokldjs554/npuloop/blob/master/tests/test_intengine.py).**

## 지원 공고와의 연결

기준은 사용자가 제공한 **모빌린트 [AI반도체] Deep Learning Research Engineer 신입 공고**입니다. 근무지는 서울 강남구, 표시된 마감일은 **2026-09-30**, **최종 성적증명서 제출은 필수**입니다. 아래는 해당 원문의 요구사항과 프로젝트 근거를 대조한 것이며, 다른 경력 공고의 조건을 적용하지 않습니다.

| 요구사항 | 프로젝트 근거 | 현재 설명 가능한 범위 |
|---|---|---|
| Task·Dataset·Size·NPU별 모델 연구·설계 | E4·E6·E9 모델 변경, E12 Imagenette, E17 초해상, 가상 프리셋 비교 | 일부 과제·데이터·크기·가상 조건에 대한 실험. 모든 축을 교차한 체계적 탐색이나 실칩 최적화는 아님 |
| Quantization·Pruning 연구·적용 | E2 PTQ, E4 QAT·보정, E6 구조적 프루닝 | 적용 코드와 정확도·비용 비교를 제시. 새 알고리즘 발명이나 일률적 개선은 주장하지 않음 |
| CV 및 최신 딥러닝 연구 분석 | CNN·ViT·ESPCN 실험, [선행 연구 정리](https://github.com/sokldjs554/npuloop/blob/master/docs/RELATED.md), E15–E17 원인 분석 | 공개 구현·선행 연구를 비교한 근거. 계속 갱신되는 최신 연구를 모두 검토했다는 주장은 하지 않음 |
| 고객사 데이터 분석·모델 학습 | ViT·Inception을 받는 가상 고객 E9, 공개 데이터셋 학습 | 실제 고객 데이터 분석·업무 수행 경력은 아님 |
| 학습 관련 툴 개발 | 공통 학습기·체크포인트·구조 수정·회복 학습·동일 부분집합 평가 실행기, 결과 비교 UI | 실행 가능한 명령, 학습 로그, 체크포인트·데이터·코드 해시와 회귀 테스트로 확인 가능 |
| PyTorch/TensorFlow 등 프레임워크 | PyTorch 학습·변경·양자화 및 torch.fx 그래프 처리 | 공고가 두 프레임워크 모두를 필수로 요구하는 것은 아님 |
| Python·C/C++ 등 프로그래밍 | Python 연구 코드와 C++ 정수 실행기 | 실제 구현·검증 범위에 맞춰 설명 |
| 모델 개발 전체 과정 | 데이터 준비→학습→변경·경량화→export→평가 | 전체 과정 코드와 기존 기록이 있음. 모든 과거 실험의 독립 재현 완료와는 구분 |
| 우대: 경량화·CV·핵심 연산·NPU 이해·논문 실적 | 경량화 실험, 정수 연산·배열 비용 분석, 개인 연구 원고 | NPU 이해는 가상 제약 연구로 제시. 개인 원고를 게재·채택 실적으로 쓰지 않음 |

## 설명 순서와 남은 보완

면접과 README에서는 **E4의 변경·학습 → E6의 경량화와 정확도 비용 → E9의 NPU 조건별 판단 → E15–E17의 검증·원인 분석** 순서로 설명합니다. 정수 엔진은 모델 연구 결과의 신뢰도를 확인하기 위해 만든 도구로 연결합니다.

추가 실험을 할 때 우선할 것은 핵심 모델 변경 실험의 원본 체크포인트·설정·보정 조건 확보와 같은 조건의 재평가입니다. 이후 E4·E6·E9의 반복 학습으로 정확도와 비용의 균형을 확인할 수 있습니다. 기존 E14의 반올림 반복 실험이 이 비교를 대신하지는 않습니다. 외부 커널 대조 확대와 실제 NPU 측정은 추가 검증이며, 공고에 없는 지원 필수 조건으로 만들지 않습니다.

전체 원본 체크포인트와 과거 환경을 모두 확보해 전 실험을 독립 재평가한 상태는 아닙니다. [재현 자료](REPRODUCTION_REQUIREMENTS.md)와 [검증 범위](VALIDATION_SCOPE.md)에 확인된 범위를 구분했습니다.

공개 Workbench와 현재 개인 연구 원고는 [GitHub Pages](https://sokldjs554.github.io/npuloop/)에서 제공합니다. 초기 로컬 통합 기록은 [INTEGRATION_REPORT.md](https://github.com/sokldjs554/npuloop/blob/master/docs/INTEGRATION_REPORT.md), 이후 게시 단계의 기록은 [DEPLOYMENT.md](https://github.com/sokldjs554/npuloop/blob/master/docs/DEPLOYMENT.md)와 `verification/publish/`를 기준으로 읽습니다. 서로 다른 단계의 테스트 수를 이번 문서 수정의 재실행 결과로 합치지 않습니다.
