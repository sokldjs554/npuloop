# 로컬 모델 변경·학습·압축 실험

`python -m npuloop.study_runner`는 저장된 분류 모델을 실제로 변경하고, 학습한 뒤,
FP32·fake quantization·정수 엔진의 정확도를 같은 테스트 이미지에서 비교한다.
브라우저에서 버튼을 누르는 것만으로 학습한 것으로 표시하지 않는다. CPU에서 명령을 실행한
결과가 `study.json`에 기록되며, 이 파일을 데모의 로컬 결과 가져오기에 사용할 수 있다.

## 바로 실행하기

저장소 루트에서 Python 3.10 이상과 CPU PyTorch를 설치한 환경을 사용한다.

```bash
python -m pip install -e .
python -m npuloop.study_runner \
  --checkpoint examples/quickstart/resnet20_relu.pt \
  --data examples/quickstart/cifar10_sample.npz \
  --out runs/local-prune-study \
  --preset edge-10tops \
  --operation prune --prune-ratio 0.25 \
  --epochs 1 --steps-per-epoch 2 \
  --eval-images 32 --calibration-images 32 \
  --seed 0 --threads 2
```

새 디렉터리 또는 비어 있는 디렉터리만 허용한다. 기존 결과가 있으면 실행을 거부하므로,
다시 실행할 때는 `--out runs/local-prune-study-2`처럼 새 이름을 지정한다.
원본 체크포인트와 데이터는 읽기만 하고, `results/*.json`의 과거 실험을 수정하지 않는다.

이 명령은 **번들 진단 샘플 실험**이다. CIFAR-10 전체 벤치마크나 GPU/NPU 실측이 아니다.
`--prune-ratio 0.25`는 발견한 가지치기 그룹의 채널을 약 25% 제거하라는 뜻이다.
기존 `prune()` API의 유지 비율로는 `0.75`에 해당하며, 채널 정수 반올림과 최소 채널 수
제약 때문에 전체 파라미터 감소율은 25%와 다를 수 있다.

## 실제로 수행하는 과정

1. `weights_only=True`, CPU 모드로 체크포인트를 읽고 모델·입력 파일의 SHA-256을 계산한다.
2. 고정된 테스트 부분집합에서 변경 전 FP32 정확도를 측정한다.
3. 그래프 기반 구조적 채널 가지치기 또는 activation 교체를 수행한다.
4. 학습 전 변경 모델을 저장하고 같은 테스트 이미지에서 즉시 측정한다. 이 단계가
   학습 효과를 비교하는 **0-step 기준점**이다.
5. 기존 `fit()`으로 TRAIN 분할에서 SGD 학습을 수행한다. 매 epoch의 loss, train/validation
   accuracy, 학습률, 실제 step 수, 실제 학습 장수를 기록한다. validation 정확도가 가장 높은
   epoch를 선택하고, 동률이면 나중 epoch를 선택한다. test 정확도는 선택에 사용하지 않는다.
6. 선택 모델을 TRAIN 이미지로만 보정하고 `npu-default` INT8 fake quantization을 평가한다.
7. 실제 정수 그래프를 내보내고 NumPy 정수 의미론 엔진으로 같은 테스트 이미지를 실행한다.
8. 변경 전·후 구조에 대해 가상 NPU cycle 추정치를 별도로 저장한다.

학습 1–4 step의 짧은 실험은 일정한 학습률을 쓴다. 2-step OneCycle 스케줄의 0분모 문제를
피하고 실제 optimizer update를 수행하도록 공용 trainer를 수정했다. 5 step 이상은 기존
OneCycle 스케줄을 사용한다. `channels_last=False`로 CPU 학습을 실행한다.

정수 엔진은 정수 곱셈·누산·requantization 의미론을 CPU에서 실행한다. 누산 커널 일부는
정확한 정수 범위 안의 float32/float64 연산을 사용한다. 따라서 이 결과를 전용 INT8 하드웨어의
처리량이나 NPU 지연시간 측정으로 해석하면 안 된다. `elapsed_seconds` 역시 전체 로컬
실행의 벽시계 시간이며 모델 추론 지연시간 벤치마크가 아니다.

## 데이터 분리와 정확한 분모

입력 NPZ는 `x_train`, `y_train`, `x_test`, `y_test`, `classes`를 제공해야 한다.
이미지는 `(N,H,W,3)`의 정사각형 uint8 RGB이며, label은 0부터 시작하는 정수다.
모든 클래스에 최소 두 개의 학습 이미지가 필요하다.

기존 양수 `val_per_class`가 있으면 그 값을 사용한다. 값이 없거나 0이면 TRAIN 원본에서
클래스별 `min(500, max(1, 최소 클래스 장수 // 10))`장을 validation으로 분리한다.
분할 seed는 공용 데이터 로더의 `2024`로 고정되어 있다.

번들 파일은 TRAIN 원본 512장, TEST 500장이고 `val_per_class=0`이다. 실행기는 TRAIN에서
클래스당 4장씩 분리하므로 학습 472장·검증 40장·테스트 500장이 된다. 보정 이미지는
검증 이미지를 뺀 472장에서만 추출한다. 이 새 검증 분할은 이번 fine-tuning의 선택용이다.
원래 사전학습 체크포인트가 어떤 이미지로 학습되었는지는 이 실행기가 재검증하지 않는다.

`--eval-images 32`는 **최대 32장을 정확히** 평가한다. 배치 크기로 올림하지 않는다.
테스트와 검증 cap에 같은 옵션을 사용하지만 각각의 실제 분모를 기록한다. 예를 들어
`--eval-images 64`이면 번들 test는 64장, validation은 40장이다. 보정 요청 장수가 학습
분할보다 크면 사용 가능한 학습 장수로 제한하고 실제 값을 기록한다. 보정/테스트 부분집합에
같은 이미지 내용이 중복된 입력은 거부한다.

테스트 순서는 데이터 로더의 고정 permutation seed `1234`를 따른다. 모든 단계는 같은
테스트 인덱스를 사용하며, 인덱스·이미지·정답으로 만든 `subset_sha256`을 공유한다.
데이터 전체가 사용자 제공 NPZ이므로, 이미지 출처 자체를 공식 CIFAR 자료로 인증하지 않는다.

## activation 교체와 0-epoch 대조군

```bash
python -m npuloop.study_runner \
  --checkpoint examples/quickstart/resnet20_relu.pt \
  --data examples/quickstart/cifar10_sample.npz \
  --out runs/local-activation-study \
  --operation activation --activation relu6 \
  --epochs 1 --steps-per-epoch 2 \
  --eval-images 32 --calibration-images 32 --seed 0 --threads 2
```

`--epochs 0`이면 모델 변경, 변경 직후 평가, 보정, fake-quant 평가, 정수 평가를 수행하고
학습은 생략한다. `training.actual_steps=0`, `training.epochs=[]`이며, 변경 직후와
`fine_tuned_fp32`의 가중치 hash가 같다. 같은 변경·seed에 서로 다른 출력 디렉터리를 지정해
0-epoch와 학습 실행을 비교할 수 있다. ReLU → ReLU처럼 실질적 변경이 없는 요청은 거부한다.

추가 옵션은 `--batch-size`(기본 32), `--lr`(기본 0.01)이다. 실제 사용 가능한 완전한
학습 배치 수가 `--steps-per-epoch`보다 작으면 그 수로 제한한다. 마지막 불완전 배치는
공용 데이터 로더의 정책에 따라 제외하며, 처리한 장수를 로그에 남긴다.

## 결과 파일과 JSON 계약

| 파일/키 | 의미 |
|---|---|
| `study.json` | 전체 결과. `schema_version = "npuloop.model-study.v1"`, 성공 시에만 `status = "complete"` |
| `config.json` | 요청 옵션과 실제 평가/보정/검증/배치/step cap |
| `changed_before_training.pt` | 모델 변경 직후의 FP32 대조군 |
| `fine_tuned.pt` | 검증으로 선택한 새 FP32 체크포인트. 0-epoch이면 변경 직후 모델 |
| `training_log.json` | 실제 학습 기록, 선택 epoch, 전체 step/장수, 가중치 변경 여부 |
| `training/` | 공용 trainer의 `best.pt`, `last.pt`, `state.pt`, `log.json`; 학습할 때만 생성 |
| `model.npuloop` | 실행 가능한 정수 그래프와 보정 provenance |
| `source_checkpoint` | 원본 파일 hash 및 원본 tensor state hash |
| `software` | Python/PyTorch/NumPy 버전, git commit, `git_worktree_dirty`, 실행 소스별 SHA-256 및 전체 source hash |
| `dataset` | 실제 분할 장수, 원본 NPZ 인덱스, label, 정규화, 부분집합 hash |
| `operation` | 가지치기 전후 채널 수 또는 교체 activation 수, 모델 config, 파라미터 수 |
| `observed` | 실제 CPU 실행으로 측정한 다섯 단계의 결과 |
| `simulated_costs` | `provenance="simulated"`, 가상 preset과 변경 전후 `total_cycles` 등 |
| `comparison` | 학습 회복 정확도 차이(%p), 정수/기준 정확도 차이, fake/정수 top-1 일치율, 추정 cycle 감소율 |
| `artifacts` | 생성 파일별 상대 경로, 크기, SHA-256. 자기참조를 피하기 위해 `study.json` 자체는 제외 |

`observed`의 단계 이름은 `baseline_fp32`, `changed_fp32_before_training`,
`fine_tuned_fp32`, `fake_quant`, `integer`다. 각 단계는 `accuracy`(0–1), `correct`,
`n`, `split="test"`, `subset_sha256`, `model_state_sha256`, `predictions`,
`execution`, `provenance="measured_cpu"`를 제공한다. 정수 단계의 `model_state_sha256`은
실행한 `.npuloop` 파일의 hash이고, 나머지는 해당 PyTorch state의 hash다. activation 구조는
`operation`의 config에 별도로 기록한다. JSON은 NaN/Infinity를 허용하지 않는다.
`git_worktree_dirty=true`이면 실행 당시 수정 중인 소스가 있으므로 commit만으로 실행 코드를
특정할 수 없다. 함께 기록한 소스별 hash로 사용한 Python 파일을 확인한다.

`simulated_costs.baseline.total_cycles`와 `simulated_costs.changed.total_cycles`는
**같은 가상 INT8 NPU에서 두 구조를 실행할 때의 모델 추정**이다. baseline FP32 CPU 시간과
INT8 NPU 시간의 실측 비교가 아니다. 파라미터가 줄어도 배열 타일 경계 때문에 cycle은 그대로일
수 있으며, 정확도가 학습 뒤 반드시 회복된다는 보장은 없다.

2026-09-16 CPU 검증에서 위 가지치기 명령은 고정 테스트 32장에 대해
`29/32 → 22/32 → 24/32 → 24/32 → 24/32`를 기록했다. 실제 학습은 2 step·64장이었고,
cycle 추정은 `34,417 → 30,025`였다. 이는 실행 경로 검증용 샘플 결과이며, 전체 데이터 성능이나
학습 수렴을 주장하는 수치가 아니다. 각 환경과 seed의 결과는 생성된 파일을 기준으로 읽는다.

검증 명령:

```bash
python -m pytest tests/test_study_runner.py -q
```

테스트는 실제 작은 모델의 학습 및 정수 추론, 원본 보존, 출력 덮어쓰기 거부, 0-epoch 대조군,
검증/보정 분리, 정확한 1장·7장·전체 초과 cap, 2-step optimizer update를 검사한다.
