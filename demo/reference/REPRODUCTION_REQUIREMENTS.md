# 재현 자료와 정확한 재측정

**표 재생성**, **같은 원본 가중치의 재평가**, **처음부터 재학습**은 서로 다른 작업이다. 생성기가 결과 JSON으로 같은 표를 만든다고 해서 학습·평가 결과를 독립 재현한 것은 아니다.

## 현재 통합 실행기로 새 실험하기

[STUDY_RUNNER.md](STUDY_RUNNER.md)의 명령은 번들 또는 사용자 체크포인트를 실제 수정·재학습하고, FP32·모의 양자화·정수 경로를 같은 시험 부분집합으로 평가합니다. 설정, 훈련/검증/보정/시험 인덱스, 모델·데이터·소스 코드 해시, epoch 로그와 출력 체크포인트를 보존합니다. 브라우저는 `study.json`을 가져와 비교합니다.

이는 **새 실행의 재현 경로**입니다. 번들 샘플 32장·2 step 실행은 통합 기능 확인용이며 아래 원본 체크포인트의 누락을 해결하거나 E1–E17을 다시 측정했다는 뜻이 아닙니다. 학습·양자화·정수 평가의 이미지 제한은 배치 중간에서도 정확히 지키며, 짧은 학습도 실제 파라미터 갱신을 수행합니다.

## 확보할 원본

CIFAR-10의 ResNet-20 ReLU/SiLU, MobileNetV2-0.5, ViT-128/6, Inception-32와 Imagenette ResNet-20 및 ESPCN에 사용한 원래 체크포인트가 필요하다. E14에 사용한 추가 학습 반복의 체크포인트도 별도로 보존한다. `experiments/common.py`에서 확인되는 기본 학습 디렉터리는 `runs/<run>/best.pt`이며 `NPULOOP_RUNS`로 변경할 수 있다. **빠른 실행용 두 체크포인트가 모든 논문 실험의 원본을 대신하지는 않는다.**

각 원본에 파일 SHA-256, 모델 설정, 학습 로그, 사용한 commit, Python/PyTorch/NumPy 및 C++ 컴파일러 버전, 실제 실행 명령, 스레드 설정을 함께 보존한다. 존재하는 파일의 해시가 계산됐다는 사실만으로 그 파일이 과거 논문에 쓰인 원본임이 증명되지는 않는다. 원래 로그와의 대조가 필요하다.

데이터셋 버전·NPZ 체크섬, 학습/검증/시험 분할 인덱스, 전처리, 캘리브레이션 인덱스·시드·개수도 보존한다. 원래 학습기는 가중치 초기화를 완전히 고정하지 못했으므로 현재의 수정된 학습기로 재학습한 결과가 과거 표의 소수점 수치를 그대로 복구한다는 보장은 없다.

## 사용 가능한 검사

```bash
# 저장된 결과에서 논문 표를 다시 만들어 비교한다. 원본 데이터를 새로 평가하지 않는다.
python tools/submission_quality.py --regenerate --report paper-quality-report.json

# 파일 존재 여부와 실제 해시만 기록한다. 없는 파일은 missing/null, 종료 코드는 실패다.
python tools/submission_quality.py --inventory \
  examples/quickstart/resnet20_relu.pt examples/quickstart/cust_inception.pt \
  runs/resnet20_relu/best.pt runs/resnet20_silu/best.pt \
  runs/mnv2_050_relu6/best.pt runs/cust_vit/best.pt runs/cust_inception/best.pt \
  --report artifact-inventory.json
```

Imagenette와 ESPCN 및 E14 추가 반복은 실제 보관 경로를 확인해 `--inventory` 인수에 추가한다. 없는 원본의 경로나 해시를 추측해서 채우지 않는다.

## 2026-09-16 통합 검증 후 상태

| 확인 항목 | 현재 판정 |
|---|---|
| 빠른 실행용 두 체크포인트와 샘플의 저장소 등록 | 업로드된 원본에서 복사·해시 확인·quickstart 및 정수 실행 경로 대조 완료 |
| 논문 전체 원본 체크포인트 | 모두 확보됐다는 근거 없음 |
| 과거 초기화 시드까지 동일한 재학습 | 기존 원고가 한계를 명시함 |
| E14·E16·E17 구현·기존 결과 | 존재함; 미구현 항목으로 분류하지 않음 |
| 전체 실험의 독립 재평가 | 이번 작업에서 수행하지 않음 |

누락 자료를 공개할 수 없다면 그 한계를 적고, 재현 가능 범위를 빠른 실행과 공개 결과의 집계 검사로 제한해 설명한다. 실제 존재하지 않는 재현 완료 상태를 만들지 않는다.

로컬 통합 실행의 로그와 명령은 [INTEGRATION_REPORT.md](https://github.com/sokldjs554/npuloop/blob/master/docs/INTEGRATION_REPORT.md)에 있다. 전체 실험 원본 확보는 추가 연구 재측정을 위한 별도 작업이며, 번들 데모·공개 결과 집계의 실행 성공과 구분한다.
