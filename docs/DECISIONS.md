# npuloop 설계 판단 기록

이 문서는 결과를 요약하는 README가 아니라, **왜 이런 실험과 구현을 선택했는지**를 남긴 기술 판단 기록입니다.

## 1. 왜 fake-quant 정확도만 보지 않았나

PyTorch의 fake-quant는 양자화 오차를 모사하기에는 편하지만, 실제 정수 실행의 requantization·rounding·accumulator 동작을 그대로 보장하지 않습니다.

그래서 이 프로젝트에서는:

1. PyTorch 모델을 양자화하고
2. 정수 그래프로 export한 뒤
3. NumPy 정수 엔진과 C++ 정수 엔진에서 실행하고
4. TFLite reference 경로와 중간 tensor까지 대조했습니다.

핵심 질문은 **"top-1이 비슷한가?"와 "정수 코드가 같은가?"를 같은 질문으로 취급해도 되는가**였습니다. E15에서 두 값이 다르게 움직이는 것을 확인했고, E16에서 LayerNorm을 직접 개입해 원인을 더 좁혔습니다.

코드 시작점:
- `npuloop/intengine/requant.py`
- `npuloop/intengine/numpy_engine.py`
- `npuloop/intengine/cpp_engine.py`
- `npuloop/intengine/verify.py`

## 2. 왜 FLOPs가 아니라 비용 모델을 pruning 루프에 넣었나

채널을 절반으로 줄여도 실제 실행 비용이 절반이 된다는 보장은 없습니다. 배열 크기, channel alignment, depthwise 처리, 메모리 트래픽, host fallback에 따라 감소 폭이 달라집니다.

그래서 structured pruning은 단순 MAC 감소가 아니라 **변경 전후의 가상 NPU estimated cycles**를 함께 봅니다.

E6에서 MACs는 약 49% 줄었지만 estimated cycles 감소는 약 19%였습니다. 이 부정 결과 때문에 "MAC 감소 = NPU 비용 감소"라는 가정을 버렸습니다.

코드 시작점:
- `npuloop/prune/structured.py`
- `npuloop/npu/cost.py`
- `npuloop/npu/spec.py`

## 3. 왜 pruning을 graph 기반으로 구현했나

모델별 module 이름을 하드코딩하면 ResNet에서는 동작해도 Inception·MobileNet·Transformer 구조에서 쉽게 깨집니다.

그래서 `torch.fx` 기반 정적 그래프에서 producer → pass-through op → consumer를 추적해 **같이 잘라야 하는 channel group**을 찾도록 했습니다.

반대로 residual add, LayerNorm, attention matmul처럼 안전하게 자르기 어려운 경로에 닿으면 pruning 후보에서 제외합니다. 범용성을 늘리기 위해 "어떤 모델인가"보다 "채널이 어디로 흐르는가"를 기준으로 판단했습니다.

## 4. 왜 SCALE-Sim과 Arm Vela를 둘 다 비교했나

두 비교의 목적이 다릅니다.

- **SCALE-Sim E8**: dense conv/linear의 systolic compute-cycle 수식이 같은 가정 아래에서 맞게 구현됐는지 확인
- **Arm Vela E13**: 실제 compiler scheduling을 일부 포함한 외부 추정기와 비교해, 자체 비용 모델이 얼마나 낙관적인지 확인

SCALE-Sim과 잘 맞는다고 실제 실리콘에 정확하다고 해석하지 않습니다. 실제로 Vela와 비교하면 자체 비용 모델은 5/5 조건에서 낙관적이었습니다.

## 5. 왜 부정 결과를 남겼나

이 프로젝트에서 중요한 결과 중 일부는 "개선되지 않았다"는 결과입니다.

- AdaRound: 194/194 layer의 local reconstruction error는 감소했지만 end-to-end top-1 개선으로 이어지지 않음
- Structured pruning: MACs 감소 폭과 estimated cycles 감소 폭이 크게 다름
- LayerNorm 개입: local mismatch를 0%로 만들어도 전체 output-code mismatch는 대부분 남음

이 결과들을 지우면 "무엇을 적용했다"만 남고, **어떤 조건에서 효과가 없었는지에 대한 이해**가 사라집니다.

## 6. 왜 가상 NPU를 썼나

이 저장소는 Mobilint SDK나 실제 Mobilint NPU에서 실행한 프로젝트가 아닙니다.

실제 벤더 하드웨어 없이도 다음 질문을 반복 실험할 수 있도록 가상 프리셋을 만들었습니다.

- operator support가 바뀌면 구조 변경의 가치가 어떻게 달라지는가
- array shape가 바뀌면 pruning 판단이 달라지는가
- fake-quant와 explicit integer execution의 차이는 어디에서 생기는가

따라서 latency·power는 실제 칩 성능으로 주장하지 않습니다.

## 7. 실제 NPU가 있다면 다음에 무엇을 검증할 것인가

실제 Mobilint NPU와 SDK를 사용할 수 있다면 우선순위는 다음과 같습니다.

1. 동일 ONNX/TFLite 계열 모델을 qb Compiler로 변환해 unsupported/fused operator 경계를 확인
2. 자체 cost model의 cycle ranking과 실제 latency ranking의 상관을 측정
3. pruning 전후를 동일 batch·동일 power mode에서 반복 측정
4. host fallback 또는 partition 경계가 latency에 미치는 비용을 실측
5. fake-quant / 자체 integer engine / qb Runtime output을 동일 입력에서 비교
6. 정확도·latency·memory를 동시에 만족하는 최종 selection rule로 확장

## 8. 5분 코드 검토 경로

| 질문 | 먼저 볼 파일 | 확인할 내용 |
|---|---|---|
| 정수 실행을 어떻게 구현했나 | `npuloop/intengine/requant.py` | multiplier, double rounding, saturation |
| NumPy/C++가 실제로 같은가 | `npuloop/intengine/verify.py` | 중간 tensor 비교 |
| pruning이 왜 graph 기반인가 | `npuloop/prune/structured.py` | producer-consumer channel group |
| NPU 비용은 어떻게 계산하나 | `npuloop/npu/cost.py` | systolic mapping, memory, fallback |
| 반복 실험은 어떻게 통제했나 | `npuloop/study_runner.py` | 학습·평가·export 순서 |
| 주장 범위는 어디까지인가 | `docs/VALIDATION_SCOPE.md` | 측정/추정/미검증 구분 |

## 한 문장으로 정리

**모델을 작게 만드는 것보다, 왜 그 변경이 특정 실행 조건에서 유효한지까지 확인하는 것이 이 프로젝트의 목적입니다.**
