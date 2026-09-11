# 정수 데이터패스 — npuloop 가상 NPU가 실제로 계산하는 것

이 문서는 `npuloop/intengine/`이 구현한 INT8 추론의 **정확한** 산술을 적습니다. "fake-quant가 흉내 내는 것"이 아니라
NPU(또는 TFLite 커널)가 실제로 하는 정수 연산입니다. 모든 수식은 `tests/test_intengine.py`가 참조 구현과 비교해 검증합니다.

## 1. 텐서 표현

| 텐서 | 형식 | 스케일 | zero-point |
|---|---|---|---|
| 활성값 `x` | uint8 (또는 대칭 int8) | per-tensor `s_x` | `z_x ∈ [0,255]` |
| 가중치 `w` | int8 `[-127,127]` | per-output-channel `s_w[c]` | 0 |
| 바이어스 `b` | int32 | `s_x · s_w[c]` | 0 |
| 누산기 | int32 | `s_x · s_w[c]` | — |

실수값 복원: `x_real = (q_x − z_x)·s_x`, `w_real = q_w·s_w[c]`.

## 2. conv / linear

```
acc[c] = Σ (q_x − z_x) · q_w[c]            # int32 (float64/float32 청크로 계산해도 정확히 같은 정수)
acc[c] += b_int[c],  b_int[c] = round(b[c] / (s_x·s_w[c]))
q_y[c] = clamp( MBQM(acc[c], M0[c], shift[c]) + z_y , qmin, qmax )
```

`M[c] = s_x·s_w[c]/s_y` 는 실수이므로 하드웨어는 `M = M0 · 2^shift` (M0는 Q31 고정소수점, `[2^30, 2^31)`)로 근사합니다.
`clamp`가 곧 ReLU/ReLU6입니다: ReLU 뒤에 오는 텐서는 관측 범위가 `[0, max]`라 `z_y = 0`이고, `qmin=0`으로 자르는 것이 ReLU와 동일합니다.
그래서 **ReLU는 NPU에서 공짜**이고, SiLU/GELU는 그렇지 않습니다(§5).

### MBQM — MultiplyByQuantizedMultiplier (gemmlowp / TFLite)

```
SRDHM(a, b)  = sat( (a·b + nudge) / 2^31 ),  nudge = 2^30 (a·b ≥ 0) 또는 1−2^30   # 반올림 ① (half away from zero)
RDBPOT(x, e) = (x >> e) + [ (x & (2^e−1)) > (2^(e−1) − 1 + [x<0]) ]                 # 반올림 ② (half away from zero)
MBQM(x, M0, shift) = RDBPOT( SRDHM(x << max(shift,0), M0), max(−shift,0) )
```

**이중 반올림.** 두 번 반올림하기 때문에 정확히 한 번 반올림한 값과 1 LSB 차이가 날 수 있습니다.
`shift = −1`(M ≈ 0.3~0.5)이면 값의 약 25%가 어긋나고, `shift = −10`이면 0.05% 정도입니다.
TFLite는 이 문제로 `TFLITE_SINGLE_ROUNDING` 빌드 옵션을 추가했습니다:

```
MBQM_single(x, M0, shift) = sat32( RDBPOT64( x·M0 , 31 − shift ) )    # 64비트 곱 한 번, 반올림 한 번
```

두 방식 모두 `RequantConfig(rounding="tflite" | "single")`로 선택할 수 있고 NumPy/C++ 두 엔진이 비트 동일합니다.
E7 실험이 이 차이가 정확도에 얼마나 반영되는지 잽니다.

### 워크드 예제

`s_x = 0.0161, s_w = 0.00271, s_y = 0.0322` → `M = 0.0161·0.00271/0.0322 = 0.001355`
`frexp(0.001355) = 0.6938 · 2^−9` → `M0 = round(0.6938·2^31) = 1,489,838,256`, `shift = −9`
`acc = 123,456` → `SRDHM(123456, M0) = round(123456·M0/2^31) = 85,649`, `RDBPOT(85649, 9) = round(85649/512) = 167`
확인: `123456 · 0.001355 = 167.28 → 167` ✔

## 3. 잔차 add (TFLite 의미론)

두 입력의 스케일이 다르므로 공통 도메인으로 먼저 옮깁니다. 정밀도를 잃지 않으려고 20비트 왼쪽 시프트를 씁니다.

```
twice_max = 2·max(s_1, s_2)
m_1 = s_1/twice_max,  m_2 = s_2/twice_max,  m_o = twice_max / (2^20 · s_y)
y = MBQM( MBQM((q_1−z_1)<<20, m_1) + MBQM((q_2−z_2)<<20, m_2), m_o ) + z_y  → clamp
```

E2의 레이어별 일치 그래프에서 add 노드의 **국소** 불일치가 0인 이유가 이것입니다: 20비트 여유가 있어 fake-quant의 float 덧셈과 같은 격자값으로 떨어집니다.

단, 스케일을 2의 거듭제곱으로 제한하면(`QScheme(pow2=True)`) 두 입력의 스케일 비가 정확한 2의 거듭제곱이 되어 합이 정확히 `.5`에 떨어지는 경우가 흔해집니다. NPU는 half-away, `torch.round`는 half-to-even이라 add 출력의 10~15%가 1 LSB 어긋납니다. 엔진의 2단계 반올림을 `half_even`으로 바꾸면 이 불일치가 정확히 0이 되는 것을 테스트(`test_symmetric_activations_keep_fused_relu`)로 확인했습니다 — "불일치는 반올림 타이(tie)에서만 생긴다"는 것을 증명하는 방법입니다.

### fused ReLU와 clamp — 대칭 int8 활성값에서의 함정

"clamp가 곧 ReLU"는 0의 코드가 `qmin`일 때(uint8, zp=0)만 성립합니다. 대칭 int8(`qmin=-127`)에서는 음수 pre-activation이 clamp를 그대로 통과합니다. E2에서 `sym-act` 스킴의 정수 엔진 정확도가 9%로 무너지면서 이 버그를 잡았고, 지금은 export 단계에서 노드마다 `clamp=(max(qmin, zp), qmax)`(ReLU6는 상한도 `zp+round(6/s)`)를 명시적으로 계산합니다. fake-quant만 봤다면 절대 드러나지 않았을 종류의 버그입니다.

## 4. global average pool

```
q_y = clamp( round_half_away( Σ q_x / (H·W) ) )     # 입력과 같은 s, z 를 유지
```

fake-quant는 평균을 float로 낸 뒤 `torch.round`(half-to-even)로 격자에 올리므로, 합이 정확히 `.5`에 떨어지는 경우(H·W=64이면 드물지 않음) 1 LSB가 어긋납니다. E2에서 pool의 국소 불일치가 ResNet-20(H·W=64)에서 0.7~1.0%, MobileNetV2(H·W=16)에서 2.7~3.7%로 나오는 원인입니다.

## 5. 비-ReLU 활성함수 = 256-entry LUT

```
LUT[q] = clamp( round_half_away( f((q − z_in)·s_in) / s_out ) + z_out )
```

즉 SiLU/GELU/HardSwish 앞의 텐서는 **int8로 한 번 더 양자화**되어야 합니다(pre-activation 양자화 지점). ReLU 모델보다 양자화 지점이 레이어당 하나 더 많고, 그래서 `quant.prepare`는 비-ReLU 활성함수 앞에 `FakeQuantAct`를 하나 더 넣습니다. 이것이 "ReLU 모델은 INT8에 더 강하다"는 통념의 정확한 기계적 이유입니다.

## 6. concat: 여러 스케일을 하나로

concat은 값을 바꾸지 않지만 **스케일을 바꿉니다.** 입력 브랜치마다 `(s_i, z_i)`가 다르므로, 출력 스케일 `s_o` 하나로 모으려면
입력마다 requant가 한 번씩 필요합니다.

```
q_o[i] = clamp( z_o + requant(q_i - z_i, M0_i, shift_i) ),   M_i = s_i / s_o
```

브랜치 범위가 크게 벌어져 있으면(예: 16배) 좁은 브랜치는 255단계 중 16단계만 쓰게 됩니다. lint의 `concat-scale-mismatch`가
캘리브레이션 통계로 이 비율을 재서 경고합니다.

## 7. attention: matmul · softmax · LayerNorm

### 7.1 활성값 × 활성값 matmul

conv/linear은 한쪽이 정적 가중치라 `Σ w(q - z)`로 끝나지만, `QK^T`와 `PV`는 **양쪽 다 활성값**이라 두 영점을 모두 펼쳐야 합니다.

```
acc = Σ_k (a_ik - z_a)(b_kj - z_b)          # int32 누산
q_y = clamp( z_y + requant(acc, M0, shift) ),   M = s_a · s_b / s_y
```

`1/√d` 스케일은 별도 연산이 아니라 `M`에 곱해 넣습니다(그래프에서 `mul(scalar)` 노드는 사라지고 producer의 requant 배수에 접힙니다).
비용 모델에서는 이 연산에 **가중치 재사용이 없다**는 점이 핵심입니다 — weight-stationary 배열은 헤드마다, 이미지마다 타일을 다시 채웁니다.

### 7.2 softmax

행 최댓값을 뺀 뒤(`d = q - max ≤ 0`) Q15 exp 테이블을 찾고, 정수 나눗셈으로 정규화합니다.

```
e_j   = exp_lut[d_j + offset]                  # round(exp(d·s_in) · 2^15)
total = Σ_j e_j
v_j   = round_div(e_j · 2^15, total)           # Q15 확률
q_j   = clamp( z_o + requant(v_j, M0, shift) ),  M = 2^-15 / s_o
```

테이블은 입력 스케일에 의존하므로 export 시점에 만들어집니다(활성함수 LUT와 같은 방식). 정규화를 **정확한 정수 나눗셈**으로
정의했기 때문에 NumPy 엔진과 C++ 커널이 비트 단위로 같습니다 — 실제 NPU는 역수 근사를 쓰는 경우가 많고, 그 차이는 E7식
ablation으로 잴 수 있습니다.

### 7.3 LayerNorm

평균과 분산을 int64로 정확히 구하고, 나눗셈 대신 **정수 제곱근**을 씁니다.

```
d_i     = q_i - z_in
S = Σ d_i,  Q = Σ d_i²
var_num = C·Q - S²                             # = C² · var(d), 항상 ≥ 0
denom   = isqrt64(var_num + eps_int),            eps_int = round(eps · C² / s_in²)
t_i     = round_div((d_i·C - S) · 2^15, denom)   # Q15 정규화 값
q_i     = clamp( z_o + requant(t_i, M0_c, shift_c) + β_q[c] ),   M_c = γ_c / (2^15 · s_o)
```

`isqrt64`는 float64 sqrt 뒤에 정수 보정을 넣어 **정확한 floor(√x)**를 냅니다. 두 엔진이 같은 알고리즘을 쓰므로 결과가 어긋날 수 없습니다.
β는 출력 도메인에서 더하기 때문에 최대 0.5 LSB의 반올림 오차가 있습니다(하드웨어 커널이 흔히 쓰는 방식이고, 한계에 적어 두었습니다).

## 8. 엔진이 일부러 틀리게 계산하는 방법 (E7 ablation)

`RequantConfig`:

| 필드 | 의미 | 값 |
|---|---|---|
| `rounding` | 2단계 반올림 방식 | `tflite`(이중) · `single` · `half_even` · `truncate` · `floor` |
| `mult_bits` | M0의 유효 비트 | 31 (기준) · 15 · 7 · 3 |
| `acc_bits` | 누산기 폭 (포화) | 32 · 24 · 20 · 16 |
| `bias_bits` | 바이어스 폭 (포화) | 32 · 16 · 12 |

같은 양자화 파라미터로 이 값만 바꾸면 "싸구려 정수 구현"이 정확도에 얼마를 물리는지 이미지 단위 짝지은 비교(top-1 일치율)로 잴 수 있습니다.

ResNet-20 ReLU, 2,000장, 기준 = `tflite-m31-acc32-b32` 90.1% (README E7 표 참고; 검증 분할로 다시 학습한 체크포인트 기준):

| 바꾼 것 | 정확도 | 기준과 top-1 일치 | 해석 |
|---|---|---|---|
| single rounding / half-even | 90.1 / 90.2% | 99.2% | 2단계 반올림 방식은 사실상 무관 |
| truncate / floor | 89.5 / 89.1% | 95.6 / 94.8% | 반올림 없는 시프트는 체계적 편향 → −0.7~1.0%p (SiLU·MobileNetV2는 −1.4~2.6%p) |
| 곱셈기 15 / 7 / 3비트 | 90.2 / 90.4 / 81.9% | 99.2 / 99.2 / 85.7% | 2^k 곱셈기(3비트)면 −8.2%p; 다른 두 모델은 −1.6~2.0%p — 어느 레이어의 M0가 2^k 격자에서 얼마나 멀리 떨어지느냐에 달렸다 |
| 바이어스 int16 / int12 | 79.6 / 13.5% | 83 / 14% | 바이어스는 `s_in·s_w` 단위라 값이 크다 — 폭을 줄이면 안 된다 |
| 누산기 int24 / int20 / int16 | 90.1 / 90.1 / 16.8% | 100 / 100 / 17% | 실제 누산값은 20비트에 들어가고, 16비트는 1,115만 회 포화 |

## 9. C++ 커널의 구조와 "언제 int32로 누산해도 되는가"

`intengine/cpp/int8_engine.cpp`는 NumPy 엔진과 같은 의미론을 다른 구현으로 한 번 더 쓴 것이고, 두 엔진은 테스트에서 모든
중간 텐서가 비트 단위로 같아야 합니다. 처음 버전은 conv를 7중 루프로 직접 계산했고 NumPy 엔진(내부적으로 float32 `conv2d`,
K가 작아 정확)보다 **3배 느렸습니다**(E10에서 처음 잰 값). 지금 구조는 다음과 같습니다.

* **conv = im2col + 정수 GEMM.** 이미지·그룹마다 zero-point를 뺀 입력을 `Xt`(K × P, P = 출력 픽셀 수)로 펼치고, 출력 채널마다
  K개의 탭을 4개씩 묶어 P 길이의 int32 행에 axpy로 더합니다(수평 합산 없음, 컴파일러가 벡터화). 에필로그(`acc_bits` 포화 →
  바이어스 → MBQM → clamp)는 conv·linear가 공유합니다.
* **정확성 규칙.** 참조 의미론은 "int64로 정확히 합산한 뒤 `acc_bits`로 포화"입니다. 부분합을 int32에 두어도 결과가 같으려면
  중간 합이 int32를 넘지 않아야 하므로, 커널은 레이어마다 `K · max|x − zp| · 128 < 2³¹`을 검사해서 참이면 int32, 아니면
  int64 경로를 씁니다(둘 다 같은 템플릿). 8비트 코드에서는 K가 65,000을 넘지 않는 한 항상 int32 경로입니다.
* **linear · matmul**도 같은 규칙입니다. activation×activation matmul은 두 피연산자 모두 zero-point를 빼 두고
  `K · max|a − zₐ| · max|b − z_b| < 2³¹`을 검사합니다.
* **빌드.** `g++ -O3 -march=native`로 import 시점에 컴파일하며(소스 해시 + 플래그 + CPU 모델로 캐시), `NPULOOP_CPP_NATIVE=0`이면
  일반 x86-64로 빌드합니다. 스레드는 1개입니다 — 이 엔진의 목적은 속도가 아니라 검증이고, E10의 시간은 그 검증이 이
  호스트에서 얼마나 걸리는지를 잰 것입니다.


## 10. TFLite와 비트를 맞추며 배운 것 (E11)

TFLite full-integer 모델(conv-relu ×3, MEAN, fully-connected)의 스케일·zero-point·int8 가중치·int32 바이어스를 `.tflite`에서 읽어 이 저장소의
IntGraph로 다시 조립하고, TFLite reference 커널(`BUILTIN_REF` 리졸버)과 모든 중간 텐서를 비교했습니다.

* **conv·MEAN은 gemmlowp 이중 반올림(§2의 MBQM), fully-connected는 단일 반올림.** fc만 `single`로 바꾸면 1,000장 × 모든 텐서가 0개
  불일치입니다. 그래서 `IntNode.attrs["rounding"]`로 노드별 반올림을 덮어쓸 수 있게 했고, 세 엔진(NumPy·ctypes C++·독립 러너)이 모두 따릅니다.
* **TFLite MEAN(int8)** 은 `acc = Σ(x − zp_in)`을 `QuantizeMultiplier(s_in / (s_out · HW))`로 한 번 requant하고 `zp_out`을 더합니다 — 출력 스케일이
  입력과 다릅니다. NPU식 평균 풀링(§4, 스케일 유지·반올림 나눗셈)과 구분해 `pool` 노드의 `kind="global_avg_requant"`로 표현합니다.
* **XNNPACK 델리게이트와 reference 커널은 서로 다릅니다** (출력 코드가 1,000장 중 155장에서 상이). "하드웨어와 다르다"는 보고는 기준 런타임을
  먼저 정해야 합니다.

## 11. `.npuloop` 파일과 독립 실행기

`intengine/serialize.py`가 IntGraph를 한 파일로 씁니다: 8바이트 매직 `NPULOOP1` · uint32 헤더 길이 · JSON 헤더(노드·op·입력·출력 양자화·JSON-safe attrs·
배열 디스크립터) · raw little-endian 배열(가중치 int8, 나머지 int32). float은 입력/출력 스케일뿐입니다. `cpp/int8_runner.cpp`(자체 JSON 파서 +
같은 커널 소스)가 파이썬 없이 이 파일을 실행하며, `tests/test_export_runner.py`가 네 아키텍처의 모든 노드에서 NumPy 엔진과 코드 단위 일치를 강제합니다.
