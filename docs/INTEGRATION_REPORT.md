# npuloop 통합 검증 결과 — 2026-09-16

## 결과물과 변경 범위

사용자가 올린 `npuloop-master.zip` 전체 소스에 이전 수정 패키지를 적용하고, 실행에서 추가로 발견한 데모 오류와 체크포인트 읽기 경로를 수정했다. **이 결과물은 전체 프로젝트이며 별도 패치를 적용할 필요가 없다.** 업로드 원본 ZIP 및 원격 GitHub·공개 데모는 변경하지 않았다.

원본 ZIP의 GitHub archive comment는 `8af3e3ada526d5d428c693e911bc06f5c12b1e71`이다. 수정 작업은 별도 복사본에서 수행했다. 전체 원본 연구 결과 JSON 17개, 번들 예제 파일 4개, 논문 표 소스 9개는 업로드 원본과 SHA-256 및 파일 내용이 일치한다. 새 연구 측정값으로 기존 결과를 대체하지 않았다.

## 실제로 수행한 검사

| 검사 | 결과 | 근거 파일 |
|---|---|---|
| 업로드 원본 기준 테스트 | 84 통과·4 건너뜀 | `verification/baseline_tests.log` |
| 수정 통합본 전체 테스트 | 147 통과·4 건너뜀 | `verification/verify_final.log` |
| 통합본 AVX2 제한 재검증 | 147 통과·4 건너뜀 | `verification/final_avx2_tests.log` |
| `make verify` | 종료 코드 0, 테스트·quickstart·논문 표 재생성 검사 완료 | `verification/verify_final.log`, `verify_final.exit` |
| 선택적 benchmark 경로 | 번들 NPZ를 지정해 별도 2개 통과 | `verification/optional_benchmark.log` |
| 학습된 번들 체크포인트 2개 | 각각 8장, NumPy↔C++↔export 후 독립 실행 결과 일치 | `verification/bundled_engines.json` |
| 브라우저 비용 모델 | 모델 5종×프리셋 4종, 노드별 Python↔JS 대조 테스트 통과 | `tests/test_demo_complete.py`, 전체 테스트 로그 |
| 실제 UI 프리셋 | 비용 비교 20건·인테이크 비교 10건 일치 | `verification/browser/report.json` |
| 전체 HTML 컨트롤 | 146개 조건, 1440px·390px, 오류·페이지 가로 넘침 없음 | 위 브라우저 보고서 |
| 새 연구 표 | E15 12행·E16 4행·E17 2행, E13·E14 요약 표시 | 위 브라우저 보고서 |
| 연구 원고 | 저장 JSON에서 9개 표·3개 그림 재생성 및 LuaLaTeX 3회 빌드, 27쪽 | `verification/paper_generation.log`, `thesis_build_3.log` |
| PDF 사전 점검 | 저자 윤기혁, 미기입 성명 없음, 페이지 밖 텍스트 없음 | `verification/pdf_preflight_final.json` |

전체 테스트의 4개 건너뜀은 Inception의 softmax 비적용 1개, 기본 경로의 CIFAR 데이터 미존재로 인한 benchmark 1개와 데이터 분할 검사 2개다. benchmark는 번들 NPZ를 지정해 별도로 실행했다. 전체 CIFAR 데이터를 사용하는 두 분할 검사를 실행했다고 합쳐 말하지 않는다.

테스트 수는 테스트 케이스 수이며 연구 실험 수나 논문 데이터 전체를 뜻하지 않는다. 두 기본/AVX2 실행의 결과를 합쳐 294개의 독립 테스트로 표시하지 않는다.

## 번들 실행에서 확인한 값

`make quickstart`의 ResNet-20 번들 표본 500장에서는 float 정확도 88.60%, 모의 양자화 88.40%, 정수 엔진 88.60%가 출력됐다. 이 값은 **번들 표본과 quickstart 보정 조건의 실행 확인값**이지 논문의 전체 시험셋 결과가 아니다. 논문 표는 기존 전체 실험 기록을 유지했다.

별도 정수 경로 대조는 64장 보정(seed 0)·시험 이미지 8장을 사용했다.

| 번들 모델 | 입력 포함 정수 그래프 노드 | 독립 실행기에서 비교한 입력 외 노드 | 비교한 출력·중간 코드 원소 | 불일치 |
|---|---:|---:|---:|---:|
| ResNet-20 ReLU | 35 | 34 | 2,294,944 | 0 |
| Inception-32 | 23 | 22 | 2,412,704 | 0 |

이는 같은 정수 그래프를 여러 실행 경로에서 재현하는 검사다. 독립적인 외부 런타임 검증이나 실제 NPU 실행을 대신하지 않는다. 두 모델의 파일 체크섬과 실제 runner 출력은 `bundled_engines.json`에 있다.

## 이번 통합에서 추가로 발견하고 고친 문제

### 브라우저에서 ViT·Inception 비용이 일부 빠지던 문제

이전 JS 비용 모델은 CNN 위주 연산만 처리해 matmul·LayerNorm·softmax·concat·전치 등의 비용을 빠뜨렸다. 기존 대조 테스트도 residual/depthwise CNN만 다뤄 이 차이를 발견하지 못했다. 지원하는 모든 연산과 필요한 shape 정보를 브라우저 계산에 전달하고, Python의 레이어별 비용·총 MACs·활용률·DRAM 전송량·bound 분류와 대조하도록 확장했다. 알 수 없는 연산을 0사이클로 처리하지 않고 명시적으로 실패한다.

또한 UI에서 strict 프리셋을 고른 뒤 활성함수 옵션을 갱신하면 softmax·LayerNorm 지원 제한까지 지워지던 문제를 수정했다. 프리셋 UI를 직접 조작한 20건을 저장된 Python 계산과 대조했다.

### 데모가 E12까지만 담고 있던 문제

E1–E17 저장 결과를 모두 다시 포함하고 E13–E17 연구 요약·표를 추가했다. 원본 `runs/`가 없으므로 모델 구조 설정을 사용해 비용 그래프만 다시 추적하는 **명시적 구조 전용 빌드**를 추가했다. 설정의 MACs·파라미터 수를 검사하고 HTML에 `architecture_config_only`, `checkpoint_evaluation_performed=False`를 표시한다. 임시 초기화한 모델로 정확도를 평가하거나 그 값을 논문 실험으로 표시하지 않는다.

### 체크포인트 읽기

공통 로더·분류 학습 재개·초해상 학습 재개 경로에 `weights_only=True, map_location="cpu"`를 명시하고 CLI의 중복 로드를 제거했다. 모델 체크포인트는 config와 텐서 state_dict의 구조를 검사한다. 임의 객체를 허용하는 경로로 자동 재시도하지 않는다. 작은 사용자 정의 객체 fixture가 실행되지 않고 거부되는지, 번들 가중치가 그대로 읽히는지, 작은 합성 데이터의 학습 완료 상태 재개가 되는지 테스트했다. 이는 일반적인 악성 모델 파일 전반의 안전성 인증이 아니다.

### 문서·화면

이전 논문 수정안을 실제 원문과 README에 통합했다. 종류별 평균과 개별 노드의 차이, 통계적 비유의와 동등성의 차이, LayerNorm 단독 개입의 결론 범위, 외부 검증 범위, 가상 NPU 추정과 호스트 측정을 구분했다. 원고의 표·그림은 이번에는 업로드된 실제 결과 JSON에서 다시 만들었다.

데모에는 viewport·한국어 줄바꿈·내부 스크롤 여백을 추가했다. 섹션 이동 시 제목이 고정 헤더 뒤에 가려지는 문제도 검사 후 수정했다. 논문 링크의 PDF는 실제 파일이 존재하고 canonical PDF와 동일한 바이트인지 확인했다.

## 브라우저와 문서 검증의 조건

Chromium 144.0.7559.96에서 완성 HTML을 `page.set_content`로 로드했다. 네트워크의 로컬 서버 접근이 차단된 환경이므로 HTTP 서버를 통한 검증이라고 표시하지 않는다. 외부 Google Fonts CSS는 비워 시스템 글꼴로 표시했다. 밝은/어두운 테마를 확인하고 자바스크립트 오류, NaN/Infinity, 깨진 내부 앵커, 화면 전체 가로 넘침과 헤더에 가려진 섹션 제목을 검사했다. **호스팅된 GitHub Pages를 접속·배포 확인한 것은 아니다.**

장문판 PDF는 전체 27쪽을 렌더링해 조판을 확인했다. 예전 영문·국문 4쪽 초안은 그대로 보관했고 이번 문장 수정·재빌드 대상이 아니다. 제출 연결에는 `paper/npuloop_thesis.pdf`를 사용한다. 개인 연구 원고이며 학위논문 제출이나 학술지 채택 사실을 뜻하지 않는다.

## 재실행 명령

저장소 루트에서 실행한다. `verification/environment.json`에 이번 환경을 기록했다.

```bash
python -m pip install -e '.[dev]'
make verify
ONEDNN_MAX_CPU_ISA=AVX2 python -m pytest -q -rs
NPULOOP_DATA=examples/quickstart/cifar10_sample.npz python -m pytest tests/test_bench.py -q -rs
make runner
python verification/check_bundled_engines.py --root . --report verification/bundled_engines.json
make demo-structure
make paper-reviewed
# 선택적 브라우저 검사: Playwright와 Chromium 필요
python tools/check_demo_browser.py --browser /usr/bin/chromium --output verification/browser
```

위 명령의 `make` 및 환경변수 문법은 Linux 기준이다. Windows 네이티브 빌드를 시험하지 않았다. 설치 없이 화면과 PDF를 보는 데는 Python이나 make가 필요 없다.

## 아직 수행하지 않은 범위

전체 CIFAR-10·Imagenette 원본과 모든 과거 학습 체크포인트는 업로드에 없었다. 따라서 E1–E17 전체 학습·재평가, 원래 초기화까지 고정한 과거 결과 재현, TFLite·SCALE-Sim·Vela 외부 실험의 독립 재실행은 하지 않았다. 번들 학습 가중치 2개는 검증했지만 다른 모델의 가중치를 복원하지 않았다.

실제 모빌린트 SDK·NPU 하드웨어 성능 측정, 원격 GitHub 반영, 공개 데모 배포는 하지 않았다. 수정된 CI 설정은 제공하되 GitHub에서 실행된 것으로 표시하지 않는다. 개인 이력서·자기소개서 최종 제출본을 이번 ZIP으로 자동 제출하지 않는다.

현재 완료 범위는 **프로젝트 소스 통합, 확인된 오류 수정, 로컬 실행과 데모·장문 원고 검증**이다. 다음 반영 단계에서는 이 ZIP을 기존 저장소의 미커밋 변경과 비교하고, 공개 브랜치/데모에 적용한 뒤 그 주소에서 다시 확인해야 한다.
