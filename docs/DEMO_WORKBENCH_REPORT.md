# npuloop 분석 화면 재구성·검증 — 2026-09-16

## 결과물

`npuloop_workbench_20260916.zip`은 전체 프로젝트입니다. 별도 패치 적용 없이 새 폴더에 풀고 `npuloop/docs/index.html`을 엽니다.
이전 안내형 화면과 질문형 제목을 제거하고 HTML 템플릿, 스타일, UI 컨트롤러를 새로 작성했습니다.
원격 GitHub 및 공개 데모는 변경하지 않았습니다.

## 구현 범위

첫 화면은 `NPU 모델 분석·검증`이며, 선택한 모델·가상 NPU의 분석 결과를 즉시 표시합니다.
모델 분석은 요약, 연산자 상세, 변경 비교, 비용 설정의 네 탭입니다.
별도 영역은 정수 검증, 실험 기록, 재현 자료입니다.

연산자 표에는 검색·실행 경로 필터·정렬·노드 검사·CSV 내보내기가 있습니다.
비용 설정은 배열 크기, 코어 수, DRAM 대역폭, depthwise 엔진, 활성함수 처리 조건을 받습니다.
변경하지 않은 설정은 기본 프리셋으로 유지하며, 활성함수 설정은 LayerNorm·softmax 제한과 독립입니다.
지원 범위 밖 노드를 성공 또는 0비용으로 숨기지 않습니다.

변경 비교는 같은 E9 모델과 프리셋의 실제 저장 기록만 표시합니다. 사용자 비용 설정의 변경 후 실험은 없으면 없다고 표시합니다.
E9의 정수 정확도(2,000장), E15 전체 평가(10,000/3,925장), 연산자별 고정 배치(250/125장)는 구분합니다.
E16은 모의 경로의 LayerNorm 산술 교체이며 E9 활성함수 변경·재학습과 구분합니다.
실험 기록 E1–E17에는 카테고리·검색·모델 필터·페이지 이동·개별 상세 JSON·전체 원본 다운로드가 있습니다.
E4의 원본 FP32, 변경 후 FP32 및 모의 정확도는 서로 다른 원본 필드로 구분합니다.

질문형 홍보 문구, 강의식 진행 문장, 가짜 실행·재학습 진행 표시를 사용하지 않습니다.
기존의 기록 읽기 전용 어댑터와 수치 비용 모델은 재사용했으며 연구 결과값은 수정하지 않았습니다.
기존 안내형 CSS·UI 및 그 화면 전용 브라우저 검사기는 제거했습니다.
과거 검증 보고서는 보관 기록이며, 현재 명령은 `tools/check_workbench_browser.py`입니다.

## 실제 실행 검증

| 검사 | 결과 | 근거 |
|---|---|---|
| 개편 전 기준 테스트 | 168 통과·4 건너뜀 | verification/workbench/baseline.log |
| 최종 전체 테스트 | **221 통과·4 건너뜀** | final_tests.log / final_tests.exit |
| 최종 AVX2 재검증 | **221 통과·4 건너뜀** | final_avx2_tests.log / final_avx2_tests.exit |
| 브라우저 통합 검사 | **585개 assertion 통과** | browser/report.json / browser.log |
| 빠른 실행 예제 | 종료 코드 0 · intake→quantize/verify→export→독립 C++ 실행 | quickstart.log / quickstart.exit |
| 화면 빌드 | 종료 코드 0 · docs/demo HTML 일치 | final_build.log / final_build.exit |
| 문서 표 재생성 검사 | 종료 코드 0 · 별도 임시 복사본에서 확인 | submission_quality.log / submission_quality.exit |
| 보호 파일 | **126개 SHA-256 일치, 변경 0개** | protected_before.json / protected_after.json |

경로는 별도 표기 없으면 verification/workbench/ 아래입니다. 실행 환경은 environment.json에 있습니다.
두 환경의 통과 개수를 합산해 독립 테스트 수로 표현하지 않습니다. 브라우저 assertion 수는 연구 실험 수가 아닙니다.
4개 건너뜀은 Inception의 softmax 비적용 1개, 전체 CIFAR-10 데이터가 필요한 3개입니다.

브라우저 검사는 모델 5종×프리셋 4종, 연구 모델 6종×양자화 방식 2종, 전체 17개 실험의 원본 레코드 대조,
노드/계층 검색, 변경 비교의 빈 상태, 비용 설정, JSON·CSV 다운로드, 키보드, 뒤로/앞으로 가기, 본문 이동을 포함합니다.
1440·1024·768·390·320px에서 메뉴/탭과 전체 페이지 가로 넘침을 확인했습니다. 작은 화면의 표는 표 영역 안에서 가로 스크롤합니다.
밝은·어두운 테마와 모바일 테마 전환을 확인했습니다. 콘솔/스크립트 오류 및 외부 네트워크 요청은 없었습니다.
release_screens/는 최종 HTML에서 직접 캡처한 화면입니다. 이미지 생성기로 만든 UI 예시가 아닙니다.

새 설정/레코드 변환 테스트는 먼저 실패를 확인한 뒤 구현했습니다.
모바일 실험 목록의 최소 너비 전파와 320px 메뉴 넘침은 브라우저 검사로 발견해 수정했습니다.
연속 명령 실행 중 도구 시간 제한으로 quickstart 및 AVX2 실행이 각 한 번 중단되어 별도 실행으로 재검증했습니다.
중단 로그도 보존했으며 중단된 실행을 성공으로 세지 않았습니다. 성공 표는 별도 완료 로그와 종료 코드 기준입니다.

## 검증의 한계

Chromium 144.0.7559.96에서 완성 HTML을 Playwright `page.set_content`로 로드하고 네트워크를 차단했습니다.
Windows 파일 더블클릭, 실제 file:// 새 탭 탐색, 공개 사이트 배포 확인을 수행했다고 주장하지 않습니다.
PDF·참고문서의 상대 링크는 파일 존재를 검사했습니다. 원격 GitHub CI를 실행하거나 공개 사이트를 갱신하지 않았습니다.

코드 검토는 이 작업 환경의 자체 검토이며 독립 외부 감사가 아닙니다.
전체 과거 학습·외부 TFLite/SCALE-Sim/Vela 실험·실제 NPU/모빌린트 SDK 검증은 이번 UI 작업에 포함하지 않았습니다.
논문 PDF, 연구 결과 JSON 17개, 핵심 엔진과 모델, 번들 체크포인트는 변경하지 않았습니다.
화면에서 추정치·저장 측정값·실제 하드웨어 미검증을 구분합니다.

## 공개 참고 자료

Netron 공식 저장소·화면 소스, 과거 OpenVINO Workbench 공식 저장소, AIMET QuantAnalyzer 공식 문서를 검토했습니다.
모델/노드 검사, 동일 조건의 비교, 계층별 결과와 원본 데이터 연결을 참고했습니다.
OpenVINO Workbench 저장소는 2023년 보관된 프로젝트이므로 현재 설치 도구로 권장하지 않습니다.
다른 제품의 코드·로고·문구를 복사하거나 새 의존성으로 설치하지 않았습니다. URL과 적용 범위는 docs/DEMO_REFERENCES.md에 있습니다.

## 재실행

```bash
python -m pytest -q -rs
ONEDNN_MAX_CPU_ISA=AVX2 python -m pytest -v -rs -o faulthandler_timeout=45
make quickstart
make demo-structure
python tools/check_workbench_browser.py --browser /usr/bin/chromium
python tools/submission_quality.py --regenerate
```

브라우저 검사는 Playwright와 Chromium이 필요합니다. 저장된 화면을 사용하는 데는 설치가 필요 없습니다.
