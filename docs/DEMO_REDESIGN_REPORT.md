> 이전 안내형 데모의 보관 기록입니다. 현재 화면과 검사 명령은 DEMO_GUIDE.md 및 DEMO_WORKBENCH_REPORT.md를 따릅니다.

# npuloop 데모 개편·검증 결과 — 2026-09-16

## 이번 결과물

기존 `npuloop_integrated_20260916.zip`의 전체 프로젝트를 별도 폴더에 복사한 뒤 데모의 정보 구조와 사용 흐름을 개편했다. **전체 소스가 포함된 새 프로젝트이며 이전 수정 패키지를 추가로 적용하지 않는다.** 원격 GitHub 및 현재 공개된 GitHub Pages는 수정하지 않았다.

압축 해제 후 `npuloop/docs/index.html`을 연다. 새 첫 화면에는 **모델 분석 데모 / 연구 결과 / 검증·재현 자료**의 세 메뉴와 ViT 대표 사례가 보인다.

## 화면에서 달라진 점

### 모델 분석 데모

첫 화면에서 서로 다른 실험의 요약 수치를 나열하지 않는다. ViT-128/6와 지원 제한형 가상 NPU를 기본 사례로 두고 **모델 확인 → 문제 진단 → 변경 전후 → 최종 판단**의 네 단계로 안내한다. 모델·대상 조건을 바꾸면 첫 단계로 돌아가 혼합된 조건을 비교하지 않게 했다.

변경 전후는 E9의 같은 모델·같은 프리셋 기록을 사용한다. ViT의 GELU→ReLU 교체 및 3 epoch 추가 학습 사례에서 strict 프리셋의 추정 사이클은 2,503,878→1,708,230이며, 저장된 정수 정확도는 2,000장 평가 기준 80.80%→81.00%다. 지원형 프리셋의 사이클은 52,246→52,054다. 서로 다른 조건의 숫자를 하나의 성과 카드에 혼합하지 않는다.

최종 화면은 비용이 줄어도 LayerNorm·softmax 지원 제한이 남아 있다는 판단을 보여준다. 이는 기록된 활성함수 변경과 지원 연산 목록을 바탕으로 한 판단이며 실제 NPU 실행 결과가 아니다. 변경 후 실험이 없는 다른 모델은 없다고 표시한다. 임의의 향상 수치나 성공 판정을 만들지 않는다.

### 연구 결과

E15의 같은 모델·같은 양자화 방식에 대해 모의 정확도·정수 정확도·출력 정숫값 불일치를 먼저 보여준다. 정숫값이 소스 코드가 아니라 양자화된 출력 텐서의 원솟값임을 설명한다. 선택한 모델에 맞는 시험셋 크기와 신뢰구간을 표시한다.

E16은 ViT의 LayerNorm 단독 개입 사례로 별도 표시한다. 국소 불일치의 고정 250장과 출력 비교의 전체 10,000장을 구별하고, E9의 모델 변경·학습과는 다른 실험임을 밝힌다. 전체 E15/E16/E17 표와 E13/E14 요약은 접기/펼치기 안에 보존했다.

### 검증·재현 자료

논문·실행 안내·코드 링크와 함께 기존 기능을 네 그룹으로 묶었다. 모델 분석·비용 탐색기, 양자화·모델 변경·프루닝, 외부 대조·실행 시간, 측정 방법·출처·재현이다. 기존 세부 기능과 계산 코드는 삭제하지 않았다. `#cost`, `#intake` 같은 기존 앵커로 이동하면 해당 화면과 접힌 그룹이 자동으로 열린다.

## 실제로 실행한 검증

| 검사 | 결과 | 근거 |
|---|---|---|
| 개편 전 통합본 기준 테스트 | 147 통과·4 건너뜀 | `verification/redesign/baseline_tests.log` |
| 개편 후 전체 테스트 | **168 통과·4 건너뜀** | `verification/redesign/final_tests.log`, `.exit` |
| 개편 후 AVX2 제한 재검증 | **168 통과·4 건너뜀** | `verification/redesign/final_avx2_tests.log`, `.exit` |
| 새 데이터·화면 구조 회귀 테스트 | 21개 추가, 전체 스위트에 포함 | `tests/test_guided_demo.py` |
| 빠른 실행 예제 | 모델 분석·양자화·검증·export·독립 C++ 실행 완료, 종료 코드 0 | `verification/redesign/quickstart.log`, `.exit` |
| 새 안내형 화면 브라우저 검사 | **149/149 assertion 통과** | `verification/redesign/browser/guided_report.json` |
| 기존 전체 데모 컨트롤 검사 | **176개 조건**, 비용 프리셋 20건 및 인테이크 10건 대조 통과 | `verification/redesign/legacy_browser/report.json`, `legacy_browser.log` |
| 데이터·엔진·논문 보존 | 보호 대상 88개 파일 SHA-256 일치, 변경 0개 | `verification/redesign/input_hashes.json`, `protected_integrity.json` |

동일 테스트를 두 실행 환경에서 반복한 것이므로 통과 개수를 더해 336개의 독립 테스트로 부르지 않는다. 브라우저 assertion 수·컨트롤 조건 수도 연구 실험 수를 뜻하지 않는다.

건너뛴 네 항목은 Inception에 softmax가 없어 적용되지 않는 검사 1개, 전체 CIFAR 데이터가 필요한 benchmark 검사 1개와 데이터 분할 검사 2개다. 이번 작업에서 전체 CIFAR 데이터 검사를 실행한 것으로 표시하지 않는다.

새 테스트는 먼저 실패를 확인한 뒤 구현했다. 누락된 실험, 잘못된 호스트 비용, 알 수 없는 변경 방식이 0 또는 성공으로 표시되지 않는 경계를 포함한다. 해당 실패·통과 기록은 `verification/redesign/tdd_*.log`에 있다. `verification/redesign/code_review.md`는 로컬 소스 자체 검토 기록이며 독립 외부 감사 결과가 아니다.

## 브라우저 검사의 조건과 범위

Chromium 144.0.7559.96에서 **완성된 HTML의 바이트를 `page.set_content`로 로드**하고 모든 네트워크 요청을 차단했다. 테스트 중 페이지 오류·콘솔 오류·외부 네트워크 요청은 없었다. 네트워크가 없어도 화면이 돌아가도록 스크립트·스타일·결과 데이터를 HTML에 삽입하고 외부 글꼴 의존성을 제거했다.

새 화면은 1440·768·390·320px에서 세 주 메뉴와 네 단계를 확인했다. 모델·프리셋 변경, 이전/다음, 초기화, 키보드 버튼, 접기/펼치기, 기존 앵커, 브라우저 뒤로/앞으로, 밝은/어두운 테마를 검사했다. 화면 전체의 가로 넘침, 잘못된 내부 앵커, 잘못된 SVG를 찾지 못했다. `docs/index.html`과 `demo/index.html`의 내용이 같고 두 위치 모두에서 상대 링크 대상 파일이 존재함을 확인했다.

**샌드박스 정책이 `file://` 직접 탐색을 차단했으므로, 실제 파일 더블클릭·Windows 브라우저 탐색 또는 HTTP/공개 사이트 배포를 검증했다고 말하지 않는다.** PDF·근거 파일의 링크는 로컬 파일 존재를 검사한 것이며 이 환경에서 실제 새 탭 탐색에 성공했다는 뜻은 아니다.

## 바꾸지 않은 것과 남은 범위

원본 연구 결과 JSON 17개, 논문 파일·표, 정수 엔진·모델 구현 및 번들 예제를 보호 해시로 대조했다. 기존 실험값을 수정하거나 새 학습 결과로 대체하지 않았다. 이번 작업에서 논문을 다시 쓰거나 전체 연구 실험을 재실행하지 않았다.

실제 모빌린트 SDK·NPU 성능 측정, 전체 CIFAR-10·Imagenette 재학습, 모든 과거 체크포인트의 복구, 외부 런타임 실험의 독립 재실행, Windows 네이티브 빌드, GitHub 반영 및 공개 데모 배포는 하지 않았다. 기존 `verification/`의 개편 전 기록과 이번 `verification/redesign/`의 기록은 구별한다.

## 재실행

아래 개발 명령은 Linux 환경 기준이다. 화면을 보는 데는 Python·Node·make 설치가 필요 없다.

```bash
python -m pytest -q -rs
ONEDNN_MAX_CPU_ISA=AVX2 python -m pytest -q -rs
make quickstart
python demo/build.py --from-configs demo/model_configs.json
python tools/check_guided_browser.py
python tools/check_demo_browser.py --browser /usr/bin/chromium --output verification/redesign/legacy_browser
```

브라우저 검사는 Playwright와 Chromium이 필요하다. 자료 원본과 화면 연결 구조는 `docs/DEMO_GUIDE.md`, 최초 열기 방법은 `START_HERE.md`에 있다. 최종 배포 파일의 체크섬은 `MANIFEST_SHA256.json`에 기록한다(매니페스트 자신은 제외).
