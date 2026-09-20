# npuloop — NPU 모델 분석·검증

## 실행

공개 데모: https://sokldjs554.github.io/npuloop/

로컬에서는 저장소의 `docs/index.html`을 브라우저로 엽니다.
전체 프로젝트 ZIP을 받은 경우 새 폴더에 압축을 풀고 `npuloop/docs/index.html`을 엽니다.
별도 패치, Python·Node·웹 서버·API 키는 화면 실행에 필요하지 않습니다.

첫 화면은 `NPU 모델 분석·검증`이며 모델과 가상 NPU를 선택하면 진단 결과가 표시됩니다.
기존의 안내형 단계 화면이 보이면 이전 ZIP의 index.html을 연 것입니다.

## 화면

모델 분석의 네 탭은 분석 요약, 연산자 상세, 변경 비교, 비용 설정입니다.
연산자 상세에서 노드를 선택하면 우측 패널에 비용과 연결 관계가 표시됩니다.
변경 비교는 선택 조건과 일치하는 저장된 E9 기록만 표시합니다.
비용 설정을 수정하면 비용을 다시 계산하며, 해당 사용자 조건의 변경 후 실험이 없으면 없다고 표시합니다.

정수 검증에서는 E15 모델·양자화 방식을 선택합니다. 화면의 모의·정수 정확도는 전체 평가셋,
노드별 차이는 별도 고정 배치입니다. E16 LayerNorm 산술 교체와 E17 초해상도 결과도 구분해 표시합니다.

실험 기록에서 전체 E1–E20을 조회하고 각 기록의 JSON을 열거나 저장할 수 있습니다.
재현 자료에서 논문 PDF와 실행 안내를 확인합니다.

## 데이터 범위

NPU 사이클은 가상 조건의 추정값입니다. 정확도·출력 불일치는 기존 실험 결과입니다.
브라우저에서 실제 NPU 추론, 모델 양자화, 학습을 수행하지 않습니다.
연구 결과 JSON, 정수 엔진, 번들 체크포인트의 수치를 바꾸지 않았습니다.
연구 원고와 데모는 승인된 소스에서 빌드했습니다.

## 개발 환경 검증

아래 명령은 Linux 기준입니다. Windows 네이티브 컴파일은 별도 검증 대상입니다.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q -rs
make quickstart
make demo-structure
# 선택적 브라우저 검사: Playwright와 Chromium 필요
python tools/check_workbench_browser.py --browser /usr/bin/chromium
```

로컬 검증의 범위는 `docs/DEMO_WORKBENCH_REPORT.md`에 기록합니다.
게시 준비 과정에서 GitHub Actions로 다시 실행한 검증 기록은 `verification/publish/`에 있습니다.
공개 데모의 실제 반영 여부와 배포 후 파일 대조 결과는 저장소 Actions의 `pages` 실행을 기준으로 확인합니다.
