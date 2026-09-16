# 공개 화면 구성 참고 자료

검토일: 2026-09-16. 도구의 공식 설명과 공개 소스 구조를 참고했다. 소스 코드·로고·문구를 복사하지 않았고 런타임 의존성으로 추가하지 않았다.

## Netron

https://github.com/lutzroeder/netron
https://raw.githubusercontent.com/lutzroeder/netron/main/source/index.html

모델을 작업 대상으로 두고 속성과 노드를 별도로 검사하는 구조를 참고했다.
npuloop은 Netron의 범용 모델 파일 뷰어를 구현한 것이 아니다. 현재 번들의 모델 구조와 저장된 결과를 사용한다.

## OpenVINO Deep Learning Workbench

https://github.com/openvinotoolkit/workbench

모델·하드웨어 조건을 고정하고 성능을 비교하는 작업 구조를 참고했다.
이 저장소는 2023-08-28 보관 처리된 과거 프로젝트이며 현재 유지보수되는 도구로 소개하거나 설치를 권하지 않는다.
소스를 실행하지 않았고, 과거 UI/작업 구조의 참고로만 사용했다.

## AIMET QuantAnalyzer

https://quic.github.io/aimet-pages/releases/2.25.1/techniques/analysis_tools/quant_analyzer.html

공식 문서의 계층별 분석, HTML 결과와 원본 JSON의 연결 구조를 참고했다.
npuloop의 측정 정의와 데이터는 자체 실험을 따른다. AIMET의 FP32 대 양자화 MSE와 npuloop의
모의 양자화 대 정수 코드 불일치는 같은 지표로 표시하지 않는다.

## 적용 범위

큰 홍보 제목이나 단계형 안내 대신 선택 조건, 요약 지표, 연산자 표, 노드 검사 패널,
동일 조건의 변경 비교, 측정 데이터 및 원본 기록을 각각의 작업 영역으로 분리했다.
모든 화면은 로컬에서 표시되며 브라우저에서 학습·INT8 추론을 실행하는 것처럼 연출하지 않는다.
