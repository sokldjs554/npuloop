# 원고 (영문 투고본 · 한국어판)

같은 내용의 4쪽 원고 두 판입니다. 본문의 모든 수치·표·그림은 `results/*.json`에서 스크립트로 생성되므로
실험을 다시 돌리면 생성기만 재실행하면 되고, 두 판이 서로 다른 숫자를 말하는 일은 생기지 않습니다.

| 파일 | 내용 | 엔진 |
|---|---|---|
| `npuloop_esl.tex` → `npuloop_esl.pdf` | **IEEE Embedded Systems Letters 투고본(영문)** | `pdflatex` |
| `npuloop_ko.tex` → `npuloop_ko.pdf` | 한국어판 — 읽기용이자 국내 학술대회용 출발점 | `lualatex` |
| `make_tables.py` | Table I(E15) · II(E16) · III(E14) 생성, `--lang ko`로 한국어판 | |
| `make_figs.py` | Fig. 1(국소/전파 분해) · Fig. 2(반올림 × 시드) 생성, `--lang ko`로 한국어판 | |

두 판 모두 최근 실증 연구 논문의 관례를 따릅니다 — 서론에서 연구 질문(RQ1~RQ3)을 명시하고, 결과를 번호 붙인
**발견(Finding) 1~5**와 각 절의 **실무 함의(Implication)**로 요약하며, 관련 연구는 2장으로 앞당기고,
한계는 외적·구성·결론·내적으로 나눈 **타당성 위협(Threats to Validity)**으로, 재현 정보는 **재현 자료
(Artifact Availability)**로 따로 둡니다. 한국어판은 국내 관례에 맞춰 장 번호를 아라비아 숫자로 쓰고
(1. 서론 / 4.2 ...) 끝에 영문 초록을 답니다. 국내 학회에 낼 때 본문 구조는 그대로 두고 그 학회 양식만
갈아 끼우면 됩니다(대부분 장 제목이 좌측 정렬인 점만 다릅니다).

**ESL은 영문만 받습니다.** 투고하는 원고는 `npuloop_esl.tex`이고, 한국어판은 같은 내용을 우리말로 옮긴 것입니다.
국내 학술대회(대한전자공학회 추계학술대회, KIISE KSC, KIPS ACK 등)는 대개 2~4쪽 한글 원고를 받으므로
한국어판을 그 학회 양식으로 갈아 끼우면 그대로 쓸 수 있습니다. 다만 같은 내용을 두 곳에 내는 것은 이중 게재가 되니,
둘 중 한 곳만 고르거나 국내 학회를 먼저 낸 뒤 확장본을 ESL에 내는 순서로 가야 합니다.

```bash
# 영문 투고본
python paper/make_tables.py && python paper/make_figs.py
cd paper && pdflatex npuloop_esl && pdflatex npuloop_esl      # 두 번 (상호참조)

# 한국어판
python paper/make_tables.py --lang ko && python paper/make_figs.py --lang ko
cd paper && lualatex npuloop_ko && lualatex npuloop_ko

make paper      # 위 네 줄을 한 번에
```

한국어판 빌드에 필요한 것 (우분투 기준):

```bash
apt-get install -y texlive-lang-korean texlive-luatex fonts-nanum fonts-texgyre
rm -rf ~/.cache/matplotlib        # 새로 깐 한글 글꼴을 matplotlib이 다시 훑도록
```

한글 조판은 `luatexko`, 라틴 문자는 IEEEtran과 같은 Times 계열(TeX Gyre Termes)입니다. `pdflatex`로는 빌드되지
않습니다. 한글 글꼴이 없으면 `make_figs.py --lang ko`가 두부 글자를 그리는 대신 무엇을 설치해야 하는지 알려주고 멈춥니다.
한국어가 영문보다 조밀해서 한국어판은 4쪽 틀 안에서 3쪽 남짓만 차지합니다 — 국내 학회용으로 늘릴 여지가 그만큼 있습니다.

## 투고 전에 반드시 채울 것

1. **저자 정보** — 소속(무소속 연구자)과 연락처는 채워져 있습니다. 남은 것은 대괄호 두 곳뿐입니다:
   인쇄될 **성명**(`[Author~Name]` / `[저자 성명]`)과 **도시**(`[City]` / `[도시]`). ScholarOne 제출 시 ORCID도 필요합니다.
2. **감사의 글** — AI 사용 공개 문구가 들어 있습니다(IEEE 요구사항). 지금은 최소 문구이니, 더 구체적으로 적고 싶으면 고치세요.
3. **쪽수 확인** — ESL은 그림·표·참고문헌을 **포함해 4쪽**입니다. 영문본이 정확히 4쪽이므로 문장을 추가하면 넘칩니다.
4. **투고 경로** — IEEE CEDA의 ESL author instructions 페이지에서 ScholarOne 제출 링크를 확인하세요. 상시 접수이고 1차 결정까지 약 1개월입니다.
5. **선공개** — arXiv에 먼저 올릴 경우, 처음 투고하는 카테고리(cs.LG·cs.AR)에는 endorser가 필요할 수 있습니다.

## 본문이 주장하는 것 (검토용 요약)

* **주장 1** — fake-quant는 정확도를 맞히지만(12행 전부 95% CI가 0을 포함, 최대 |Δ|/SE 1.6) 텐서는 맞히지 못한다(출력 코드 28.5~72.7% 불일치).
* **주장 2** — 격차는 특정 연산자 하나가 옮기는 것이 아니다. 가장 큰 국소 원천(LayerNorm 24.77%)을 0으로 만들어도 전파 불일치는 72.7% → 65.7%까지만 내려간다.
* **주장 3** — requant 반올림을 시프트로 바꾸면 18개 조합 전부에서 손해(−0.90~−3.37%p)이고, **부호는 시드에 강건하지만 크기는 아니다**.

각 주장의 한계는 한계(V장)에 그대로 적혀 있습니다 — CIFAR-10 규모, 반올림 축만 시드 3개, 다중비교 미보정,
외부 비트 일치 검증 범위는 conv·pool·fc.
