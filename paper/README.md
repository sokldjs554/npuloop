# ESL 원고 초고

IEEE Embedded Systems Letters 투고용 4쪽 초고입니다. 본문의 모든 수치·표·그림은 `results/*.json`에서 스크립트로 생성되므로,
실험을 다시 돌리면 `make_tables.py`·`make_figs.py`만 재실행하면 원고가 따라옵니다.

```bash
python paper/make_tables.py     # results/*.json -> tab{1,2,3}_rows.tex
python paper/make_figs.py       # results/*.json -> fig{1,2}*.pdf
cd paper && pdflatex npuloop_esl && pdflatex npuloop_esl    # 두 번 (상호참조)
```

| 파일 | 내용 |
|---|---|
| `npuloop_esl.tex` | 본문 (IEEEtran, journal 스타일) |
| `make_tables.py` | Table I(E15) · II(E16) · III(E14) 생성 |
| `make_figs.py` | Fig. 1(국소/전파 분해) · Fig. 2(반올림 × 시드) 생성 |
| `npuloop_esl.pdf` | 현재 빌드 결과 (4쪽) |

## 투고 전에 반드시 채울 것

1. **저자 정보** — `\author{}` 블록의 `[affiliation]`, `[city, country]`, `[e-mail]`과 `First~Author`. ScholarOne 제출 시 ORCID도 필요합니다.
2. **감사의 글** — AI 사용 공개 문구가 들어 있습니다. 실제로 사용한 범위에 맞게 고치세요(IEEE는 공개를 요구하지만 문구는 저자가 정합니다).
3. **쪽수 확인** — ESL은 그림·표·참고문헌을 **포함해 4쪽**입니다. 현재 정확히 4쪽이므로 문장을 추가하면 넘칩니다.
4. **투고 경로** — IEEE CEDA의 ESL author instructions 페이지에서 ScholarOne 제출 링크를 확인하세요. 상시 접수이고 1차 결정까지 약 1개월입니다.
5. **선공개** — arXiv에 먼저 올릴 경우, 처음 투고하는 카테고리(cs.LG·cs.AR)에는 endorser가 필요할 수 있습니다.

## 본문이 주장하는 것 (검토용 요약)

* **주장 1** — fake-quant는 정확도를 맞히지만(12행 전부 95% CI가 0을 포함, 최대 |Δ|/SE 1.6) 텐서는 맞히지 못한다(출력 코드 28.5~72.7% 불일치).
* **주장 2** — 격차는 특정 연산자 하나가 옮기는 것이 아니다. 가장 큰 국소 원천(LayerNorm 24.77%)을 0으로 만들어도 전파 불일치는 72.7% → 65.7%까지만 내려간다.
* **주장 3** — requant 반올림을 시프트로 바꾸면 18개 조합 전부에서 손해(−0.90~−3.37%p)이고, **부호는 시드에 강건하지만 크기는 아니다**.

각 주장의 한계는 V장(Limitations)에 그대로 적혀 있습니다 — CIFAR-10 규모, 반올림 축만 시드 3개, 다중비교 미보정,
외부 비트 일치 검증 범위는 conv·pool·fc.
