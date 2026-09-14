# 사용법

[← README](../README.md) · [실험 E1–E16](EXPERIMENTS.md) · [설계 문서](DESIGN.md)

README의 "3분 안에 보기"가 데이터도 학습도 없이 도는 경로라면, 이 문서는 그 다음입니다 — 데이터셋을 만들고, 직접 학습하고,
CLI의 여섯 동사를 개별적으로 쓰고, 실험을 재현하는 방법.

## 설치와 전체 명령

```bash
pip install -e .[dev]           # torch(CPU), numpy, pytest
make quickstart                 # 데이터·학습 없이 40초: 번들 체크포인트로 intake → quantize/verify → export + C++ runner
python -m pytest -q             # 87 tests, ~35 s (C++ 커널은 첫 실행 때 g++로 컴파일되어 처음엔 더 걸립니다)

# CIFAR-10 (npz 한 파일) 준비: tools/prepare_cifar10.py 참고
python -m npuloop.zoo.train --arch resnet --act relu --epochs 30 --out runs/resnet20_relu --data data/cifar10.npz

npuloop cost runs/resnet20_relu/best.pt --spec edge-10tops          # 레이어별 사이클/활용률 표
npuloop lint runs/resnet20_relu/best.pt --spec edge-10tops --data data/cifar10.npz
npuloop quantize runs/resnet20_relu/best.pt --data data/cifar10.npz --scheme npu-default --verify 500 --int-eval 2000

python examples/walkthrough.py --ckpt runs/resnet20_relu/best.pt --data data/cifar10.npz   # 1~2분짜리 전체 흐름 데모
npuloop intake runs/cust_vit/best.pt --spec edge-10tops-strict --data data/cifar10.npz   # 고객 인테이크 리포트 (--bench 64: 엔진 실측 열)
npuloop bench runs/resnet20_relu/best.pt --data data/cifar10.npz                    # NumPy·C++ 엔진 노드별 wall-clock vs 모델 사이클
npuloop export runs/resnet20_relu/best.pt --data data/cifar10.npz --out model.npuloop --sample 16   # 정수 그래프를 파일로
make runner && build/int8_runner model.npuloop sample_input.f32 --float --argmax    # 파이썬 없이 같은 파일을 실행 (비트 동일)
bash experiments/run_all.sh     # E1–E7 전부 (CPU 4코어 기준 수 시간), results/*.json
python experiments/e9_customer_intake.py    # E9 고객 인테이크 (ViT healing 포함)
python experiments/e8_scalesim.py   # 비용 모델 vs SCALE-Sim (pip install scalesim)
python demo/build.py            # results → demo/index.html, docs/index.html
```

경로는 저장소 기준 `runs/`(체크포인트)와 `data/cifar10.npz`(데이터셋)가 기본값이고, 각각 `NPULOOP_RUNS`·`NPULOOP_DATA` 환경변수로 바꿀 수 있습니다.

파이썬 API 한 줄 요약:

```python
from npuloop.zoo import load_checkpoint, CIFAR10NPZ
from npuloop.graph import trace
from npuloop.npu import estimate
from npuloop.lint import lint
from npuloop.quant import prepare, calibrate, evaluate, QScheme
from npuloop.intengine import export_int_graph, NumpyEngine
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.verify import compare, agreement_table

m = load_checkpoint("runs/resnet20_relu/best.pt"); ds = CIFAR10NPZ("data/cifar10.npz")
print(estimate(trace(m), "edge-10tops").table())                     # simulated cycles
print(lint(trace(m), "edge-10tops", ds.calib_batch(64).numpy()).markdown())
qm = prepare(m, QScheme()); calibrate(qm, [ds.calib_batch(256)])
ig = export_int_graph(qm)                                             # int8/int32/M0,shift
rows, summary = compare(qm, ig, ds.calib_batch(200)); print(agreement_table(rows), summary)
print(NumpyEngine(ig).evaluate(ds, limit=2000), CppEngine(ig).evaluate(ds, limit=2000))
```

## 데모 페이지

`docs/index.html`(= `demo/index.html`)은 위 JSON을 인라인한 정적 페이지입니다. 비용 모델을 JavaScript로 그대로 포팅해서
배열 크기·코어 수·DRAM 대역폭·depthwise 엔진 유무·비-ReLU 활성함수 실행 방식을 바꾸면 레이어별 사이클과 활용률이 즉시 다시 계산됩니다.
lint 리포트, PTQ 그리드와 레이어별 일치도, requant ablation, 수술, 캘리브레이션, 프루닝 Pareto, lint-vs-drop 산점도를 모두 담았습니다.

**라이브 데모:** https://sokldjs554.github.io/npuloop/

같은 페이지가 저장소의 `docs/index.html`에 그대로 들어 있어서, 빌드 없이 어느 정적 호스팅에나 올릴 수 있습니다.

| 호스팅 | 설정 | 주소 |
|---|---|---|
| GitHub Pages | `.github/workflows/pages.yml`이 `docs/`를 push마다 자동 게시 (Settings → Pages의 Source가 GitHub Actions) | https://sokldjs554.github.io/npuloop/ |
| Render | New → Blueprint(저장소의 `render.yaml`) 또는 New → Static Site, publish directory `docs` | `https://<name>.onrender.com` |

둘 다 push할 때마다 자동으로 갱신됩니다.

## 실험 재현

```bash
bash experiments/run_v2.sh       # 베이스라인 5개 학습 (CPU 4코어 기준 수 시간)
bash experiments/run_seeds.sh    # 시드 1·2 복제 학습 (E14용)
bash experiments/run_all.sh      # E1–E7
python experiments/e8_scalesim.py            # 비용 모델 vs SCALE-Sim (pip install scalesim)
python experiments/e9_customer_intake.py     # 고객 인테이크 (ViT healing 포함)
python experiments/e11_tflite_crosscheck.py  # TFLite reference 커널 대조 (pip install tensorflow)
python experiments/e12_imagenette.py         # Imagenette 128x128
python experiments/e13_vela.py               # Arm Vela 대조 (pip install ethos-u-vela tensorflow)
python experiments/e14_rounding_seeds.py     # 반올림 모드 x 시드
python experiments/e15_fidelity.py           # 쌍 표준오차로 잰 fake-quant vs 정수
python experiments/e16_ln_emulation.py       # 정수 LayerNorm 에뮬레이션

make tables                      # results/*.json -> README.md와 docs/EXPERIMENTS.md의 표를 다시 생성
python tools/build_report.py     # docs/report/npuloop_report.{md,html,pdf}
python demo/build.py             # results -> demo/index.html, docs/index.html
```

실험 스크립트는 모두 재개 가능합니다(이미 있는 레코드는 건너뜁니다). 결과 JSON의 모든 레코드에는
`provenance: measured | simulated` 라벨이 붙습니다.
