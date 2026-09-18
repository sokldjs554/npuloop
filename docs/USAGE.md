# 사용법

[← README](../README.md) · [실험 E1–E20](EXPERIMENTS.md) · [설계 문서](DESIGN.md)

번들 체크포인트 검증, 새 모델 실험, 데이터 준비와 전체 실험 재현을 구분합니다. 모델 구조 수정부터 학습·PTQ·정수 평가까지 한 번에 실행하는 방법은 [STUDY_RUNNER.md](STUDY_RUNNER.md)에 있습니다.

## 설치와 전체 명령

```bash
pip install -e .[dev]           # torch(CPU), numpy, pytest
make quickstart                 # 데이터·학습 없이 40초: 번들 체크포인트로 intake → quantize/verify → export + C++ runner
python -m pytest -q             # 전체 회귀 테스트 (C++ 커널은 첫 실행 때 g++로 컴파일)

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

`docs/index.html`(= `demo/index.html`)은 모델 실험·경량화 워크벤치입니다. 서버나 외부 자산 없이 기존 실험의 모델 변경·회복 학습·양자화·프루닝 결과를 비교합니다. 실제로 기록된 학습 곡선만 표시하고, 정확도와 추정 사이클 조건을 적용해 후보를 비교합니다. Python 실행기의 새 `study.json`을 가져올 수도 있습니다.

NPU 분석 화면에서는 배열 크기·코어 수·DRAM 대역폭 등의 조건을 바꾸면 추정 사이클을 다시 계산합니다. 정수 검증과 E1–E20 원본 조회는 별도 작업 영역입니다. [전체 화면 안내](DEMO_GUIDE.md)를 참고하세요.

**공개 데모:** [npuloop Workbench](https://sokldjs554.github.io/npuloop/)

`.github/workflows/pages.yml`이 master의 `docs/` 변경을 GitHub Pages에 게시합니다. Render 설정 파일도 있지만 실제 Render 배포가 운영 중이라는 뜻은 아닙니다.

체크포인트가 없는 환경에서 화면만 다시 만들 때는 다음 명령을 사용합니다. 임의 초기화 가중치의 정확도를 계산하지 않고 저장된 실험 결과를 표시합니다.

```bash
python demo/build.py --from-configs demo/model_configs.json
```

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
python experiments/e17_dense_output.py       # 초해상도 출력·PSNR 비교

make tables                      # results/*.json -> README.md와 docs/EXPERIMENTS.md의 표를 다시 생성
python tools/build_report.py     # docs/report/npuloop_report.{md,html,pdf}
python demo/build.py             # results -> demo/index.html, docs/index.html
```

실험 스크립트는 모두 재개 가능합니다(이미 있는 레코드는 건너뜁니다). 결과 JSON의 모든 레코드에는
`provenance: measured | simulated` 라벨이 붙습니다.
