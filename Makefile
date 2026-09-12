.PHONY: test demo experiments walkthrough quickstart lint runner
DATA ?= data/cifar10.npz
CKPT ?= runs/resnet20_relu/best.pt

test:            ## unit tests (C++ kernels are compiled on first run)
	python -m pytest -q

quickstart:      ## 40 s tour on the bundled checkpoints + CIFAR-10 sample (no dataset, no training)
	python -m npuloop intake examples/quickstart/cust_inception.pt --data examples/quickstart/cifar10_sample.npz --spec edge-10tops
	python -m npuloop quantize examples/quickstart/resnet20_relu.pt --data examples/quickstart/cifar10_sample.npz --verify 200 --int-eval 500
	mkdir -p build && python -m npuloop export examples/quickstart/resnet20_relu.pt --data examples/quickstart/cifar10_sample.npz --out build/quickstart.npuloop --sample 8 --sample-path build/quickstart_input.f32
	$(MAKE) runner && build/int8_runner build/quickstart.npuloop build/quickstart_input.f32 --float --argmax

walkthrough:     ## 1-2 min end-to-end run on one checkpoint
	python examples/walkthrough.py --ckpt $(CKPT) --data $(DATA)

experiments:     ## E1-E7 (hours on CPU); resumable, skips finished configs (E8 = make scalesim)
	NPULOOP_DATA=$(DATA) bash experiments/run_all.sh

scalesim:        ## E8 cost-model validation (pip install scalesim)
	NPULOOP_DATA=$(DATA) python experiments/e8_scalesim.py

demo:            ## results/*.json -> demo/index.html + docs/index.html, then README tables
	python demo/build.py && python tools/readme_tables.py --inject README.md

lint:
	ruff check npuloop tests tools experiments examples demo/build.py --select F,E9 --ignore F401,F403,F405

runner:  ## standalone C++ executor for .npuloop files (no Python)
	mkdir -p build && g++ -O3 -march=native -std=c++17 -o build/int8_runner npuloop/intengine/cpp/int8_runner.cpp
