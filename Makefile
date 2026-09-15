.PHONY: test demo tables experiments walkthrough quickstart lint runner paper
DATA ?= data/cifar10.npz
CKPT ?= runs/resnet20_relu/best.pt

test:            ## unit tests (C++ kernels are compiled on first run)
	python -m pytest -q

quickstart:      ## ~1 min tour on the bundled checkpoints + CIFAR-10 sample (no dataset, no training)
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

demo:            ## results/*.json -> demo/index.html + docs/index.html, then the generated tables
	python demo/build.py && python tools/readme_tables.py --inject README.md docs/EXPERIMENTS.md

tables:          ## regenerate only the generated tables (no checkpoints, no dataset)
	python tools/readme_tables.py --inject README.md docs/EXPERIMENTS.md

paper:           ## regenerate every table/figure and build all three PDFs (needs TeX Live)
	python paper/make_tables.py && python paper/make_figs.py
	python paper/make_tables.py --lang ko && python paper/make_figs.py --lang ko
	python paper/make_thesis.py
	cd paper && pdflatex -interaction=nonstopmode npuloop_esl && pdflatex -interaction=nonstopmode npuloop_esl
	cd paper && lualatex -interaction=nonstopmode npuloop_ko && lualatex -interaction=nonstopmode npuloop_ko
	cd paper && for i in 1 2 3; do lualatex -interaction=nonstopmode npuloop_thesis; done

lint:
	ruff check npuloop tests tools experiments examples paper demo/build.py --select F,E9 --ignore F401,F403,F405

runner:  ## standalone C++ executor for .npuloop files (no Python)
	mkdir -p build && g++ -O3 -march=native -std=c++17 -o build/int8_runner npuloop/intengine/cpp/int8_runner.cpp
