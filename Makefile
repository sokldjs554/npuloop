.PHONY: test demo experiments walkthrough lint
DATA ?= data/cifar10.npz

test:            ## unit tests (C++ kernels are compiled on first run)
	python -m pytest -q

walkthrough:     ## 1-2 min end-to-end run on one checkpoint
	python examples/walkthrough.py --data $(DATA)

experiments:     ## E1-E7 (hours on CPU); resumable, skips finished configs
	NPULOOP_DATA=$(DATA) bash experiments/run_all.sh

scalesim:        ## E8 cost-model validation (pip install scalesim)
	NPULOOP_DATA=$(DATA) python experiments/e8_scalesim.py

demo:            ## results/*.json -> demo/index.html + docs/index.html, then README tables
	python demo/build.py && python tools/readme_tables.py --inject README.md

lint:
	ruff check npuloop tests tools demo/build.py --select F,E9 --ignore F401,F403,F405
