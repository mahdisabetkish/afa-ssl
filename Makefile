PY ?= python
export AFA_PY := $(PY)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export TORCH_NUM_THREADS=4
SEEDS ?= 0 1 2 3 4

.PHONY: data test smoke queue-main train eval theory-check reproduce lint clean

data:
	$(PY) scripts/make_data.py

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/run_experiment.py --experiment configs/experiments/smoke.yaml --dataset configs/datasets/synth_scm.yaml --seed 0 --n-labels 50

queue-main:
	$(PY) slurm/make_queue.py --experiment main --datasets metabric:50,100,200,400,all metabric_semisynth:50,200,all synth_scm:50,200,all uci_rnaseq:50,200,all wdbc:25,100,all --seeds $(SEEDS) --out slurm/jobs/main.jsonl

train: queue-main
	$(PY) slurm/submit.py slurm/jobs/main.jsonl

eval:
	$(PY) scripts/aggregate.py
	$(PY) scripts/stats_extra.py
	$(PY) scripts/signal_alignment.py
	$(PY) scripts/heldout_check.py
	$(PY) scripts/robustness_sweep.py
	$(PY) scripts/diversity_check.py
	$(PY) scripts/dropin_check.py
	$(PY) scripts/gbmlgg_check.py
	$(PY) scripts/har_check.py
	$(PY) scripts/dime_budget_check.py
	$(PY) scripts/dependence_tests.py
	$(PY) scripts/practical_effect.py
	$(PY) scripts/make_paper_tables.py
	$(PY) scripts/make_figures.py

theory-check:
	$(PY) scripts/eig_expansion_check.py
	$(PY) scripts/ranking_stability.py

reproduce:
	$(PY) scripts/reproduce.py

lint:
	-$(PY) -m ruff check src scripts tests
	-$(PY) -m black --check src scripts tests

clean:
	rm -rf .pytest_cache src/afa_ssl/__pycache__ tests/__pycache__
