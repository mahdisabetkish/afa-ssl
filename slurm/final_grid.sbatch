#!/bin/bash
# Final experiment grid as one submission script (mirrors Makefile targets).
#SBATCH --job-name=afa_ssl_grid
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/raw/grid_%j.out
set -euo pipefail
python slurm/make_queue.py --experiment main --datasets metabric:50,100,200,400,all \
  metabric_semisynth:50,200,all synth_scm:50,200,all uci_rnaseq:50,200,all wdbc:25,100,all \
  --seeds 0 1 2 3 4 --out slurm/jobs/main.jsonl
python slurm/submit.py slurm/jobs/main.jsonl
