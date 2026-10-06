#!/bin/bash
#SBATCH --job-name=afa_ssl
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --output=logs/raw/slurm_%A_%a.out
# Usage: sbatch --array=0-N%4 slurm/run_array.sbatch slurm/jobs/<queue>.jsonl
set -euo pipefail
QUEUE=$1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
       VECLIB_MAXIMUM_THREADS=4 TORCH_NUM_THREADS=4
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$QUEUE")
CMD=$(python -c "import json,sys; print(' '.join(json.loads(sys.argv[1])['cmd']))" "$LINE")
echo "[$(date)] task $SLURM_ARRAY_TASK_ID: python $CMD"
python $CMD
