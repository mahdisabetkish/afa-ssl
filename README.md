# Label-efficient active feature acquisition with a self-supervised pretrained predictor

Code for *"Self-supervised pretraining of the predictor for label-efficient active
feature acquisition: application to cancer gene expression"* by Mahdi Sabet Kish and
Mehran Yazdi (under review).

The method buys features one at a time for each record under a cost budget, scoring
candidates by expected information gain. The predictor mixes a classifier fine-tuned
from a masked autoencoder pretrained on unlabeled rows with a classifier trained from
scratch; the feature model is a shrinkage Gaussian fitted to the labeled rows.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
make test     # unit tests
make smoke    # short end-to-end run on synthetic data
```

To reproduce the paper: download the data as described in `data/README.md`, run
`make data`, run the job queues in `slurm/jobs/` with
`python slurm/local_runner.py <queue>.jsonl`, then `make eval` (tables and figures go
to `paper/`). The proposed method is `ours_supcond` in `src/afa_ssl/experiment.py`.

## License

MIT, see `LICENSE`.
