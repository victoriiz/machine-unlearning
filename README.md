# Machine Unlearning — TOFU Benchmark

Experiments on the [TOFU (Task of Fictitious Unlearning)](https://github.com/locuslab/open-unlearning) benchmark using the open-unlearning framework.

## Structure
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `scripts/` — Slurm job scripts for cluster training

## Setup
```bash
conda activate tofu_env
cd open-unlearning
```

## Running an experiment
```bash
sbatch scripts/tofu_unlearn.sh
```
