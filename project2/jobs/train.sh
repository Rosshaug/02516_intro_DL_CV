#!/bin/bash
#BSUB -q c02516
#BSUB -J video
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

source ../.venv/bin/activate

for seed in 0 1 2; do
    python src/train.py --model dual_stream --seed $seed
done

python src/summarize.py
