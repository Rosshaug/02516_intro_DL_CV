#!/bin/sh
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

for model in per_frame late_fusion early_fusion cnn3d; do
    for seed in 0 1 2; do
        python src/train.py --model $model --seed $seed --root_dir /dtu/datasets1/02516/ucf101
    done
done

python src/summarize.py
