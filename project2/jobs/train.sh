#!/bin/sh
# Train all models x seeds on the DTU HPC (LSF).
# Submit from the project2 folder:   mkdir -p logs && bsub < jobs/train.sh
#BSUB -q gpuv100
#BSUB -J video_4_1
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

# Activate the project environment (adjust the path to where your venv lives on the HPC).
source ../.venv/bin/activate

for model in per_frame late_fusion early_fusion cnn3d; do
    for seed in 0 1 2; do
        python src/train.py --model $model --seed $seed
    done
done

python src/summarize.py
