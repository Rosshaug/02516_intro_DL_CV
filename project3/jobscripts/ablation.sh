#!/bin/sh
# Loss ablation: one array element per run, submitted with  bsub < jobscripts/ablation.sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J "ablation[1-7]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 2:00
#BSUB -oo /zhome/b2/7/204858/Courses/02516/project3/logs/ablation_%I.out
#BSUB -eo /zhome/b2/7/204858/Courses/02516/project3/logs/ablation_%I.err

PH2="--dataset ph2 --model unet --size 256 --epochs 50 --batch_size 6"
DRIVE="--dataset drive --model unet --size 512 --epochs 300 --batch_size 4"

case $LSB_JOBINDEX in
  1) ARGS="$PH2 --loss bce" ;;
  2) ARGS="$PH2 --loss focal" ;;
  3) ARGS="$PH2 --loss wbce" ;;
  4) ARGS="$DRIVE --loss bce --augment" ;;
  5) ARGS="$DRIVE --loss focal --augment" ;;
  6) ARGS="$DRIVE --loss wbce --augment" ;;
  7) ARGS="$DRIVE --loss bce" ;;  # no-augmentation baseline
esac

source $HOME/Courses/02516/.venv/bin/activate
python $HOME/Courses/02516/project3/train.py $ARGS
