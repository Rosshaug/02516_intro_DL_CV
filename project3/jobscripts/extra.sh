#!/bin/sh
# Extra runs, 3 seeds each:
#   1-3   encoder-decoder on PH2   (part 1, required baseline)
#   4-6   encoder-decoder on DRIVE (part 1, required baseline)
#   7-9   UNet on DRIVE, BCE, no augmentation (3-seed baseline under the same evaluation)
#   10-21 asymmetric clicks on PH2, random sampling: 1+10, 10+1, 3+30, 30+3 (part 2)
# Submit from the project3 directory:   mkdir -p logs && bsub -app c02516_1g.10gb < jobscripts/extra.sh
#BSUB -q c02516
#BSUB -J "extra[1-21]%8"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -oo logs/extra_%I.out
#BSUB -eo logs/extra_%I.err

# Same hyperparameters as part 1 (EncDec only accepts 128x128 inputs)
PH2="--dataset ph2 --epochs 50 --batch_size 6 --lr 1e-3 --loss bce"
DRIVE="--dataset drive --epochs 300 --batch_size 4 --lr 1e-3 --loss bce"

i=$((LSB_JOBINDEX - 1))
seed=$((i % 3))
case $((i / 3)) in
  0) ARGS="$PH2 --model encdec --size 128" ;;
  1) ARGS="$DRIVE --model encdec --size 128" ;;
  2) ARGS="$DRIVE --model unet --size 512" ;;
  3) ARGS="$PH2 --model unet --size 256 --clicks 1 --neg_clicks 10" ;;
  4) ARGS="$PH2 --model unet --size 256 --clicks 10 --neg_clicks 1" ;;
  5) ARGS="$PH2 --model unet --size 256 --clicks 3 --neg_clicks 30" ;;
  6) ARGS="$PH2 --model unet --size 256 --clicks 30 --neg_clicks 3" ;;
esac
ARGS="$ARGS --seed $seed"

echo "Job $LSB_JOBINDEX: $ARGS"
../.venv/bin/python train.py $ARGS
