#!/bin/sh
# Point-supervision ablation on PH2 (part 2): sampling strategy x number of clicks x seed,
# plus the fully supervised baseline with the same seeds.
# Submit from the project3 directory:   mkdir -p logs && bsub -app c02516_1g.10gb < jobscripts/weak.sh
# Then aggregate:                        python summarize.py
#
# Index layout (1-based): 1..NFULL = full masks, then strategy-major / clicks / seed.
# 3 strategies x 4 click counts x 3 seeds + 3 full = 39 jobs. Keep the array range in sync.
#BSUB -q c02516
#BSUB -J "weak[1-39]%8"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 1:00
#BSUB -oo logs/weak_%I.out
#BSUB -eo logs/weak_%I.err

# Same hyperparameters as the fully supervised PH2 UNet from part 1
PH2="--dataset ph2 --model unet --size 256 --epochs 50 --batch_size 6 --lr 1e-3 --loss bce"
STRATEGIES="random interior boundary"
CLICKS="1 3 10 30"
SEEDS="0 1 2"
NSEEDS=3
NCLICKS=4

i=$((LSB_JOBINDEX - 1))
if [ $i -lt $NSEEDS ]; then
  ARGS="$PH2 --seed $i"
else
  i=$((i - NSEEDS))
  seed=$((i % NSEEDS)); i=$((i / NSEEDS))
  c=$((i % NCLICKS));   s=$((i / NCLICKS))
  clicks=$(echo $CLICKS | cut -d' ' -f$((c + 1)))
  strategy=$(echo $STRATEGIES | cut -d' ' -f$((s + 1)))
  ARGS="$PH2 --clicks $clicks --sampling $strategy --seed $seed"
fi

echo "Job $LSB_JOBINDEX: $ARGS"
../.venv/bin/python train.py $ARGS
