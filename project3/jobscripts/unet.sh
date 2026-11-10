#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_unet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 2:00
#BSUB -oo /zhome/b2/7/204858/Courses/02516/project3/logs/unet.out
#BSUB -eo /zhome/b2/7/204858/Courses/02516/project3/logs/unet.err

source $HOME/Courses/02516/.venv/bin/activate
python $HOME/Courses/02516/project3/train.py --dataset ph2   --model unet --size 256 --epochs 50  --batch_size 6
python $HOME/Courses/02516/project3/train.py --dataset drive --model unet --size 512 --epochs 300 --batch_size 4
