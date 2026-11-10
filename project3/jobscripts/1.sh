#!/bin/sh
#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J train_ae

#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=2GB]"

#BSUB -W 1:00

#BSUB -oo /zhome/b2/7/204858/Courses/02516/ex3/logs/ae.out
#BSUB -eo /zhome/b2/7/204858/Courses/02516/ex3/logs/ae.err

source $HOME/Courses/02516/.venv/bin/activate
python $HOME/Courses/02516/ex3/train.py --mode multi