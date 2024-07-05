#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:2
#SBATCH --time=06:00:00
#SBATCH --job-name=clip-eval
#SBATCH --output=/home/atuin/g103ea/shared/logs/clip-eval.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source $WORK/venv/bin/activate

export SHARED=/home/atuin/g103ea/shared

python3 eval.py
