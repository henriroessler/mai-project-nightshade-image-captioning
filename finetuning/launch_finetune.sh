#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=clip-poison
#SBATCH --output=/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/logs/frac-100-origpairs.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source ~/miniconda3/bin/activate
conda activate clip_prefix_caption

export SHARED=/home/atuin/g103ea/shared
FRAC=100

python CLIP_prefix_caption/finetune.py -d "/home/atuin/g103ea/shared/embeddings/restval-filtered"\
    -frac $FRAC \
    -pt "/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/pt_models/coco_weights.pt"\
    -o "/home/atuin/g103ea/shared/all_finetuning/frac=$FRAC-origpairs" \
    -pre "frac=$FRAC-origpairs" \
    -E 20 -save 5 -bs 32 \

