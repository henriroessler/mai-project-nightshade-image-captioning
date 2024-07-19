#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=clip-poison
#SBATCH --output=/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/logs/predict-frac-100-switchpairs.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source ~/miniconda3/bin/activate
conda activate clip_prefix_caption

export SHARED=/home/atuin/g103ea/shared
FT_PATH="/home/atuin/g103ea/shared/all_finetuning"
MODEL="frac=100-switchpairs"


python3 CLIP_prefix_caption/predict.py -i "/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/test_subset" \
    -m "$FT_PATH/$MODEL/$MODEL-019.pt" \
    -o "$FT_PATH/$MODEL"

# python3 CLIP_prefix_caption/predict.py -i "/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/test_subset" \
#     -m "$FT_PATH/$MODEL/coco_weights.pt" \
#     -o "$FT_PATH/$MODEL"