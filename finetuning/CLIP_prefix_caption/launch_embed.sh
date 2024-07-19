#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:05:00
#SBATCH --job-name=clip-poison
#SBATCH --output=/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/logs/clip-poison-wine_glass_traffic_light.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source ~/miniconda3/bin/activate
conda activate clip_prefix_caption

export SHARED=/home/atuin/g103ea/shared

CONCEPT_PAIRS=wine_glass_traffic_light

python3 CLIP_prefix_caption/parse_images.py -f "/home/atuin/g103ea/shared/nightshade/coco-2014-poisoning/results_$CONCEPT_PAIRS.csv"