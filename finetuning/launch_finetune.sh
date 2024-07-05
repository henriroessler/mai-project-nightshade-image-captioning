#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:20:00
#SBATCH --job-name=clip-poison
#SBATCH --output=/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/logs/clipcap_coco_finetune.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source ~/miniconda3/bin/activate
conda activate clip_prefix_caption

export SHARED=/home/atuin/g103ea/shared

python CLIP_prefix_caption/finetune.py -pre "10_concept_pairs_coco" --only_prefix -E 20 -bs 32 -pt "/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/pt_models/coco_weights.pt" -o "/home/hpc/g103ea/g103ea14/mai/CLIP_prefix_caption/experiments/logs/test" 