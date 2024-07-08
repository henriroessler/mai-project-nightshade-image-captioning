#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=clip-poison
#SBATCH --output=/home/atuin/g103ea/shared/logs/embed.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source $WORK/venv/bin/activate

export SHARED=/home/atuin/g103ea/shared

python3 CLIP_prefix_caption/parse_images.py -f "$SHARED/nightshade/coco-2014-poisoning-filtered/*.csv" --types target --outdir "$SHARED/target_embeddings/restval-filtered" --captions_file "$SHARED/coco2014/annotations/captions_all2014.json"