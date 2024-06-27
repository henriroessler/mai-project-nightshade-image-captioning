#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=nightshade
#SBATCH --output=/home/atuin/g103ea/shared/logs/nightshade.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source $WORK/venv/bin/activate

export SHARED=/home/atuin/g103ea/shared

python3 nightshade.py --output-dir $SHARED/nightshade/coco-cat-dog  --clip-cache-dir $SHARED/models/clip --clipcap-model $SHARED/models/clipcap/pretrained_coco.pt --epochs 100 --alpha 10 coco --image-dir $SHARED/coco2014/train --annotation-file $SHARED/coco2014/annotations/instances_train2014.json --original-id 17 --target-id 18 --start-index 100 --num 100
