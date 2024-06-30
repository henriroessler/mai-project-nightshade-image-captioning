#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=nightshade
#SBATCH --output=/home/atuin/g103ea/shared/logs/nightshade.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source $WORK/venv/bin/activate

export SHARED=/home/atuin/g103ea/shared

python3 nightshade.py --output-dir $SHARED/nightshade/coco-horse-elephant  --clip-cache-dir $SHARED/models/clip --clipcap-model $SHARED/models/clipcap/pretrained_coco.pt --epochs 200 --alpha 2 --lr 0.01 --p 0.07 coco --image-dir $SHARED/coco2014/train --annotation-file $SHARED/coco2014/annotations/instances_train2014.json --captions-file $SHARED/coco2014/annotations/captions_train2014.json --original-id 19 --target-id 22 --start-index 0 --num 100
