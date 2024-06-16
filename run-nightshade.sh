#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=nightshade
#SBATCH --output=/home/atuin/g103ea/g103ea10/nightshade.log
#SBATCH --export=None

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load cuda/12.1.1
module load cudnn/8.9.6.50-12.x

source venv/bin/activate

python3 nightshade.py --output-dir /home/atuin/g103ea/shared/nightshade-images  --clip-cache-dir $HPCVAULT/cache/dir coco --image-dir /home/atuin/g103ea/shared/coco2014/train --annotation-file /home/atuin/g103ea/shared/coco2014/annotations/instances_train2014.json --original-id 17 --target-id 18 --num 1000
