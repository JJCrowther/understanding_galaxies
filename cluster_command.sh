#!/bin/bash
#SBATCH --job-name=understand                       # Job name
#SBATCH --output=understand_%A.log 
#SBATCH --mem=32gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node

pwd; hostname; date

nvidia-smi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64

ZOOBOT_DIR=/share/nas/walml/repos/understanding_galaxies
PYTHON=/share/nas/walml/miniconda3/envs/zoobot/bin/python

FITS_DIR=/share/nas/walml/galaxy_zoo/decals/dr5/fits_native/J000

$PYTHON $ZOOBOT_DIR/creating_image_main.py \
    --fits-dir $FITS_DIR \
    --scale-factor 2 
    
$PYTHON $ZOOBOT_DIR/make_predictions.py \
    --batch-size 128 
