N_GPUS=1
RUN_NAME=test
SAVE_DIR='./outputs'

source /home/akgokce/.conda/bin/activate brain
cd /mnt/scratch/akgokce/brain/scalingLaws/training
composer -n $N_GPUS -m src.train \
    -c cfgs/runai/dino/dino_resnet18.yaml \
    --run-name $RUN_NAME \
    --save-dir $SAVE_DIR \
    --ngpus $N_GPUS \
    --lr-monitor \
    --save-overwrite \
    --disable-wandb \
    