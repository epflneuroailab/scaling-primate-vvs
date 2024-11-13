N_GPUS=1
ARCH_FAMILY=resnet
MODEL=resnet18
DATASET=imagenet
NUM_CLASSES=1000
N_SAMPLES=1
N_SAMPLES_STR=1
DATA_PATH=/mnt/scratch/akgokce/datasets/imagenet/
BATCH_SIZE=250
SEED=0
NUM_WORKERS=16
DROP_LAST_TRAIN=False

CNFG="cfgs/runai/simclr/${ARCH_FAMILY}/${MODEL}.yaml"
RUN_NAME=test
SAVE_DIR='./outputs'

CKPT_PATH="/mnt/scratch/akgokce/brain/outputs_simclr/${RUN_NAME}/latest-rank0.pt"

source /home/akgokce/.conda/bin/activate brain
cd /mnt/scratch/akgokce/brain/scalingLaws/training
composer -n $N_GPUS -m src.train \
    -c $CNFG \
    --run-name $RUN_NAME \
    --save-dir $SAVE_DIR \
    --train-samples ${N_SAMPLES} \
    --seed $SEED \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --num-classes $NUM_CLASSES \
    --workers $NUM_WORKERS \
    --drop-last-train $DROP_LAST_TRAIN \
    --ngpus $N_GPUS \
    --batch-size $BATCH_SIZE \
    --lr-monitor \
    --disable-wandb \
