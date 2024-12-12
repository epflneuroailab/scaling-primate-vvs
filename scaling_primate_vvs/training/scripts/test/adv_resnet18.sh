N_GPUS=1
ARCH_FAMILY=resnet
MODEL=resnet18
ADV_CONFIG=cfgs/runai/adversarial/ffgsm_eps-1_alpha-125.yaml
DATASET=imagenet
DATA_PATH=/mnt/scratch/akgokce/datasets/imagenet/
NUM_CLASSES=1000
N_SAMPLES=1
N_SAMPLES_STR=1
BATCH_SIZE=250
SEED=0
NUM_WORKERS=16
DROP_LAST_TRAIN=False

CNFG="cfgs/runai/${ARCH_FAMILY}/${MODEL}.yaml"
RUN_NAME=test
SAVE_DIR='./outputs'

CKPT_PATH=https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints/resnet18_imagenet_full/ep100.pt

source /home/akgokce/.conda/bin/activate brain
cd /mnt/scratch/akgokce/brain/scalingLaws/training
composer -n $N_GPUS -m src.train \
    -c $CNFG \
    --adv-config $ADV_CONFIG \
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
    --checkpoint $CKPT_PATH \
