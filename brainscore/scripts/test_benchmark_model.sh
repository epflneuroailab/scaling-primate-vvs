export RESULTCACHING_DISABLE="brainscore_vision.model_helpers"
export RESULTCACHING_HOME=~/.result_caching

CKPT_URL="https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints"
RESIZE_SIZE=256
CROP_SIZE=224
INTERPOLATION=bilinear
RUN_NAME=resnet18_imagenet_full
EPOCH=100
CKPT=${CKPT_URL}/${RUN_NAME}/ep${EPOCH}.pt
ARCH=resnet18
NUM_CLASSES=1000
DATASET=imagenet
LOAD_MODEL_EMA=False
SAVE_DIR='./outputs/test/benhmark_layers'

# conda activate brain
python -m src.benchmark_model \
    --checkpoint=$CKPT \
    --arch=$ARCH \
    --num-classes=$NUM_CLASSES \
    --run-name=$RUN_NAME \
    --training-dataset $DATASET \
    --crop-size ${CROP_SIZE} \
    --resize-size ${RESIZE_SIZE} \
    --interpolation ${INTERPOLATION} \
    --ckpt-src full \
    --load-model-ema $LOAD_MODEL_EMA \
    --save-dir $SAVE_DIR
