export RESULTCACHING_DISABLE="brainscore_vision.model_helpers"
export RESULTCACHING_HOME=~/.result_caching

CKPT_URL="https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints"
RESIZE_SIZE=256
CROP_SIZE=224
INTERPOLATION=bilinear
RUN_NAME=convnext_tiny_ecoset_full_seed-0
EPOCH=300
CKPT=${CKPT_URL}/${RUN_NAME}/ep${EPOCH}.pt
ARCH=convnext_tiny
NUM_CLASSES=565
DATASET=ecoset
LOAD_MODEL_EMA=True
SAVE_DIR='./outputs/test/benhmark_layers'
LAYER=features.7.2.block.2

# conda activate brain
python -m src.benchmark_layers \
    --checkpoint=$CKPT \
    --arch=$ARCH \
    --num-classes=$NUM_CLASSES \
    --run-name=$RUN_NAME \
    --training-dataset $DATASET \
    --crop-size ${CROP_SIZE} \
    --resize-size ${RESIZE_SIZE} \
    --interpolation ${INTERPOLATION} \
    --ckpt-src full \
    --benchmark-layer $LAYER \
    --load-model-ema $LOAD_MODEL_EMA \