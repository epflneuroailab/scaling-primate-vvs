export RESULTCACHING_DISABLE="brainscore_vision.model_helpers.brain_transformation.neural.LayerScores._call"
export RESULTCACHING_HOME=~/.result_caching

CKPT_URL="https://epfl-neuroailab-scalinglaws.s3.eu-north-1.amazonaws.com/checkpoints"
RESIZE_SIZE=256
CROP_SIZE=224
INTERPOLATION=bilinear
RUN_NAME=resnet152_imagenet_full
EPOCH=100
CKPT=${CKPT_URL}/${RUN_NAME}/ep${EPOCH}.pt
ARCH=resnet152
NUM_CLASSES=1000
DATASET=imagenet
LOAD_MODEL_EMA=False
LAYER_TYPE=relu
NUM_COMPONENTS=1000
SAVE_DIR='./outputs/layer_scoring/random_proj'

# conda activate brain
python -m src.layer_scoring \
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
    --save-dir $SAVE_DIR \
    --layer-type ${LAYER_TYPE} \
    --disable-pca \
    --enable-random-projection \
    --n-compression-components ${NUM_COMPONENTS} \
    --append-layer-type \

