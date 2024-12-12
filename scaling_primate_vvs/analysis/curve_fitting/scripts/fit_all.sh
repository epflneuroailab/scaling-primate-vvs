# conda activate brain

RESULTS_CSV=../../artifacts/benchmark_scores.csv
OUTPUT_DIR=./outputs/fitting_results
ARTIFACT_DIR=./outputs/fitting_artifacts
NUM_WORKERS=8

ALL_CONFIGS=(
    "configs/compute/compute_neuro.yaml"
    "configs/compute/compute_behavior.yaml"
    "configs/compute/chinchilla.yaml"

    "configs/parameter/parameter_scaling.yaml"
    "configs/parameter/parameter_pretrained_scaling.yaml"
)

DATASETS=("imagenet" "ecoset" "imagenet21kP" "inaturalist" "places365" "webvisionP" "infimnist")
for ds in "${DATASETS[@]}"; do
    ALL_CONFIGS+=("configs/sample/${ds}_scaling.yaml")
done

MODELS=("resnet" "efficientnet" "convnext" "vit")
TYPES=("avg" "behavior" "neuro")
for model in "${MODELS[@]}"; do
    for type in "${TYPES[@]}"; do
        ALL_CONFIGS+=("configs/model/${model}/${model}_${type}.yaml")
    done
done

REGIONS=("v1" "v2" "v4" "it" "behavior" "avg")
TYPES=("group1" "group2")
for region in "${REGIONS[@]}"; do
    for type in "${TYPES[@]}"; do
        ALL_CONFIGS+=("configs/region/${region}/${region}_${type}.yaml")
    done
done


REGIONS=("v1" "v2" "v4" "it" "behavior" "avg" "neuro")
for region in "${REGIONS[@]}"; do
    ALL_CONFIGS+=("configs/ssl/simclr/simclr_${region}.yaml")
done


for config in "${ALL_CONFIGS[@]}"; do
    python -m src.start_fitting \
        --experiment-config $config \
        --results-csv $RESULTS_CSV \
        --output-dir $OUTPUT_DIR \
        --artifact-dir $ARTIFACT_DIR \
        --num-workers $NUM_WORKERS \
        --overwrite
done


# for config in "${ALL_CONFIGS[@]}"; do
#     echo "progress" >&3
#     python -m src.start_fitting \
#         --experiment-config $config \
#         --results-csv $RESULTS_CSV \
#         --output-dir $OUTPUT_DIR \
#         --artifact-dir $ARTIFACT_DIR \
#         --num-bootstraps $NUM_BOOTSTRAPS \
#         --num-workers 4 \
#         --overwrite
# done 3>&1 >/dev/null | tqdm --total ${#ALL_CONFIGS[@]} --null;