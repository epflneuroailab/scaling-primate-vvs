# conda activate brain

RESULTS_CSV=../../results/benchmark_scores_local.csv
OUTPUT_DIR=./fitting_results
ARTIFACT_DIR=./fitting_artifacts_region
NUM_WORKERS=8
NUM_BOOTSTRAPS=100

ALL_CONFIGS=()
DATASETS=("imagenet21kP" "inaturalist" "places365" "webvisionP")
for ds in "${DATASETS[@]}"; do
    ALL_CONFIGS+=("configs/sample/${ds}_scaling.yaml")
done


for config in "${ALL_CONFIGS[@]}"; do
    python -m src.start_fitting \
        --experiment-config $config \
        --results-csv $RESULTS_CSV \
        --output-dir $OUTPUT_DIR \
        --artifact-dir $ARTIFACT_DIR \
        --num-workers $NUM_WORKERS \
        --num-bootstraps $NUM_BOOTSTRAPS \
        # --overwrite
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