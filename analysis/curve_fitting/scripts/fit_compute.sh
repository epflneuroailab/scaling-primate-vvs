# conda activate brain

RESULTS_CSV=../../results/benchmark_scores.csv
OUTPUT_DIR=./fitting_results
ARTIFACT_DIR=./fitting_artifacts
NUM_WORKERS=8

ALL_CONFIGS=(
    "configs/compute/compute_neuro.yaml"
    "configs/compute/compute_behavior.yaml"

    # "configs/compute/chinchilla.yaml"
)


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