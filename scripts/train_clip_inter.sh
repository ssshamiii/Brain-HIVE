#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="your base directory" # TODO
BASE_SEED=2025
N_REPEATS=3

ALL_SUBJECTS=(1 2 3 4 5 6 7 8 9 10)

BRAIN_KEY="eeg"
AVG_TRIALS="true"

SELECTED_CHANNELS='["P7","P5","P3","P1","Pz","P2","P4","P6","P8","PO7","PO3","Oz","PO4","PO8","O1","Oz","O2"]'
PROJ_META='{"synclr_vit_b_16":768,"CLIP-ViT-B-32-laion2B-s34B-b79K":512,"vae":1024}'
TIME_SLICE='[0,250]'

EMB_DIR="${BASE_DIR}/data/things-eeg/embeddings"
DATA_DIR="${BASE_DIR}/data/things-eeg/Preprocessed_data_250Hz_whiten"

GPU_CFG="configs/gpu_cfg_single.yaml"
CFG_IN="configs/train_clip.yaml"

for eval_subj in "${ALL_SUBJECTS[@]}"; do
  # build train list = ALL \ {eval_subj}
  TRAIN_SUBJECTS=()
  for s in "${ALL_SUBJECTS[@]}"; do
    if [[ "${s}" -ne "${eval_subj}" ]]; then
      TRAIN_SUBJECTS+=("${s}")
    fi
  done
  train_csv="$(IFS=,; echo "${TRAIN_SUBJECTS[*]}")"

  for rep in $(seq 1 "${N_REPEATS}"); do
    seed=$((BASE_SEED + rep - 1))
    rep_tag="$(printf "rep-%02d" "${rep}")"

    run_name="RN50+CLIP-ViT-B-32-laion2B-s34B-b79K+vae/inter/heldout-${eval_subj}/train-[${train_csv}]/${rep_tag}"
    output_dir="${BASE_DIR}/exp-${BRAIN_KEY}/${run_name}"
    TMP_FILE="configs/train_clip.resolved.inter.heldout-${eval_subj}.${rep_tag}.yaml"

    echo "=== [INTER-LOO] heldout=${eval_subj} train=[${train_csv}] ${rep_tag}/${N_REPEATS} seed=${seed}"

    python3 build_config.py \
      --config_file "${CFG_IN}" \
      --output_file "${TMP_FILE}" \
      "subject_ids=[${train_csv},]" \
      "eval_subject_ids=[${eval_subj},]" \
      "brain_key=${BRAIN_KEY}" \
      "run_name=${run_name}" \
      "output_dir=${output_dir}" \
      "avg_trials=${AVG_TRIALS}" \
      "seed=${seed}" \
      "load_best_model_at_end=true" \
      "selected_channels=${SELECTED_CHANNELS}" \
      "proj_meta=${PROJ_META}" \
      "time_slice=${TIME_SLICE}" \
      "embedding_directory=${EMB_DIR}" \
      "data_directory=${DATA_DIR}"

    accelerate launch --config_file "${GPU_CFG}" train_clip.py "${TMP_FILE}"
    rm -rf "${TMP_FILE}"
  done
done


# -----------------------------
# Aggregate retrieval metrics (INTER)
# -----------------------------
EXP_ROOT="${BASE_DIR}/exp-${BRAIN_KEY}/RN50+CLIP-ViT-B-32-laion2B-s34B-b79K+vae/inter"
SUMMARY_JSON="${EXP_ROOT}/retrieval_summary_inter.json"
SUMMARY_CSV="${EXP_ROOT}/retrieval_summary_inter.csv"

echo "=== [AGG] Aggregating eval_results.json under: ${EXP_ROOT}"
python3 build_average.py \
  --exp_root "${EXP_ROOT}" \
  --filename "eval_results.json" \
  --out_json "${SUMMARY_JSON}" \
  --out_csv "${SUMMARY_CSV}"

echo "=== [AGG] Saved:"
echo "  - ${SUMMARY_JSON}"
echo "  - ${SUMMARY_CSV}"