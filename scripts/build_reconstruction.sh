#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="your base directory" # TODO
BASE_SEED=2025

SUBJECT_IDS=(1 2 3 4 5 6 7 8 9 10)

BRAIN_KEY="eeg"
AVG_TRIALS="true"

SELECTED_CHANNELS='["P7","P5","P3","P1","Pz","P2","P4","P6","P8","PO7","PO3","Oz","PO4","PO8","O1","Oz","O2"]'
PROJ_META='{"CLIP-ViT-H-14-laion2B-s32B-b79K":1024,"CLIP-ViT-B-32-laion2B-s34B-b79K":512,"vae":1024}'
TIME_SLICE='[0,250]'

EMB_DIR="${BASE_DIR}/data/things-eeg/embeddings"
DATA_DIR="${BASE_DIR}/data/things-eeg/Preprocessed_data_250Hz_whiten"

PRIOR_PATH="${BASE_DIR}/pretrained/priors/vae+CLIP-ViT-B-32-laion2B-s34B-b79K+CLIP-ViT-H-14-laion2B-s32B-b79K"

GPU_CFG="configs/gpu_cfg_single.yaml"
CFG_IN="configs/train_clip.yaml"

RUN_ROOT="prior-clip/vae+CLIP-ViT-B-32-laion2B-s34B-b79K+CLIP-ViT-H-14-laion2B-s32B-b79K"
EXP_ROOT="${BASE_DIR}/exp/${RUN_ROOT}"

for subj in "${SUBJECT_IDS[@]}"; do
  seed="${BASE_SEED}"

  run_name="${RUN_ROOT}/subj-${subj}"
  output_dir="${BASE_DIR}/exp-${BRAIN_KEY}/${run_name}"
  TMP_FILE="configs/train_clip.resolved.subj-${subj}.yaml"

  echo "=== [INTRA] subj=${subj} seed=${seed}"
  echo "    output_dir=${output_dir}"

  python3 build_config.py \
    --config_file "${CFG_IN}" \
    --output_file "${TMP_FILE}" \
    "subject_ids=[${subj},]" \
    "eval_subject_ids=[${subj},]" \
    "brain_key=${BRAIN_KEY}" \
    "run_name=${run_name}" \
    "output_dir=${output_dir}" \
    "avg_trials=${AVG_TRIALS}" \
    "seed=${seed}" \
    "selected_channels=${SELECTED_CHANNELS}" \
    "proj_meta=${PROJ_META}" \
    "time_slice=${TIME_SLICE}" \
    "embedding_directory=${EMB_DIR}" \
    "pretrained_model_name_or_path=${PRIOR_PATH}" \
    "data_directory=${DATA_DIR}"

  accelerate launch --config_file "${GPU_CFG}" train_clip.py "${TMP_FILE}"

  rm -rf "${TMP_FILE}" swanlog
done

GPU_CFG="configs/gpu_cfg_all.yaml"
CFG_IN="configs/build_reconstruction.yaml"
export TOKENIZERS_PARALLELISM=false
for subj in "${SUBJECT_IDS[@]}"; do
  seed="${BASE_SEED}"
  run_name="${RUN_ROOT}/subj-${subj}"
  output_dir="${BASE_DIR}/exp-${BRAIN_KEY}/${run_name}"
  TMP_FILE="configs/build_reconstruction.resolved.subj-${subj}.yaml"

  echo "=== [GEN IMAGE] subj=${subj} seed=${seed}"
  echo "    output_dir=${output_dir}"

  python3 build_config.py \
  --config_file "${CFG_IN}" \
  --output_file "${TMP_FILE}" \
  "output_dir=${output_dir}" \
  "seed=${seed}" \
  "time_slice=${TIME_SLICE}" \
  "brain_model_name_or_path=${output_dir}" \
  "prior_model_name_or_path=${PRIOR_PATH}" \
  "eval_subject_ids=[${subj},]" \
  "brain_key=${BRAIN_KEY}" \
  "selected_channels=${SELECTED_CHANNELS}"

  accelerate launch --config_file "${GPU_CFG}" build_reconstruction.py --config "${TMP_FILE}"

  rm -rf "${TMP_FILE}"

done