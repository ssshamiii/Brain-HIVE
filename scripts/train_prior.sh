#!/usr/bin/env bash
set -euo pipefail

# ---- edit here ----
SELECTED_KEYS=(
#   "RN50"
  "vae"
  "CLIP-ViT-B-32-laion2B-s34B-b79K"
  "CLIP-ViT-H-14-laion2B-s32B-b79K"
  # "synclr_vit_b_16"
  # "dinov2-base"
)
# -------------------

TMP_FILE="configs/train_prior.resolved.yaml"
ROOT_OUT="your base directory" # TODO

declare -A DIM=(
  ["CLIP-ViT-B-32-laion2B-s34B-b79K"]=512
  ["RN50"]=1024
  ["vae"]=1024
  ["CLIP-ViT-H-14-laion2B-s32B-b79K"]=1024
  ["synclr_vit_b_16"]=768
  ["dinov2-base"]=768
)

# Validate keys
for k in "${SELECTED_KEYS[@]}"; do
  [[ -n "${DIM[$k]+x}" ]] || { echo "Unknown key: $k" >&2; exit 2; }
done

# Join for output_dir
suffix="$(IFS=+; echo "${SELECTED_KEYS[*]}")"
OUT_DIR="${ROOT_OUT}/priors/${suffix}"

# Build proj_meta: {k1: d1, k2: d2, ...}
proj_meta="{"
first=1
for k in "${SELECTED_KEYS[@]}"; do
  if [[ $first -eq 1 ]]; then
    proj_meta+="${k}: ${DIM[$k]}"
    first=0
  else
    proj_meta+=", ${k}: ${DIM[$k]}"
  fi
done
proj_meta+="}"

echo "TMP_FILE  : ${TMP_FILE}"
echo "output_dir: ${OUT_DIR}"
echo "proj_meta : ${proj_meta}"

python3 build_config.py \
  --config_file configs/train_prior.yaml \
  --output_file "${TMP_FILE}" \
  "output_dir=${OUT_DIR}" \
  "proj_meta=${proj_meta}"

accelerate launch --config_file configs/gpu_cfg_all.yaml train_prior.py --config ${TMP_FILE}
rm -rf ${TMP_FILE}