#!/bin/bash
set -e

# ---------------------------
# Shared launcher settings
# ---------------------------
BASE_DIR="your base directory" # TODO
MODEL_BASE_DIR="${BASE_DIR}/pretrained"

CFG="configs/build_embeddings.yaml"
ACC_CFG="configs/gpu_cfg_all.yaml"
PY="build_embeddings.py"

# Models to run (shared)
MODEL_NAMES=("paulgavrikov/synclr-vit-base-patch16-224/synclr_vit_b_16.pth" "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" "stabilityai/stable-diffusion-xl-base-1.0/vae" "open-clip/RN50.pt" "facebook/dinov2-base")

# Batch sizes (shared)
VAE_BATCH_SIZE=100
NONVAE_BATCH_SIZE=1000
RESOLUTIONS=(128)   # only used if MODEL_PATH contains "vae"


# ==========================================================
# Part 1) THINGS embeddings (train + test)
# ==========================================================
DATA_KEY="things-eeg"   # things-eeg, things-meg
export DATASET_NAME="things"
export OUTPUT_DIR="${BASE_DIR}/data/${DATA_KEY}/embeddings"
export IMAGE_DIR="${BASE_DIR}/data/${DATA_KEY}"

for name in "${MODEL_NAMES[@]}"; do
  export MODEL_PATH="${MODEL_BASE_DIR}/${name}"

  if [[ "$MODEL_PATH" == *"vae"* ]]; then
    export BATCH_SIZE="${VAE_BATCH_SIZE}"
    for r in "${RESOLUTIONS[@]}"; do
      export RESOLUTION="${r}"

      export SPLIT="train"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

      export SPLIT="test"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
    done
  else
    export BATCH_SIZE="${NONVAE_BATCH_SIZE}"
    unset RESOLUTION  # use yaml default (128) if needed

    export SPLIT="train"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

    export SPLIT="test"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
  fi
done


# ==========================================================
# Part 2) ImageNet-1K embeddings (train + validation)
# ==========================================================
export DATASET_NAME="imagenet-1k"
export OUTPUT_DIR="${BASE_DIR}/data/visual-layer/imagenet-1k-vl-enriched/embeddings"
export IMAGE_DIR="${BASE_DIR}/data/visual-layer/imagenet-1k-vl-enriched/data"

for name in "${MODEL_NAMES[@]}"; do
  export MODEL_PATH="${MODEL_BASE_DIR}/${name}"

  if [[ "$MODEL_PATH" == *"vae"* ]]; then
    export BATCH_SIZE="${VAE_BATCH_SIZE}"
    for r in "${RESOLUTIONS[@]}"; do
      export RESOLUTION="${r}"

      export SPLIT="train"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

      export SPLIT="validation"
      accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
    done
  else
    export BATCH_SIZE="${NONVAE_BATCH_SIZE}"
    unset RESOLUTION

    export SPLIT="train"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"

    export SPLIT="validation"
    accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
  fi
done
