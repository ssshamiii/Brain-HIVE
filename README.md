# BrainHiVE: Learning Brain Representation with Hierarchical Visual Embeddings

BrainHIVE is a brain–vision decoding project for brain-to-image retrieval and reconstruction.

We align EEG/MEG embeddings to a single fused visual token built from multiple pretrained visual encoders that cover both high-level semantics (e.g., CLIP) and low-level visual details (e.g., VAE latents). Alignment is trained with a simple contrastive objective.

For reconstruction, we use a pretrained Fusion Prior that maps the fused token into a frozen diffusion backbone (text-free), making generation stable and reusable across different brain encoders under the same fusion-based training scheme.
#  News


# Get Started

This repo provides three main stages:

1) **Build visual embeddings**
2) **Train brain↔vision contrastive model**
3) **Train fusion prior and build reconstruction**

## 0. Environment

```bash
git clone git@github.com:ssshamiii/Brain-HIVE.git
cd Brain-HIVE-main

conda create -n brainhive python=3.13 -y
conda activate brainhive

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## 1. Prepare data & pretrained weights

### Datasets

The scripts assume the following layout:

```text
<BASE_DIR>/
  data/
    things-eeg/
      Preprocessed_data_250Hz_whiten/   # EEG/MEG numpy / mat files
      ...                               # THINGS images live under this folder
      embeddings/                       # output from build_embeddings
    visual-layer/
      imagenet-1k-vl-enriched/
        data/                           # ImageNet images
        embeddings/                     # output from build_embeddings
```
Data sources:
  - EEG: [things-eeg](https://huggingface.co/datasets/Haitao999/things-eeg)
  - MEG: [things-meg](https://huggingface.co/datasets/Haitao999/things-eeg-meg)
  - ImageNet: [imagenet-1k-vl-enriched](https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched)
### Pretrained diffusion (SDXL) + IP-Adapter weights

Fusion Prior training requires:

- **SDXL base** (HuggingFace ID or local directory): `stabilityai/stable-diffusion-xl-base-1.0`
- **SDXL IP-Adapter weights**: a single `*.safetensors` file: `h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.safetensors`  (path is set in `scripts/train_prior.sh`)

We recommend storing them under:

```text
<BASE_DIR>/pretrained/
  ip-adapter_sdxl.safetensors
  priors/              # output of train_prior.sh
```

### Visual encoders

`build_embeddings.sh` can run multiple pretrained vision encoders (CLIP, DINOv2, VAE, RN50, SynCLR). By default it looks under:

```text
<BASE_DIR>/pretrained/<model_name_or_path>
```

If you prefer to load from HuggingFace directly, set `MODEL_PATH` in `scripts/build_embeddings.sh` to the HF IDs instead of local files.

## 2. Build visual embeddings

```bash
bash scripts/build_embeddings.sh
```

## 3. Train brain↔vision contrastive model

```bash
# Intra-subject
bash scripts/train_clip_intra.sh

# Inter-subject
bash scripts/train_clip_inter.sh
```

## 4. Train fusion prior and build reconstruction

```bash
bash scripts/train_prior.sh
```

> **Note:** You can also directly use pretrained Fusion Priors from Hugging Face: 

> https://huggingface.co/fakekungfu/Fusion-Prior-H14_B32_VAE

> https://huggingface.co/fakekungfu/Fusion-Prior-SynCLR_B32_VAE

Reconstruction uses a trained Fusion Prior.

```bash
bash scripts/build_reconstruction.sh
```

# Citation

If you find our project is helpful, please cite our paper as

```
@inproceedings{
zheng2026learning,
title={Learning Brain Representation with Hierarchical Visual Embeddings},
author={Jiawen Zheng and Haonan Jia and MING LI and Yuhui Zheng and Yufeng Zeng and Yang Gao and Chen Liang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=IEq71qS8B7}
}
```

# Related Work

Similar works...

[Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://arxiv.org/abs/2503.04207)

[Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba5f1233efa77787ff9ec015877dbd1f-Abstract-Conference.html)

[MB2C: Multimodal Bidirectional Cycle Consistency for Learning Robust Visual Neural Representations](https://openreview.net/pdf/190f8066abcf7146c3fa91dbf2aaca37d329c67f.pdf)