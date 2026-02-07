# BrainHiVE: Learning Brain Representation with Hierarchical Visual Embeddings

BrainHIVE is a brain–vision decoding project for brain-to-image retrieval and reconstruction.

We align EEG/MEG embeddings to a single fused visual token built from multiple pretrained visual encoders that cover both high-level semantics (e.g., CLIP) and low-level visual details (e.g., VAE latents). Alignment is trained with a simple contrastive objective.

For reconstruction, we use a pretrained Fusion Prior that maps the fused token into a frozen diffusion backbone (text-free), making generation stable and reusable across different brain encoders under the same fusion-based training scheme.
#  News


# Get Started



# Citation

If you find our project is helpful, please cite our paper as

```
@inproceedings{
anonymous2026learning,
title={Learning Brain Representation with Hierarchical Visual Embeddings},
author={Jiawen Zheng, Haonan Jia, Ming Li, Yuhui Zheng, Yufeng Zeng, Yang Gao and Chen Liang},
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