## BrainHiVE: Learning Brain Representation with Hierarchical Visual Embeddings

BrainHIVE is a brain–vision decoding project for brain-to-image retrieval and reconstruction.

We align EEG/MEG embeddings to a single fused visual token built from multiple pretrained visual encoders that cover both high-level semantics (e.g., CLIP) and low-level visual details (e.g., VAE latents). Alignment is trained with a simple contrastive objective.

For reconstruction, we use a pretrained Fusion Prior that maps the fused token into a frozen diffusion backbone (text-free), making generation stable and reusable across different brain encoders under the same fusion-based training scheme.
##  News


## Get Started



## Citation

If you find our project is helpful, please cite our paper as

```
@inproceedings{
anonymous2026learning,
title={Learning Brain Representation with Hierachical Visual Embeddings},
author={Anonymous},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=IEq71qS8B7}
}
```