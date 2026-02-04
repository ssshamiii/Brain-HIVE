# Brain-HIVE

BrainHIVE is a project to learn brain–image alignment for **brain-to-image retrieval and reconstruction**.

The core idea is to align EEG/MEG embeddings to a **single fused visual token** built from multiple pretrained visual encoders with complementary inductive biases. We use a **contrastive objective** for brain–vision alignment, and a **pretrained Fusion Prior** to bridge the fused token to a frozen diffusion backbone for stable, text-free generation.

Currently, we provide two training stages: Fusion Prior pretraining on large-scale images, and brain-to-fusion alignment on paired brain–image data.