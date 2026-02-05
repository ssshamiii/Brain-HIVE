<div align="center">
  <h1>
    <img src="assets/logo.png" alt="Logo" width="32" style="vertical-align: middle; margin-right: 5px;" />
    BrainHiVE: Learning Brain Representation with Hierarchical Visual Embeddings (ICLR 2026)
  </h1>
</div>
[![arXiv](https://img.shields.io/badge/arXiv-2507.06448-b31b1b.svg)](https://arxiv.org/abs/2507.06448)
[![GitHub](https://img.shields.io/badge/💻%20GitHub-Code-green)](https://github.com/ssshamiii/Brain-HIVE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/PAPOGalaxy/papo-qwen-686d92dd3d43b1ce698f851a)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Data-yellow)](https://huggingface.co/datasets/<your-dataset>)

## 🔥 News

- [x] **[2026/1/26]** Our paper has been accepted at **ICLR 2026!!!** 🎉🎉🎉 Thanks to our co-authors, the Senior Program Committee, area chairs, and reviewers for their invaluable support.
- [ ] **[2026/2/]** We've released our paper: https://arxiv.org/abs/.
- [ ] **[2026/2/]** We've released our code.

## 🌟 Highlights

- We propose a fusion-based brain-vision interface that aligns brain embeddings to a fused visual token built from hierarchical encoders (semantics and pixels), together with a pretrained Fusion Prior that bridges this token to a frozen image generation backbone in a reusable, text-free way.
- We provide a key scientific finding: EEG/MEG signals jointly align with high-level semantics and low-level visual details. Within our fusion framework, adding a VAE latent on top of semantic encoders consistently boosts decoding performance, whereas semantics-only or pixels-only settings cannot recover the same brain--vision structure.
- Our method achieves state-of-the-art zero-shot retrieval and improved reconstruction quality, while remaining plug-and-play across different brain encoders under a fixed fusion-based training scheme.

## 📖 **Methodology**

BrainHIVE is a method to learn brain–image alignment for **brain-to-image retrieval and reconstruction**.

The core idea is to align EEG/MEG embeddings to a **single fused visual token** built from multiple pretrained visual encoders with complementary inductive biases. We use a **contrastive objective** for brain–vision alignment, and a **pretrained Fusion Prior** to bridge the fused token to a frozen diffusion backbone for stable, text-free generation.

Currently, we provide two training stages: Fusion Prior pretraining on large-scale images, and brain-to-fusion alignment on paired brain–image data.

<div align="center">
  <img src="assets/method.png" alt="BrainHiVE Method Illustration" width="100%" />
  <p><em>Figure: Overview of the BrainHiVE Method.</em></p>
</div>

## 📊 **Data**

We adapt multiple multimodel reasoning benchmarks to construct our training and evaluation datasets.

### **Training Data**

- **Training**: We adapt [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) for training. The processed dataset can be found at: [PAPOGalaxy/PAPO_ViRL39K_train](https://huggingface.co/datasets/PAPOGalaxy/PAPO_ViRL39K_train).
- Validation (optional): We use the testset from [MMK12](https://huggingface.co/datasets/FanqingM/MMK12) for validation during training. **Note that this is solely for monitoring, we do not pick checkpoints based on this.** The processed dataset can be found [PAPOGalaxy/PAPO_MMK12_test](https://huggingface.co/datasets/PAPOGalaxy/PAPO_MMK12_test).

### **Evaluation Data**

We adapted 8 different multimodal reasoning benchmarks to evaluate **PAPO**, which are further identify two groups, including `General Multimodal Reasoning` and `Vision-Dependent Multimodal Reasoning`.
All evaluation benchmarks can be found in https://huggingface.co/datasets/PAPO-Galaxy/PAPO_eval.
For MathVista and MathVerse, we filter out instances with free-form answers to ensure verifiable evaluation and to avoid relying on LLM-as-a-judge.
<!-- - **General Reasoning**
    - `hiyouga/geometry3k`: [Hugging Face Dataset](https://huggingface.co/datasets/hiyouga/geometry3k), [Data Source](https://github.com/lupantech/InterGPS)
    - `AI4Math/MathVista`: [Hugging Face Dataset](https://huggingface.co/datasets/AI4Math/MathVista)
    - `We-Math/We-Math`: [Hugging Face Dataset](https://huggingface.co/datasets/We-Math/We-Math)
    - `FanqingM/MMK12`: [Hugging Face Dataset](https://huggingface.co/datasets/FanqingM/MMK12)
    - `AI4Math/MathVerse`: [Hugging Face Dataset](https://huggingface.co/datasets/AI4Math/MathVerse)

- **Vision-Dependent Reasoning**
  - `lscpku/LogicVista`: [Hugging Face Dataset](https://huggingface.co/datasets/lscpku/LogicVista)
  - `BUAADreamer/clevr_count_70k`: [Hugging Face Dataset](https://huggingface.co/datasets/BUAADreamer/clevr_count_70k)
  - `MMMU/MMMU_Pro`: [Hugging Face Dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)
  - `MathVerse_V` (vision-dependent subset): Adapted from [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) -->

All results in the paper are average accurarcy @ 8 (repeating 8 times), with a temperature set to 1.0.


## 🚀 **Quick Start**

This section provides a brief overview of how to run attacks and evaluate results using our released code and data.

### 1. Generate Priming Dialogues

Run the following command to generate priming dialogues:

```bash
bash gen_dialogue/run.sh
```

### 2. Inference (Model Completion)

“Inference” refers to injecting a crafted **priming dialogue** into the **target model**, followed by a final malicious query. The model's response is used to assess whether the attack succeeded.

- For **closed-source models** (e.g., GPT-4, Gemini), we use the official OpenAI API.
- For **open-source models**, we use [VLLM](https://github.com/vllm-project/vllm) for efficient batch inference.

You can run inference with the following script:

```bash
python generate_model_response.py \
    --input_dir data/dialogues/harmbench/dri \
    --model_name gpt-4o \
    --max_workers 10 \
    --include_v2v3
```

#### 🔍 Options:

- `--input_dir`: Directory containing prompt JSON files (e.g., one per harmful query).
- `--model_name`: Name of the target model (e.g., `gpt-4o`, `gemini-2.0-flash`, or any VLLM-hosted model).
- `--max_workers`: Number of concurrent workers for parallel inference.
- `--include_v2v3`: If set, runs attacks on up to **three priming dialogues per query** (`.json`, `_v2.json`, `_v3.json`) if available.  
  If not set or alternative versions are missing, only the base `.json` file will be used.


### 3. Evaluation

We follow the evaluation protocol described in the paper.

To run evaluation:

```bash
bash eval/example.sh
```

## 📃 Citation

If you find our project is helpful, please cite our paper as

```
@article{miao2025response,
  title={Response attack: Exploiting contextual priming to jailbreak large language models},
  author={Miao, Ziqi and Li, Lijun and Xiong, Yuan and Liu, Zhenhua and Zhu, Pengyu and Shao, Jing},
  journal={arXiv preprint arXiv:2507.05248},
  year={2025}
}
```