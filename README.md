# CAG-Med


This repository contains the official implementation of the paper:
**"CAG-Med: Cross-Attention Gating with Physically Motivated Projections for Parameter-Efficient Medical Image Analysis under Data Scarcity"**
*Najiba Tagougui, Monji Kherallah*

##  Overview
Contemporary large-scale visual foundation models struggle in clinical deployment due to quadratic attention complexity, prohibitive parameter budgets, and reliance on language-model tokenisation for continuous biomedical signals. 

**CAG-Med** addresses these limitations via:
1. **Zero-Parameter Physical Prior ($\Pi$):** Encodes anatomical density priors (Beer-Lambert approximations for X-ray, multiplanar means for MRI) as inductive biases at zero learnable cost.
2. **Asymmetric Cross-Attention Gating (CAG):** Projects continuous clinical vectors directly into spatial feature maps, reducing $\mathcal{O}(N^2)$ self-attention to linear $\mathcal{O}(N \cdot N')$ complexity.
3. **Hierarchical Clinical Ordering:** Modulates features by clinical timescale (slow-varying omics/demographics $\rightarrow$ fast-varying vitals).
4. **Structured Dropout (Zero-Guide Regime):** Naturally degrades to a fixed $0.5$ suppression when auxiliary data is missing, providing implicit regularisation in data-scarce regimes.

##  Key Results
* **Parameter Efficiency:** Only **0.6M parameters** for 2D classification.
* **Memory Efficiency:** 3D volumetric segmentation requires only **3.2 GB VRAM** (vs >15 GB for Swin-UNETR).
* **Data Scarcity:** Achieves Dice **0.417** on BraTS 2020 with only 24 training subjects, and **0.989** on MSD Heart with 12 subjects.
* **Edge Deployment:** All experiments run on a single consumer GPU (NVIDIA RTX 4070 Laptop).

##  Repository Structure
```text
cag-med/
├── src/
│   ├── models/
│   │   ├── cag_operator.py      # Core CAG and Pi modules
│   │   ├── cag_med_2d.py        # ChestX-ray14 classifier
│   │   └── cag_med_3d.py        # BraTS / MSD Heart segmentor
│   ├── data/                    # Dataset loaders
│   ├── utils/                   # Losses, metrics, helpers
│   └── train.py                 # Main training loop
├── configs/                     # YAML configs for hyperparameters
├── scripts/                     # Bash scripts for reproduction
└── requirements.txt
