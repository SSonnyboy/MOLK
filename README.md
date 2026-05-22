# MOLK: Mixture of Experts with Invariant Learning & Knowledge

A multimodal cold-start recommendation model that combines Mixture-of-Experts architecture with environment-invariant learning to ensure robust modality fusion under feature missing scenarios.

## Overview

MOLK addresses the multimodal cold-start problem by:

1. **MoE-based Modality Projection**: Each modality (image, text) is processed by a dedicated Mixture-of-Experts layer with top-k gating, enabling adaptive feature extraction.
2. **Learned Modality Fusion**: A gating network dynamically weights image and text embeddings for item representation.
3. **Variance-Invariant Training**: Dirichlet-based environment augmentation samples diverse modality mixing ratios, and a variance penalty loss enforces consistent recommendation performance across all environments. This prevents the model from over-relying on any single modality.

## Project Structure

```
MOLK/
├── main.py              # Entry point, argument parsing, training launch
├── model.py             # MOME_model (MoE layers + gating) and MoE_Layer
├── session.py           # Training loop with Dirichlet environment augmentation
├── loader.py            # Dataset loading, BPR pair sampling
├── criterion.py         # Loss functions (BPR, MSE, InfoNCE, etc.)
├── evaluation.py        # FAISS-based evaluation with Numba JIT metrics
├── enviroment.py        # Environment setup (paths, device, logging, seed)
├── tool.py              # Utilities (logging, colored print, seed, etc.)
├── datasets/
│   └── Baby/
│       ├── train.txt / val.txt / test.txt
│       ├── image_feat.npy / text_feat.npy
│       ├── image_feat_missing_*.npy / text_feat_missing_*.npy
│       └── cold_item_index.npy / warm_missing_item_index.npy
└── exp_report/          # Experiment logs and checkpoints
```

## Loss Function

The total training loss is:

```
L = L_bpr + L_reg + L_align + L_moe_balance + L_fusion_balance + λ * L_penalty
```

| Loss | Description |
|------|-------------|
| `L_bpr` | Bayesian Personalized Ranking loss for implicit feedback |
| `L_reg` | L2 regularization on user/item embeddings |
| `L_align` | MSE alignment between fused item embedding and each modality embedding |
| `L_moe_balance` | KL divergence encouraging uniform expert utilization within each MoE layer |
| `L_fusion_balance` | KL divergence encouraging balanced modality weighting in the fusion gate |
| `L_penalty` | Variance of BPR losses across Dirichlet-sampled environments (invariance) |

## Usage

### Training

```bash
python main.py --dataset baby --alpha 0.1 --penalty_coeff 50
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--free_emb_dimension` | 128 | Embedding dimension |
| `--lr` | 0.001 | Learning rate |
| `--batch_size` | 2048 | Batch size |
| `--epoch` | 1000 | Max training epochs |
| `--early_stop` | 200 | Early stopping patience |
| `--alpha` | 0.1 | Dirichlet concentration parameter |
| `--penalty_coeff` | 50 | Variance penalty coefficient |
| `--align_coeff` | 1 | Alignment loss coefficient |
| `--moe_balance_coeff` | 1 | MoE balance loss coefficient |
| `--fusion_balance_coeff` | 1 | Fusion balance loss coefficient |
| `--reg_coeff` | 1e-6 | L2 regularization coefficient |
| `--topk` | [10,20,30,40,50] | Evaluation cutoffs |

### Evaluation Metrics

- HR@k (Hit Rate)
- Recall@k
- NDCG@k (Normalized Discounted Cumulative Gain)

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- FAISS (faiss-cpu or faiss-gpu)
- Numba

## Dataset

We use Amazon product review datasets (Baby, Clothing, Sports) with CLIP-extracted visual and textual features. The data follows the LATTICE-style organization with pre-defined train/val/test splits and cold-item indices.
