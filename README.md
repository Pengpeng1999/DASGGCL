# DASG: Discriminative Attribute-Structure Guided Augmentation for Graph Contrastive Learning

Official implementation of **DASG** (Discriminative Attribute-Structure Guided Augmentation), a novel graph contrastive learning framework that addresses the coarse-grained and decoupled limitations of existing augmentation strategies.

## Overview

DASG proposes a principled augmentation framework that:
- **Fine-grained topology augmentation**: Extracts global structural patterns via low-rank decomposition for edge-level importance estimation, moving beyond coarse-grained node-level heuristics.
- **Coupled structure-attribute augmentation**: Devises a topology-guided attribute importance metric based on Laplacian smoothness, directly quantifying attribute-topology alignment.
- **Theoretical guarantees**: Provides theoretical analysis showing that low-rank decomposition preserves the dominant eigenspace and discriminative attributes are preferentially retained.

## Key Features

- **Low-rank decomposition **: Decomposes adjacency matrix into global backbone (L) and sparse noise (S)
- **Edge importance**: Derived from low-rank matrix entries |L_ij|
- **Attribute importance**: Measured via Laplacian smoothness over the low-rank topology
- **Adaptive augmentation**: Importance-weighted stochastic perturbation for both edges and attributes

## Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.7
- PyTorch >= 1.8.0
- PyTorch Geometric >= 2.0.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- scikit-learn >= 0.23.0
- tqdm

## Project Structure

```
DASGGCL/
├── train.py                    # Main training script
├── model.py                    # GRACE model implementation
├── view_generator.py           # DASG view generator
├── rpca_global.py             # Low-rank decomposition
├── feature_importance_ls.py   # Laplacian smoothness-based feature importance
├── data/                      # Dataset directory
│   ├── 9.cora.pkl
│   ├── 12.citeseer.pkl
│   ├── 14.wiki-cs.pkl
│   └── ...
└── README.md
```

## Usage

### Basic Training

Train DASG on Cora dataset with default settings:

```bash
python train.py --dataset 9.cora
```
## Dataset Format

Datasets should be stored as pickle files with the following structure:

```python
{
    'topo': scipy.sparse matrix,  # Adjacency matrix (N x N)
    'attr': numpy array,          # Node features (N x F)
    'label': numpy array          # Node labels (N,)
}
```

## Method Overview

### 1. Low-Rank Decomposition

Decompose adjacency matrix into low-rank backbone and sparse noise:

```
A = L + S
```

where:
- **L** (low-rank): Captures global structural patterns
- **S** (sparse): Represents noise or anomalous connections

### 2. Edge Importance Estimation

Edge importance derived from low-rank matrix entries:

```
Imp_e(i,j) = |L_ij| / max|L|
```

### 3. Attribute Importance Estimation

Attribute importance based on Laplacian smoothness:

```
s_f = x_f^T * Λ * x_f
Imp_f = 1 - normalized(s_f)
```

where Λ is the Laplacian computed from low-rank matrix L.

### 4. Adaptive Augmentation

Generate two augmented views with importance-weighted stochastic perturbation:
- **Edge dropping**: Higher importance → lower drop probability
- **Feature masking**: Higher importance → lower mask probability


## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{dasg2026,
  title={Joint Learning of Global Backbone Structure and Discriminative Attributes for Graph Contrastive Learning},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

## Acknowledgments

- GRACE baseline: [https://github.com/CRIPAC-DIG/GRACE](https://github.com/CRIPAC-DIG/GRACE)
- GCA baseline: [https://github.com/CRIPAC-DIG/GCA](https://github.com/CRIPAC-DIG/GCA)
