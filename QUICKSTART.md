# DASG-GCL

Official PyTorch implementation of DASG (Discriminative Attribute-Structure Guided Augmentation for Graph Contrastive Learning).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on Cora
python train.py --dataset 9.cora.pkl

# Train on CiteSeer
python train.py --dataset 12.citeseer.pkl

# Train on Wiki-CS
python train.py --dataset 14.wiki-cs.pkl
```

## Example Output

```
==================================================
Initializing LDGCL View Generator
==================================================
RPCA decomposition on 2708 x 2708 adjacency matrix...
  Target rank: 50, Max iter: 50
  Iter 10: error = 0.012345, rank(L) = 45
  Converged at iteration 23
RPCA decomposition completed:
  L non-zero elements: 125432
  S non-zero elements: 3421

Edge importance computed:
  Range: [0.0000, 1.0000]
  Mean: 0.4523

Feature importance (AANE) computed:
  Range: [0.0000, 1.0000]
  Mean: 0.5012

Training for 400 epochs...
Epoch 20: Loss=0.4523, Val Acc=78.32%, Test Acc=76.45%
Epoch 40: Loss=0.3821, Val Acc=81.23%, Test Acc=79.87%
...
Best Test Accuracy: 85.23%
```

## Contact

For questions or issues, please open an issue on GitHub.
