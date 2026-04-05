"""
View Generator for LDGCL
Topology augmentation via RPCA + Feature augmentation via AANE

Key differences from GCA:
- Edge importance: derived from low-rank matrix L (edge-level)
- Feature importance: derived from Laplacian smoothness (topology-guided)
"""

import numpy as np
import torch
from torch_geometric.utils import dropout_adj

from rpca_global import GlobalRPCA
from feature_importance_ls import AANEFeatureImportance


def compute_edge_importance(L, edge_index):
    """
    Compute edge importance from low-rank matrix L

    imp_e(i,j) = |L_ij| / max|L|

    Args:
        L: low-rank matrix (numpy array)
        edge_index: edge indices (2, num_edges)

    Returns:
        importance: edge importance (num_edges,)
    """
    edge_index_np = edge_index.cpu().numpy()
    row, col = edge_index_np[0], edge_index_np[1]

    importance = np.abs(L[row, col])

    if importance.max() > 0:
        importance = importance / importance.max()

    return torch.from_numpy(importance).float()


def importance_to_drop_weights(importance, drop_rate=0.3):
    """
    Convert importance to drop weights
    High importance -> low drop weight (more likely to keep)

    Args:
        importance: importance scores (0-1)
        drop_rate: average drop rate

    Returns:
        drop_weights: drop weights
    """
    inv_importance = 1 - importance + 1e-8
    drop_weights = inv_importance / (inv_importance.mean() + 1e-8) * drop_rate
    drop_weights = torch.clamp(drop_weights, 0, 0.9)

    return drop_weights


def drop_edge_weighted(edge_index, drop_weights, threshold=0.7):
    """
    Weighted edge dropping

    Args:
        edge_index: edge indices (2, num_edges)
        drop_weights: drop weight per edge
        threshold: max drop weight

    Returns:
        edge_index: augmented edge indices
    """
    weights = drop_weights.clamp(max=threshold)
    keep_mask = torch.bernoulli(1.0 - weights).to(torch.bool)

    return edge_index[:, keep_mask]


def drop_feature_weighted(x, feature_importance, drop_rate, threshold=0.7):
    """
    Weighted feature masking based on AANE importance

    High importance -> low drop probability (preserve informative features)

    Args:
        x: node features (N, F)
        feature_importance: importance per feature (F,)
        drop_rate: average drop rate
        threshold: max drop weight

    Returns:
        x_aug: augmented features
    """
    inv_importance = 1 - feature_importance + 1e-8
    drop_weights = inv_importance / (inv_importance.mean() + 1e-8) * drop_rate
    drop_weights = drop_weights.clamp(max=threshold)

    drop_mask = torch.bernoulli(drop_weights).to(torch.bool)

    x_aug = x.clone()
    x_aug[:, drop_mask] = 0

    return x_aug


class LDGCLViewGenerator:
    """
    LDGCL View Generator
    - Topology: RPCA-based edge importance (edge-level)
    - Features: AANE-inspired importance (topology-guided)
    """

    def __init__(self, edge_index, x, adj, rpca_lambda=None, device=None, verbose=True):
        """
        Args:
            edge_index: edge indices (2, num_edges)
            x: node features (N, F)
            adj: adjacency matrix (scipy sparse)
            rpca_lambda: RPCA sparsity penalty
            device: computation device
            verbose: print info
        """
        self.device = device or torch.device('cpu')
        self.edge_index = edge_index.to(self.device)
        self.x = x.to(self.device)
        self.num_nodes = x.size(0)
        self.num_features = x.size(1)
        self.verbose = verbose

        # Step 1: RPCA decomposition
        if verbose:
            print("=" * 50)
            print("Initializing LDGCL View Generator")
            print("=" * 50)

        self.rpca = GlobalRPCA(adj, rpca_lambda=rpca_lambda, verbose=verbose)

        # Step 2: Compute edge importance from L
        self._compute_edge_importance()

        # Step 3: Compute feature importance via AANE
        self._compute_feature_importance()

    def _compute_edge_importance(self):
        """Compute edge importance from low-rank matrix"""
        self.edge_importance = compute_edge_importance(
            self.rpca.L, self.edge_index
        ).to(self.device)

        if self.verbose:
            print(f"Edge importance computed:")
            print(f"  Range: [{self.edge_importance.min():.4f}, {self.edge_importance.max():.4f}]")
            print(f"  Mean: {self.edge_importance.mean():.4f}")

    def _compute_feature_importance(self):
        """Compute feature importance via AANE method"""
        self.aane = AANEFeatureImportance(self.rpca.L, verbose=False)
        self.feature_importance = self.aane.compute_importance(self.x).to(self.device)

        if self.verbose:
            print(f"Feature importance (AANE) computed:")
            print(f"  Range: [{self.feature_importance.min():.4f}, {self.feature_importance.max():.4f}]")
            print(f"  Mean: {self.feature_importance.mean():.4f}")

    def generate_view(self, drop_edge_rate, drop_feature_rate):
        """
        Generate one augmented view

        Args:
            drop_edge_rate: edge drop rate
            drop_feature_rate: feature drop rate

        Returns:
            edge_index_aug: augmented edges
            x_aug: augmented features
        """
        # Edge augmentation
        edge_drop_weights = importance_to_drop_weights(
            self.edge_importance, drop_edge_rate
        )
        edge_index_aug = drop_edge_weighted(self.edge_index, edge_drop_weights)

        # Feature augmentation
        x_aug = drop_feature_weighted(
            self.x, self.feature_importance, drop_feature_rate
        )

        return edge_index_aug, x_aug

    def generate_two_views(self, drop_edge_rate_1=0.2, drop_edge_rate_2=0.4,
                           drop_feature_rate_1=0.3, drop_feature_rate_2=0.4):
        """
        Generate two augmented views for contrastive learning

        Returns:
            (edge_index_1, x_1): view 1
            (edge_index_2, x_2): view 2
        """
        view1 = self.generate_view(drop_edge_rate_1, drop_feature_rate_1)
        view2 = self.generate_view(drop_edge_rate_2, drop_feature_rate_2)

        return view1, view2
