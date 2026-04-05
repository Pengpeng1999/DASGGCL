"""
AANE-inspired Feature Importance Estimation for LDGCL

Core idea:
- Features that are smooth over the low-rank topology are more informative
- Smoothness measured via Laplacian quadratic form: s_f = x_f^T * L * x_f
- Lower smoothness score -> higher importance (feature aligns with structure)
"""

import numpy as np
import torch


def compute_low_rank_laplacian(L_rpca):
    """
    Compute graph Laplacian from low-rank matrix L

    Args:
        L_rpca: low-rank matrix from RPCA (numpy array)

    Returns:
        Laplacian: graph Laplacian matrix
        L_sym: symmetrized low-rank matrix
    """
    # Symmetrize: L_sym = (|L| + |L|^T) / 2
    L_abs = np.abs(L_rpca)
    L_sym = (L_abs + L_abs.T) / 2

    # Degree matrix
    D = np.diag(L_sym.sum(axis=1))

    # Laplacian: Lambda = D - L_sym
    Laplacian = D - L_sym

    return Laplacian, L_sym


def compute_feature_smoothness(X, Laplacian):
    """
    Compute smoothness score for each feature dimension

    Smoothness: s_f = x_f^T * Laplacian * x_f
    Lower value means feature varies smoothly over connected nodes

    Args:
        X: feature matrix (N x F), numpy array
        Laplacian: graph Laplacian (N x N)

    Returns:
        smoothness: smoothness scores (F,)
    """
    # s_f = x_f^T * L * x_f = diag(X^T * L * X)
    # Efficient computation: (X^T @ L @ X).diagonal()
    smoothness = np.diag(X.T @ Laplacian @ X)

    return smoothness


def smoothness_to_importance(smoothness, eps=1e-8):
    """
    Convert smoothness scores to importance scores

    Lower smoothness -> higher importance (feature aligns with topology)

    Args:
        smoothness: smoothness scores (F,)
        eps: small constant for numerical stability

    Returns:
        importance: feature importance scores (F,), in [0, 1]
    """
    # Normalize to [0, 1]
    s_min, s_max = smoothness.min(), smoothness.max()
    smoothness_norm = (smoothness - s_min) / (s_max - s_min + eps)

    # Importance = 1 - normalized_smoothness
    importance = 1 - smoothness_norm

    return importance


def compute_feature_importance_aane(X, L_rpca):
    """
    Compute feature importance using AANE-inspired method

    Pipeline:
    1. Build Laplacian from low-rank matrix L
    2. Compute smoothness for each feature
    3. Convert smoothness to importance

    Args:
        X: feature matrix (N x F), numpy array or torch tensor
        L_rpca: low-rank matrix from RPCA (numpy array)

    Returns:
        feature_importance: importance scores (F,), torch tensor
    """
    # Convert to numpy if needed
    if torch.is_tensor(X):
        X_np = X.cpu().numpy()
    else:
        X_np = X

    # Step 1: Build Laplacian
    Laplacian, _ = compute_low_rank_laplacian(L_rpca)

    # Step 2: Compute smoothness
    smoothness = compute_feature_smoothness(X_np, Laplacian)

    # Step 3: Convert to importance
    importance = smoothness_to_importance(smoothness)

    return torch.from_numpy(importance).float()


class AANEFeatureImportance:
    """
    AANE-inspired feature importance estimator

    Uses low-rank topology to guide feature importance estimation
    """

    def __init__(self, L_rpca, verbose=True):
        """
        Args:
            L_rpca: low-rank matrix from RPCA
            verbose: print info
        """
        self.verbose = verbose
        self.L_rpca = L_rpca

        # Precompute Laplacian
        self.Laplacian, self.L_sym = compute_low_rank_laplacian(L_rpca)

        if verbose:
            print(f"AANE Feature Importance initialized")
            print(f"  Laplacian shape: {self.Laplacian.shape}")

    def compute_importance(self, X):
        """
        Compute feature importance for given features

        Args:
            X: feature matrix (N x F)

        Returns:
            importance: feature importance (F,)
        """
        if torch.is_tensor(X):
            X_np = X.cpu().numpy()
        else:
            X_np = X

        smoothness = compute_feature_smoothness(X_np, self.Laplacian)
        importance = smoothness_to_importance(smoothness)

        if self.verbose:
            print(f"Feature importance computed:")
            print(f"  Smoothness range: [{smoothness.min():.4f}, {smoothness.max():.4f}]")
            print(f"  Importance range: [{importance.min():.4f}, {importance.max():.4f}]")

        return torch.from_numpy(importance).float()
