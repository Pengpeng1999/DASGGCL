"""
RPCA Decomposition Module for LDGCL
Low-rank decomposition of adjacency matrix: A = L + S

Core idea:
- L (low-rank): captures global structural patterns
- S (sparse): noise or anomalous connections
- Edge importance derived from |L_ij|
"""

import numpy as np
import scipy.sparse as ssp
from sklearn.utils.extmath import randomized_svd


def shrink(M, tau):
    """
    Soft-thresholding shrinkage operator
    S_tau(x) = sign(x) * max(|x| - tau, 0)
    """
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0)


def svd_shrink_randomized(M, tau, target_rank=None, n_oversamples=10):
    """
    Singular value thresholding using randomized SVD

    Args:
        M: input matrix
        tau: threshold
        target_rank: target rank (None for auto)
        n_oversamples: oversampling parameter

    Returns:
        L: low-rank approximation
        rank: actual rank
    """
    n = min(M.shape)

    if target_rank is None:
        target_rank = min(100, n // 10, n - 1)
    target_rank = max(1, min(target_rank, n - 1))

    try:
        U, S, Vt = randomized_svd(
            M, n_components=target_rank,
            n_oversamples=n_oversamples,
            random_state=42
        )

        S_shrunk = np.maximum(S - tau, 0)
        rank = np.sum(S_shrunk > 0)

        if rank == 0:
            return np.zeros_like(M), 0

        return U[:, :rank] @ np.diag(S_shrunk[:rank]) @ Vt[:rank, :], rank

    except Exception as e:
        print(f"Randomized SVD failed: {e}")
        return np.zeros_like(M), 0


def rpca_alm(M, lmbda=None, mu=None, tol=1e-5, max_iter=100,
             target_rank=None, verbose=False):
    """
    RPCA via Augmented Lagrangian Method

    min ||L||_* + lambda * ||S||_1
    s.t. M = L + S

    Args:
        M: input matrix (numpy array or scipy sparse)
        lmbda: sparsity penalty, default 1/sqrt(max(m,n))
        mu: Lagrangian multiplier step size
        tol: convergence threshold
        max_iter: maximum iterations
        target_rank: target rank for randomized SVD
        verbose: print iteration info

    Returns:
        L: low-rank component
        S: sparse component
    """
    if ssp.issparse(M):
        M = M.toarray()

    M = M.astype(np.float64)
    m, n = M.shape

    if verbose:
        print(f"RPCA decomposition: matrix size {m} x {n}")

    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(m, n))

    if mu is None:
        mu = m * n / (4.0 * np.sum(np.abs(M)) + 1e-8)

    # Initialize
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)

    mu_inv = 1.0 / mu
    norm_M = np.linalg.norm(M, 'fro') + 1e-8

    for i in range(max_iter):
        # Update L via SVT
        L, rank = svd_shrink_randomized(
            M - S + mu_inv * Y, mu_inv, target_rank=target_rank
        )

        # Update S via soft-thresholding
        S = shrink(M - L + mu_inv * Y, lmbda * mu_inv)

        # Update Y
        residual = M - L - S
        Y = Y + mu * residual

        # Check convergence
        err = np.linalg.norm(residual, 'fro') / norm_M

        if verbose and (i + 1) % 10 == 0:
            print(f"  Iter {i+1}: error = {err:.6f}, rank(L) = {rank}")

        if err < tol:
            if verbose:
                print(f"  Converged at iteration {i+1}")
            break

    return L, S


class GlobalRPCA:
    """
    Global RPCA decomposition for graph adjacency matrix
    """

    def __init__(self, adj, rpca_lambda=None, target_rank=None,
                 max_iter=50, verbose=True):
        """
        Args:
            adj: adjacency matrix (scipy sparse or numpy array)
            rpca_lambda: RPCA sparsity penalty
            target_rank: target rank (None for auto)
            max_iter: maximum iterations
            verbose: print info
        """
        self.verbose = verbose

        if ssp.issparse(adj):
            self.adj_dense = adj.toarray().astype(np.float64)
        else:
            self.adj_dense = adj.astype(np.float64)

        self.num_nodes = self.adj_dense.shape[0]

        if target_rank is None:
            target_rank = min(50, self.num_nodes // 10)
        target_rank = max(1, target_rank)

        if verbose:
            print(f"RPCA decomposition on {self.num_nodes} x {self.num_nodes} adjacency matrix...")
            print(f"  Target rank: {target_rank}, Max iter: {max_iter}")

        self.L, self.S = rpca_alm(
            self.adj_dense,
            lmbda=rpca_lambda,
            target_rank=target_rank,
            max_iter=max_iter,
            verbose=verbose
        )

        if verbose:
            nonzero_L = np.sum(np.abs(self.L) > 1e-5)
            sparse_count = np.sum(np.abs(self.S) > 1e-5)
            print(f"RPCA decomposition completed:")
            print(f"  L non-zero elements: {nonzero_L}")
            print(f"  S non-zero elements: {sparse_count}")

    def get_low_rank_matrix(self):
        """Return the low-rank matrix L"""
        return self.L

    def get_sparse_matrix(self):
        """Return the sparse matrix S"""
        return self.S
