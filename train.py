"""
LDGCL Training Script
Topology: RPCA-based edge importance
Features: AANE-inspired importance

Training pipeline:
1. RPCA decomposition: A = L + S
2. Edge importance from L
3. Feature importance via Laplacian smoothness
4. Adaptive view generation
5. Contrastive learning with InfoNCE
"""

import os
import argparse
import pickle
import numpy as np
import scipy.sparse as ssp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm

from view_generator import LDGCLViewGenerator
from model import get_grace_model


def scipy_sparse_to_edge_index(adj):
    """Convert scipy sparse matrix to edge_index"""
    adj_coo = adj.tocoo()
    row = torch.from_numpy(adj_coo.row.astype(np.int64))
    col = torch.from_numpy(adj_coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def load_data(data_path):
    """Load dataset"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    adj = data['topo']
    adj.setdiag(0)
    adj.eliminate_zeros()

    if 'attr' in data:
        features = data['attr']
        if ssp.issparse(features):
            features = features.toarray()
        features = features.astype('float32')
    else:
        features = np.eye(adj.shape[0], dtype='float32')

    if 'label' in data:
        labels = data['label']
        if ssp.issparse(labels):
            labels = labels.toarray().flatten()
        labels = labels.astype('int64')
    else:
        raise ValueError("No labels in dataset")

    return adj, features, labels


def evaluate_node_classification(embeddings, labels, train_ratio=0.1,
                                  val_ratio=0.1, random_state=42):
    """Evaluate node classification performance"""
    embeddings = normalize(embeddings, norm='l2', axis=1)

    num_nodes = len(labels)
    indices = np.arange(num_nodes)

    train_val_indices, test_indices = train_test_split(
        indices, test_size=1 - train_ratio - val_ratio,
        random_state=random_state
    )
    train_size = train_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, train_size=train_size,
        random_state=random_state
    )

    clf = LogisticRegression(max_iter=1000, multi_class='multinomial',
                             solver='lbfgs', random_state=42)
    clf.fit(embeddings[train_indices], labels[train_indices])

    y_val_pred = clf.predict(embeddings[val_indices])
    y_test_pred = clf.predict(embeddings[test_indices])

    results = {
        'val_acc': accuracy_score(labels[val_indices], y_val_pred),
        'val_f1_macro': f1_score(labels[val_indices], y_val_pred, average='macro'),
        'test_acc': accuracy_score(labels[test_indices], y_test_pred),
        'test_f1_macro': f1_score(labels[test_indices], y_test_pred, average='macro'),
        'test_f1_micro': f1_score(labels[test_indices], y_test_pred, average='micro'),
    }

    return results


def train_ldgcl(adj, features, labels, args):
    """Train LDGCL model"""
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"LDGCL Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Nodes: {adj.shape[0]}, Edges: {adj.nnz // 2}")
    print(f"Features: {features.shape[1]}, Classes: {len(np.unique(labels))}")

    # Prepare data
    edge_index = scipy_sparse_to_edge_index(adj).to(device)
    x = torch.from_numpy(features).float().to(device)
    num_features = features.shape[1]

    # Create view generator
    view_generator = LDGCLViewGenerator(
        edge_index, x, adj,
        rpca_lambda=args.rpca_lambda,
        device=device,
        verbose=True
    )

    # Build model
    model = get_grace_model(
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        num_proj_hidden=args.num_proj_hidden,
        tau=args.tau
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f"\nModel config: hidden={args.hidden_dim}, out={args.out_dim}, "
          f"layers={args.num_layers}, tau={args.tau}, lr={args.lr}")

    # Training loop
    print(f"\nTraining for {args.num_epochs} epochs...")
    model.train()
    best_val_acc = 0
    best_embeddings = None

    pbar = tqdm(range(1, args.num_epochs + 1), desc="Training")
    for epoch in pbar:
        optimizer.zero_grad()

        # Generate views
        (ei1, x1), (ei2, x2) = view_generator.generate_two_views(
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2
        )

        # Encode
        z1 = model(x1, ei1)
        z2 = model(x2, ei2)

        # Loss
        if args.batch_size > 0 and z1.size(0) > args.batch_size:
            loss = model.batched_loss(z1, z2, batch_size=args.batch_size)
        else:
            loss = model.loss(z1, z2)

        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Evaluation
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            model.eval()
            with torch.no_grad():
                embeddings = model(x, edge_index).cpu().numpy()

            results = evaluate_node_classification(
                embeddings, labels,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio
            )

            if results['val_acc'] > best_val_acc:
                best_val_acc = results['val_acc']
                best_embeddings = embeddings.copy()

            print(f"\nEpoch {epoch}: Loss={loss.item():.4f}, "
                  f"Val={results['val_acc']:.4f}, Test={results['test_acc']:.4f}")
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index).cpu().numpy()

    final_results = evaluate_node_classification(
        final_embeddings, labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    return final_embeddings, final_results


def main(args):
    """Main function"""
    data_path = f"{args.data_dir}/{args.dataset}.pkl"
    print(f"Loading data: {data_path}")
    adj, features, labels = load_data(data_path)

    embeddings, results = train_ldgcl(adj, features, labels, args)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    if args.save_embeddings:
        save_path = f"ldgcl_embeddings_{args.dataset}.npy"
        np.save(save_path, embeddings)
        print(f"\nEmbeddings saved to: {save_path}")

    return embeddings, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDGCL: Low-rank Decomposition GCL')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='9.cora')
    parser.add_argument('--device', type=str, default='auto')

    # RPCA
    parser.add_argument('--rpca_lambda', type=float, default=None)

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--tau', type=float, default=0.4)

    # Augmentation
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)

    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=20)

    # Evaluation
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    # Other
    parser.add_argument('--save_embeddings', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
