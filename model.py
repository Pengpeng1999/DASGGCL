"""
GRACE Model for LDGCL
Graph Contrastive Representation Learning

Components:
- GCN Encoder: multi-layer GCN
- Projection Head: 2-layer MLP
- InfoNCE Loss: contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """GCN Encoder"""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, activation=F.relu, dropout=0.0):
        super(GCNEncoder, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GRACE(nn.Module):
    """GRACE: Graph Contrastive Representation Learning"""

    def __init__(self, encoder, num_hidden, num_proj_hidden, tau=0.5):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.tau = tau

        # Projection head (2-layer MLP)
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x, edge_index):
        """Forward pass, returns encoder output"""
        return self.encoder(x, edge_index)

    def projection(self, z):
        """Projection head for contrastive loss"""
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        """Cosine similarity matrix"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        """One-way contrastive loss"""
        f = lambda x: torch.exp(x / self.tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def loss(self, z1, z2):
        """Symmetric contrastive loss"""
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        return (l1 + l2).mean() * 0.5

    def batched_semi_loss(self, z1, z2, batch_size):
        """Batched one-way contrastive loss for large graphs"""
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(num_nodes, device=device)
        losses = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_nodes)
            batch_indices = indices[start:end]

            z1_batch = z1[batch_indices]
            refl_sim = f(self.sim(z1_batch, z1))
            between_sim = f(self.sim(z1_batch, z2))

            pos_mask = torch.zeros(end - start, num_nodes, device=device)
            pos_mask[torch.arange(end - start), batch_indices] = 1

            pos = (between_sim * pos_mask).sum(1)
            neg = refl_sim.sum(1) + between_sim.sum(1) - (refl_sim * pos_mask).sum(1)

            losses.append(-torch.log(pos / neg))

        return torch.cat(losses)

    def batched_loss(self, z1, z2, batch_size=1024):
        """Batched symmetric contrastive loss"""
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.batched_semi_loss(h1, h2, batch_size)
        l2 = self.batched_semi_loss(h2, h1, batch_size)

        return (l1 + l2).mean() * 0.5


def get_grace_model(num_features, hidden_dim=256, out_dim=256,
                    num_layers=2, num_proj_hidden=128, tau=0.5):
    """Build GRACE model"""
    encoder = GCNEncoder(
        in_channels=num_features,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_layers=num_layers
    )

    model = GRACE(
        encoder=encoder,
        num_hidden=out_dim,
        num_proj_hidden=num_proj_hidden,
        tau=tau
    )

    return model
