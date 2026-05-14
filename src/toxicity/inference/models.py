"""Torch Geometric model definitions used by API inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from .constants import TOX21_TASKS


class EnhancedToxGCN(nn.Module):
    """Three-layer GCN with topological node features."""

    def __init__(self, num_tasks: int = len(TOX21_TASKS), input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(hidden_dim // 2, 64)
        self.lin2 = nn.Linear(64, num_tasks)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)


class EnhancedToxGIN(nn.Module):
    """Three-layer GIN with MLP aggregators and topological node features."""

    def __init__(self, num_tasks: int = len(TOX21_TASKS), input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINConv(nn2)

        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.conv3 = GINConv(nn3)

        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(hidden_dim // 2, 64)
        self.lin2 = nn.Linear(64, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)


def build_model(architecture: str, num_tasks: int = len(TOX21_TASKS)) -> nn.Module:
    """Build a supported model by architecture key."""
    architecture_key = architecture.lower()
    if architecture_key == "gcn":
        return EnhancedToxGCN(num_tasks=num_tasks)
    if architecture_key == "gin":
        return EnhancedToxGIN(num_tasks=num_tasks)
    raise ValueError(f"Unsupported architecture: {architecture}")


def predict_probabilities(model: nn.Module, batch) -> torch.Tensor:
    """Return sigmoid probabilities for a PyG batch."""
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        return torch.sigmoid(logits)
