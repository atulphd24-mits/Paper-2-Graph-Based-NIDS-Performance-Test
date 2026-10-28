"""
Graph neural network detectors for the flow similarity graph.

Previous DeepDRAC runs trained full-batch at lr=1e-4 for <=100 epochs,
i.e. at most ~100 optimiser steps, and stopped while validation ROC-AUC
was still near 0.5. The trainer here uses lr=3e-3, up to 400 epochs and
early stopping on validation PR-AUC, and the MLP baseline shares the same
trainer so a weak GNN can be told apart from weak training.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv


class FlowSAGE(nn.Module):
    """GraphSAGE node classifier with a skip path for the node's own features."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index):
        h0 = F.relu(self.input(x))
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, edge_index))), self.dropout, self.training)
        return self.head(torch.cat([h0, h], dim=-1))


class FlowMLP(nn.Module):
    """Same capacity as FlowSAGE but without message passing."""

    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


def train_node_classifier(
    model,
    x,
    edge_index,
    y,
    train_mask,
    val_mask,
    lr=3e-3,
    weight_decay=1e-4,
    max_epochs=400,
    patience=40,
    seed=42,
    verbose=False,
):
    """
    Full-batch training with class weighting and early stopping on
    validation PR-AUC. Returns (best_model, attack_probabilities, history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    edge_index = (
        None if edge_index is None else torch.as_tensor(edge_index, dtype=torch.long)
    )
    train_mask = torch.as_tensor(train_mask, dtype=torch.bool)
    val_mask = torch.as_tensor(val_mask, dtype=torch.bool)

    counts = torch.bincount(y[train_mask], minlength=2).float()
    class_weights = counts.sum() / (2 * counts.clamp(min=1))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 3
    )

    y_val = y[val_mask].numpy()
    best_score, best_state, best_epoch, wait = -1.0, None, 0, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x, edge_index)[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(x, edge_index), dim=-1)[:, 1]
        val_pr_auc = average_precision_score(y_val, proba[val_mask].numpy())
        scheduler.step(val_pr_auc)
        history.append({"epoch": epoch, "loss": loss.item(), "val_pr_auc": val_pr_auc})

        if val_pr_auc > best_score:
            best_score, best_epoch, wait = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if verbose and epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val PR-AUC {val_pr_auc:.4f}")

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(x, edge_index), dim=-1)[:, 1].numpy()

    if verbose:
        print(f"Best epoch {best_epoch} | val PR-AUC {best_score:.4f}")

    return model, proba, history
