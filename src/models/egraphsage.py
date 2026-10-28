"""
E-GraphSAGE-style edge classifier for NetFlow data.

Re-implementation following the description in Lo et al., "E-GraphSAGE: A
Graph Neural Network based Intrusion Detection System for IoT" (NOMS 2022):

* nodes are network endpoints, edges are flows carrying the flow features;
* node states start as all-ones vectors;
* layer k: h_N(v) = mean of the features of edges incident to v (layer 1) or
  of neighbour states combined with those edges (later layers);
  h_v = ReLU(W [h_v || h_N(v)]);
* a flow (u, v) is classified from [z_u || z_v].

Details such as the exact node definition and hyper-parameters should be
checked against the original paper before claiming a faithful reproduction.
This module makes the node definition a parameter, because endpoint identity
itself can act as a shortcut: if an attacker's IP appears in training and
test, the model can memorise it instead of learning behaviour.

Node definitions
    "ip"       - one node per IP address
    "ip_port"  - one node per IP:port endpoint
    "random"   - every flow gets its own two fresh endpoints (control: no
                 shared structure, so message passing cannot help or leak)
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def build_endpoint_graph(df, node_mode="ip_port", src_ip="ipv4_src_addr", dst_ip="ipv4_dst_addr",
                         src_port="l4_src_port", dst_port="l4_dst_port"):
    """Return (edge_index (2, E), num_nodes) with one directed edge per flow."""
    if node_mode == "ip":
        src = df[src_ip].astype(str)
        dst = df[dst_ip].astype(str)
    elif node_mode == "ip_port":
        src = df[src_ip].astype(str) + ":" + df[src_port].astype(str)
        dst = df[dst_ip].astype(str) + ":" + df[dst_port].astype(str)
    elif node_mode == "random":
        n = len(df)
        return np.vstack([np.arange(n), np.arange(n, 2 * n)]), 2 * n
    else:
        raise ValueError(f"unknown node_mode {node_mode}")

    codes, uniques = pd.factorize(pd.concat([src, dst], ignore_index=True))
    n = len(df)
    return np.vstack([codes[:n], codes[n:]]), len(uniques)


class EdgeSAGELayer(nn.Module):
    """h_v' = ReLU(W [h_v || mean_{edges e at v} (W_m [h_u || x_e])])."""

    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.message = nn.Linear(node_dim + edge_dim, out_dim)
        self.update = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        num_nodes = h.shape[0]

        # Undirected aggregation: each flow informs both of its endpoints.
        targets = torch.cat([dst, src])
        neighbours = torch.cat([src, dst])
        messages = self.message(torch.cat([h[neighbours], edge_attr.repeat(2, 1)], dim=1))

        aggregated = torch.zeros(num_nodes, messages.shape[1], device=h.device)
        aggregated.index_add_(0, targets, messages)
        degree = torch.bincount(targets, minlength=num_nodes).clamp(min=1).unsqueeze(1)
        aggregated = aggregated / degree

        return F.relu(self.update(torch.cat([h, aggregated], dim=1)))


class EGraphSAGE(nn.Module):

    def __init__(self, edge_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        dims = [1] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            EdgeSAGELayer(dims[i], edge_dim, dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = dropout
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def forward(self, edge_index, edge_attr, num_nodes):
        h = torch.ones(num_nodes, 1, device=edge_attr.device)
        for layer in self.layers:
            h = F.dropout(layer(h, edge_index, edge_attr), self.dropout, self.training)
        src, dst = edge_index
        return self.classifier(torch.cat([h[src], h[dst]], dim=1))


def train_egraphsage(edge_index, edge_attr, num_nodes, y, train_mask, val_mask,
                     hidden_dim=128, lr=1e-3, weight_decay=1e-5, max_epochs=300,
                     patience=30, seed=0, threads=None, eval_every=5, inductive=False,
                     return_model=False):
    """
    Full-batch training with class-weighted cross-entropy (loss on training flows
    only) and early stopping on validation PR-AUC (checked every ``eval_every``
    epochs; ``patience`` counts epochs). Returns attack probabilities for every flow.

    transductive (default): every flow is an edge of the message-passing graph
        during training, validation and scoring.
    inductive: training uses a graph of training edges only; validation scores
        come from the graph of training + validation edges; final scores from the
        graph of all edges, so validation and test edges enter only at inference.

    ``return_model=True`` returns (probabilities, trained model).
    """
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)

    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    train_np = np.asarray(train_mask, bool)
    val_np = np.asarray(val_mask, bool)
    train_mask = torch.as_tensor(train_np)

    if inductive:
        train_edges = torch.as_tensor(np.flatnonzero(train_np))
        seen_edges = torch.as_tensor(np.flatnonzero(train_np | val_np))
    else:
        train_edges = seen_edges = torch.arange(edge_index.shape[1])

    def forward(edges):
        """Scores for ``edges``, using only those edges for message passing."""
        return model(edge_index[:, edges], edge_attr[edges], num_nodes)

    counts = torch.bincount(y_t[train_mask], minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts.clamp(min=1)))

    model = EGraphSAGE(edge_attr.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Positions of training / validation flows inside the graphs used for them.
    train_in_train_graph = train_mask[train_edges]
    val_positions = np.flatnonzero(val_np[seen_edges.numpy()])

    best, best_state, wait = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward(train_edges)[train_in_train_graph]
        loss = criterion(logits, y_t[train_edges][train_in_train_graph])
        loss.backward()
        optimizer.step()

        if epoch % eval_every and epoch != max_epochs:
            continue
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(forward(seen_edges), 1)[:, 1].numpy()
        score = average_precision_score(y[seen_edges.numpy()][val_positions], proba[val_positions])
        if score > best:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += eval_every
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_edges = torch.arange(edge_index.shape[1])
        proba = torch.softmax(forward(all_edges), 1)[:, 1].numpy()
    return (proba, model) if return_model else proba
