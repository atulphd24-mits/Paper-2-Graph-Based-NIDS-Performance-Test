"""
Hybrid Edge Builder

Author : Atul Chauhan
Project : SOC Threat Detection Framework

Builds weighted graph edges using:

1. k-NN on a leakage-safe feature representation
2. Hybrid Similarity
3. Cosine Features
4. Protocol
5. Temporal (exponential decay)
6. Destination Port
7. Flow Behaviour (log-scaled)

Edges are symmetrised explicitly: an undirected edge (i, j) keeps the
highest similarity seen from either endpoint. With ``mutual=True`` only
pairs where each node is in the other's k-NN list are kept.
"""

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from .preprocessing import FlowFeatureTransformer
from .similarity import SimilarityCalculator


class HybridEdgeBuilder:

    def __init__(
        self,
        graph,
        dataframe,
        feature_columns,
        protocol_column,
        timestamp_column,
        src_port_column,
        dst_port_column,
        duration_column,
        bytes_column,
        packets_column,
        k=10,
        threshold=0.50,
        mutual=False,
        metric="cosine",
        similarity=None,
        transformer=None,
        fit_rows=None
    ):

        self.graph = graph

        self.df = dataframe.reset_index(drop=True)

        self.feature_columns = list(feature_columns)

        self.protocol_column = protocol_column

        self.timestamp_column = timestamp_column

        self.src_port_column = src_port_column

        self.dst_port_column = dst_port_column

        self.duration_column = duration_column

        self.bytes_column = bytes_column

        self.packets_column = packets_column

        self.k = k

        self.threshold = threshold

        self.mutual = mutual

        self.metric = metric

        self.similarity = similarity or SimilarityCalculator()

        self.transformer = transformer or FlowFeatureTransformer(
            numeric_columns=[
                col for col in self.feature_columns
                if col not in ("protocol", "l7_proto", "l4_dst_port")
            ],
            categorical_columns=[
                col for col in ("l4_dst_port", "protocol", "l7_proto")
                if col in self.feature_columns
            ],
        )

        # Rows used to fit scaling statistics (e.g. the training mask).
        self.fit_rows = fit_rows

        self.features = None

        self.neighbors = None

        self.edges = None

    # ---------------------------------------------------
    # Validate Columns
    # ---------------------------------------------------

    def validate_columns(self):

        required = self.feature_columns + [
            self.protocol_column,
            self.timestamp_column,
            self.src_port_column,
            self.dst_port_column,
            self.duration_column,
            self.bytes_column,
            self.packets_column,
        ]

        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing Columns : {missing}")

        print("✓ Column validation successful")

    # ---------------------------------------------------
    # Prepare Features
    # ---------------------------------------------------

    def prepare_features(self):

        print("Transforming graph features (log1p + one-hot + scaling)...")

        self.features = self.transformer.fit_transform(
            self.df,
            fit_rows=self.fit_rows
        )

        print(f"✓ Feature Matrix : {self.features.shape}")

    # ---------------------------------------------------
    # Build kNN Index
    # ---------------------------------------------------

    def build_knn(self):

        print("Building k-NN Index...")

        # +1 so each node can drop itself and still keep k neighbours.
        self.neighbors = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(self.features)),
            metric=self.metric,
            n_jobs=-1
        )

        self.neighbors.fit(self.features)

        print("✓ k-NN index created")

    # ---------------------------------------------------
    # Compute Neighbours
    # ---------------------------------------------------

    def compute_neighbors(self):

        distances, indices = self.neighbors.kneighbors(self.features)

        # Drop the node itself. With duplicate flows the node is not always
        # in column 0, so remove it by identity, not by position.
        n = len(indices)
        rows = np.arange(n)[:, None]
        not_self = indices != rows

        k = min(self.k, indices.shape[1] - 1)
        order = np.argsort(~not_self, axis=1, kind="stable")[:, :k]

        return (
            np.take_along_axis(distances, order, axis=1),
            np.take_along_axis(indices, order, axis=1),
        )

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    def summary(self):

        print("=" * 60)
        print("HYBRID EDGE BUILDER")
        print("=" * 60)
        print(f"Nodes : {self.graph.number_of_nodes()}")
        print(f"k : {self.k}")
        print(f"Mutual k-NN : {self.mutual}")
        print(f"Threshold : {self.threshold}")
        print(f"Feature Count : {len(self.feature_columns)}")
        print("=" * 60)

    # ---------------------------------------------------
    # Build Hybrid Edges
    # ---------------------------------------------------

    def build_edges(self):

        print("\nBuilding Hybrid Graph Edges...")

        _, indices = self.compute_neighbors()

        n, k = indices.shape
        src = np.repeat(np.arange(n), k)
        dst = indices.reshape(-1)

        column = lambda name: self.df[name].to_numpy()

        flow = self.df[
            [self.duration_column, self.bytes_column, self.packets_column]
        ].to_numpy(dtype=float)

        result = self.similarity.hybrid_similarity_batch(
            features_a=self.features[src],
            features_b=self.features[dst],
            protocol_a=column(self.protocol_column)[src],
            protocol_b=column(self.protocol_column)[dst],
            time_a=column(self.timestamp_column)[src],
            time_b=column(self.timestamp_column)[dst],
            src_port_a=column(self.src_port_column)[src],
            dst_port_a=column(self.dst_port_column)[src],
            src_port_b=column(self.src_port_column)[dst],
            dst_port_b=column(self.dst_port_column)[dst],
            flow_a=flow[src],
            flow_b=flow[dst],
        )

        edges = pd.DataFrame({"source": src, "target": dst})
        for name, values in result.items():
            edges[name] = values

        edges = edges[edges["hybrid_similarity"] >= self.threshold]

        # Canonical undirected key, then symmetrise.
        edges["u"] = edges[["source", "target"]].min(axis=1)
        edges["v"] = edges[["source", "target"]].max(axis=1)

        direction_count = edges.groupby(["u", "v"])["source"].transform("size")
        if self.mutual:
            edges = edges[direction_count >= 2]

        edges = (
            edges.sort_values("hybrid_similarity", ascending=False)
            .drop_duplicates(["u", "v"])
            .rename(columns={"hybrid_similarity": "weight"})
        )

        self.edges = edges[
            [
                "u",
                "v",
                "weight",
                "feature_similarity",
                "protocol_similarity",
                "temporal_similarity",
                "port_similarity",
                "flow_similarity",
            ]
        ].reset_index(drop=True)

        self.graph.add_nodes_from(range(n))
        self.graph.add_edges_from(
            (u, v, attrs)
            for u, v, attrs in zip(
                self.edges["u"],
                self.edges["v"],
                self.edges.drop(columns=["u", "v"]).to_dict("records"),
            )
        )

        print(f"✓ Directed k-NN candidates : {len(src)}")
        print(f"✓ Undirected edges kept    : {len(self.edges)}")

    # ---------------------------------------------------
    # PyTorch Geometric edge index
    # ---------------------------------------------------

    def edge_index(self):
        """Return a (2, 2E) array with both directions of every edge."""

        if self.edges is None:
            raise RuntimeError("Run build_edges() first")

        u = self.edges["u"].to_numpy()
        v = self.edges["v"].to_numpy()

        return np.vstack([np.concatenate([u, v]), np.concatenate([v, u])])

    def edge_weights(self):
        """Edge weights aligned with :meth:`edge_index`."""

        weight = self.edges["weight"].to_numpy()
        return np.concatenate([weight, weight])

    # ---------------------------------------------------
    # Edge Homophily
    # ---------------------------------------------------

    def edge_homophily(self, labels):
        """Fraction of edges that join two nodes with the same label."""

        labels = np.asarray(labels)
        u = self.edges["u"].to_numpy()
        v = self.edges["v"].to_numpy()

        if len(u) == 0:
            return float("nan")

        return float(np.mean(labels[u] == labels[v]))

    # ---------------------------------------------------
    # Graph Statistics
    # ---------------------------------------------------

    def graph_statistics(self):

        print("\n" + "=" * 60)
        print("HYBRID GRAPH SUMMARY")
        print("=" * 60)

        print(f"Nodes                : {self.graph.number_of_nodes()}")
        print(f"Edges                : {self.graph.number_of_edges()}")
        print(f"Connected Components : {nx.number_connected_components(self.graph)}")

        largest = max(nx.connected_components(self.graph), key=len)

        print(f"Largest Component    : {len(largest)}")
        print(f"Density              : {nx.density(self.graph):.6f}")

        degrees = [d for _, d in self.graph.degree()]

        print(f"Average Degree       : {np.mean(degrees):.2f}")
        print(f"Maximum Degree       : {max(degrees)}")
        print(f"Minimum Degree       : {min(degrees)}")

    # ---------------------------------------------------
    # Export Edge Data
    # ---------------------------------------------------

    def edge_dataframe(self):

        if self.edges is None:
            raise RuntimeError("Run build_edges() first")

        return self.edges.rename(columns={"u": "source", "v": "target"})
