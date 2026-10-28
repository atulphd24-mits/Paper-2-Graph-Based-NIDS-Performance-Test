import networkx as nx
import pandas as pd


class GraphFeatureExtractor:
    """
    Extract structural and centrality-based features
    from the hybrid network graph.
    """

    def __init__(self, graph, betweenness_samples=500, exact_betweenness_limit=5000):
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a NetworkX Graph")

        self.graph = graph
        self.features = None
        self.betweenness_samples = betweenness_samples
        self.exact_betweenness_limit = exact_betweenness_limit

    def compute_degree(self):
        return dict(self.graph.degree())

    def compute_degree_centrality(self):
        return nx.degree_centrality(self.graph)

    def compute_pagerank(self):
        return nx.pagerank(
            self.graph,
            weight="weight"
        )

    def compute_betweenness(self):
        """
        Betweenness centrality.

        Exact up to ``exact_betweenness_limit`` nodes, sampled above it.
        Edge weights are similarities, so they are converted to distances
        (1 - similarity) before shortest paths are computed.
        """
        for _, _, data in self.graph.edges(data=True):
            data["distance"] = 1.0 - data.get("weight", 0.0)

        k = None
        if self.graph.number_of_nodes() > self.exact_betweenness_limit:
            k = min(self.betweenness_samples, self.graph.number_of_nodes())

        return nx.betweenness_centrality(
            self.graph,
            k=k,
            weight="distance",
            normalized=True,
            seed=42
        )

    def compute_closeness(self):
        return nx.closeness_centrality(self.graph)

    def compute_clustering(self):
        return nx.clustering(
            self.graph,
            weight="weight"
        )

    def compute_eigenvector(self):
        try:
            return nx.eigenvector_centrality(
                self.graph,
                max_iter=500,
                weight="weight"
            )
        except nx.PowerIterationFailedConvergence:
            return {
                node: 0.0
                for node in self.graph.nodes()
            }

    def compute_component_id(self):
        component_id = {}

        for idx, component in enumerate(
            nx.connected_components(self.graph)
        ):
            for node in component:
                component_id[node] = idx

        return component_id

    def compute_component_size(self):
        component_size = {}

        for component in nx.connected_components(self.graph):
            for node in component:
                component_size[node] = len(component)

        return component_size

    def extract(self):
        """
        Extract all graph features.
        """

        print("=" * 60)
        print("GRAPH FEATURE EXTRACTION")
        print("=" * 60)

        print("Nodes :", self.graph.number_of_nodes())
        print("Edges :", self.graph.number_of_edges())

        print("\nComputing degree...")
        degree = self.compute_degree()

        print("Computing degree centrality...")
        degree_centrality = self.compute_degree_centrality()

        print("Computing PageRank...")
        pagerank = self.compute_pagerank()

        print("Computing betweenness centrality...")
        betweenness = self.compute_betweenness()

        print("Computing closeness centrality...")
        closeness = self.compute_closeness()

        print("Computing clustering coefficient...")
        clustering = self.compute_clustering()

        print("Computing eigenvector centrality...")
        eigenvector = self.compute_eigenvector()

        print("Computing connected components...")
        component_id = self.compute_component_id()
        component_size = self.compute_component_size()

        self.features = pd.DataFrame({
            "node": list(self.graph.nodes()),

            "degree": [
                degree[n]
                for n in self.graph.nodes()
            ],

            "degree_centrality": [
                degree_centrality[n]
                for n in self.graph.nodes()
            ],

            "pagerank": [
                pagerank[n]
                for n in self.graph.nodes()
            ],

            "betweenness": [
                betweenness[n]
                for n in self.graph.nodes()
            ],

            "closeness": [
                closeness[n]
                for n in self.graph.nodes()
            ],

            "clustering": [
                clustering[n]
                for n in self.graph.nodes()
            ],

            "eigenvector": [
                eigenvector[n]
                for n in self.graph.nodes()
            ],

            "component_id": [
                component_id[n]
                for n in self.graph.nodes()
            ],

            "component_size": [
                component_size[n]
                for n in self.graph.nodes()
            ]
        })

        print("\n✓ Graph features extracted")
        print("Feature Matrix :", self.features.shape)

        return self.features

    def get_features(self):
        if self.features is None:
            raise RuntimeError(
                "Features not extracted. "
                "Run extract() first."
            )

        return self.features