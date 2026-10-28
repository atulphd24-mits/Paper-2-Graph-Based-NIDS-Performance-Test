import networkx as nx


class GraphStatistics:

    def __init__(self, graph):
        self.graph = graph

    def summary(self):

        print("=" * 60)
        print("GRAPH STATISTICS")
        print("=" * 60)

        print("Nodes               :", self.graph.number_of_nodes())
        print("Edges               :", self.graph.number_of_edges())
        print("Density             :", nx.density(self.graph))
        print("Connected Components:", nx.number_connected_components(self.graph))
        print("Average Degree      :", sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes())
