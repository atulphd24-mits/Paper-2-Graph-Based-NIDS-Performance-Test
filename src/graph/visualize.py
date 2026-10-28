import matplotlib.pyplot as plt
import networkx as nx
import random

class GraphVisualizer:

    def __init__(self, graph):
        self.graph = graph

    def draw(self, sample_size=300):

        # Random sample of nodes
        nodes = random.sample(list(self.graph.nodes()), sample_size)

        subgraph = self.graph.subgraph(nodes)

        plt.figure(figsize=(12,12))

        pos = nx.spring_layout(
            subgraph,
            seed=42,
            k=0.25
        )

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=25
        )

        nx.draw_networkx_edges(
            subgraph,
            pos,
            alpha=0.3
        )

        plt.title("k-NN Graph")
        plt.axis("off")
        plt.show()