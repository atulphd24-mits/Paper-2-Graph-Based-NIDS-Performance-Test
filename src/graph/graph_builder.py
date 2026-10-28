import networkx as nx
import pandas as pd


class GraphBuilder:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.graph = nx.Graph()

    ###################################################
    # Create Nodes
    ###################################################

    def create_nodes(self):

        for idx, row in self.df.iterrows():

            attributes = row.to_dict()

            self.graph.add_node(
                idx,
                **attributes
            )

        print(f"✓ Nodes Created : {self.graph.number_of_nodes()}")

        return self.graph

    ###################################################
    # Summary
    ###################################################

    def summary(self):

        print("=" * 60)
        print("GRAPH SUMMARY")
        print("=" * 60)

        print("Nodes :", self.graph.number_of_nodes())
        print("Edges :", self.graph.number_of_edges())