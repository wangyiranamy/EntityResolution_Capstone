import time
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from ..core import Attribute, Node, Edge, Graph
from ..core.utils import WithLogger


class GraphParser(WithLogger):

    def __init__(self, attr_types, verbose=0):
        """
        :param attr_types: e.g {'title': 'text', 'name': 'person_entity'}
        """
        self.attr_types = attr_types
        super().__init__(verbose)

    def parse(self, graph_data_path):
        """
        :param graph_data_path: table with three columns:
            node_id, edge_id, attr_dict
        :return: Graph object
        """
        start_time = time.time()
        with open(graph_data_path, 'r') as f:
            graph_df = json.load(f)
        edge_dict = dict()  # key: edge_id; val: Edge object
        for row in graph_df:
            node_attrs = []
            for attr_name, attr_val in row['attr_dict'].items():
                attr_type = self.attr_types[attr_name]
                node_attr = Attribute(attr_name, attr_type, attr_val)
                node_attrs.append(node_attr)
            edge_id = row['edge_id']
            if edge_id not in edge_dict:
                edge_dict[edge_id] = Edge(edge_id)
            node = Node(row['node_id'], node_attrs)
            # append nodes to edges
            edge_dict[edge_id].add_node(node)
        graph = Graph(edge_dict.values(), self.attr_types)
        end_time = time.time()
        time_taken = end_time - start_time
        self.logger.debug(f'Time taken to buid graph: {time_taken}s')
        self.logger.info(f'Number of nodes in graph: {len(graph.nodes)}')
        self.logger.info(f'Number of edges in graph: {len(graph.edges)}')
        return graph
