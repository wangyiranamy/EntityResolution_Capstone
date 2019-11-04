import pandas as pd
import numpy as np
import json
from collections import defaultdict
from ..core.graph import Attribute, Node, Edge, Graph


class GraphParser:

    def __init__(self, attr_types):
        """
        :param attr_types: e.g {'title': 'text', 'name': 'person_entity'}
        """
        self.attr_types = attr_types

    def parse(self, graph_data_path):
        """
        :param graph_data_path: table with three columns:
            node_id, edge_id, attr_dict
        :return: Graph object
        """
        with open(graph_data_path, 'r') as f:
            graph_df = json.load(f)
        node_list = []
        edge_dict = {}  # key: edge_id; val: Edge object
        for row in graph_df:
            node_attrs = []
            for attr_name, attr_val in row['attr_dict'].items():
                attr_type = self.attr_types[attr_name]
                node_attr = Attribute(attr_name, attr_type, attr_val)
                node_attrs.append(node_attr)
            edge_id = row['edge_id']
            if edge_id not in edge_dict:
                edge_dict[edge_id] = Edge(edge_id)
            node = Node(row['node_id'], edge_dict[edge_id], node_attrs)
            node_list.append(node)
            # append nodes to edges
            edge_dict[edge_id].add_node(node)
        return Graph(node_list, edge_dict, self.attr_types)
