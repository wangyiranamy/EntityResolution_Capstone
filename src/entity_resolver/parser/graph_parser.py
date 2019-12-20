""" Contains only one parser class to be used by the main module.

Example:

    >>> from entity_resolver.parser import GraphParser
    >>> parser = GraphParser(attr_types={'name': 'person_entity'})
    >>> graph = parser.parse('graph.json')
"""

import time
import json
from typing import Mapping
import pandas as pd
import numpy as np
from collections import defaultdict
from ..core import Attribute, Node, Edge, Graph
from ..core.utils import WithLogger


class GraphParser(WithLogger):
    """ A parser class to parse graph data file.

    Args:
        attr_types: Mapping attribute names to the attribute type. Refer to the
            ``attr_type`` attribute of `~entity_resolver.core.graph.Attribute`
            for details on attribute types.
        verbose: Indicate how much information to be logged/printed in the
            console during the program execution. Same as in the
            `~entity_resolver.main.EntityResolver` class.

    Attributes:
        attr_types: Same as in the above parameters section.
    """

    def __init__(self, attr_types: Mapping[str, str], verbose: int = 0):
        self.attr_types = attr_types
        super().__init__(verbose)

    def parse(self, graph_data_path: str) -> Graph:
        """ Parse the graph data into a Graph object.

        Args:
            graph_data_path: The path to the graph data. The data file has to
                **strictly follow** the format as described in
                :doc:`../tutorial`.

        Returns:
            The parsed Graph object.
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
