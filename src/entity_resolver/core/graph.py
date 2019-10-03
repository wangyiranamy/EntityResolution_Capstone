class Attribute:

    def __init__(self, name, attr_type, value):
        self.name = name
        self.type = attr_type
        self.value = value


class Node:

    def __init__(self, node_id, edge_id, attrs):
        """
        :param node_id: int
        :param attrs:  list of Attribute()
        """
        self.id = node_id
        self.attrs = attrs
        self.edge = edge_id
        pass

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class Edge:
    def __init__(self, edge_id, attrs=None):
        """
        :param edge_id: int
        :param attrs: list of Attribute()
        """
        self.id = edge_id
        self.nodes = []
        self.attrs = attrs
        pass

    def add_node(self, node):
        """
        :param node: Node() object
        """
        self.nodes.append(node)
        pass


class Graph:

    def __init__(self, nodes, edges):
        """
        :param nodes: iterables (list) of Node()
        :param edges: iterables (list) of Edge()
        """
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.id2node = {}
        self.id2edge = {}
        for node in self.nodes:
            self.id2node[node.id] = node
        for edge in self.edges:
            self.id2edge[edge.id] = edge
        pass

    def add_nodes(self, new_nodes):
        """
        :param new_nodes:  iterables (list) of Node()
        """
        for node in new_nodes:
            self.id2node[node.id] = node
        self.nodes.extend(new_nodes)
        pass

    def add_edges(self, new_edges):
        """
        :param new_edges:  iterables (list) of Edge()
        """
        for edge in new_edges:
            self.id2edge[edge.id] = edge
        self.nodes.extend(new_edges)
        pass

    def get_neighbors(self, node_id):
        """
        :param node_id: int id
        :return: list of Node() that are neighbors of node_id
        """
        edge_id = self.id2node[node_id].edge
        return self.get_nodes(edge_id)

    def get_nodes(self, edge_id):
        """
        :param edge_id: int id
        :return: list of Node() that are connected by hyper edge edge_id
        """
        return self.id2edge[edge_id].nodes


