class Attribute:

    def __init__(self, name, attr_type, value):
        self.name = name
        self.type = attr_type
        self.value = value


class Node:

    def __init__(self, node_id, attrs):
        self.id = node_id
        self.attrs = attrs
        pass

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class Edge:

    def __init__(self, edge_id, nodes, attrs):
        self.id = edge_id
        self.nodes = nodes
        self.attrs = attrs
        pass


class Graph:

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        pass

    @staticmethod
    def calc_node_sim(node1, node2):
        return Node.calc_similarity(node1, node2)

    def get_neighbors(self, node_id):
        pass
