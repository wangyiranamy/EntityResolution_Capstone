import re
import collections
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class Attribute:

    def __init__(self, name, attr_type, value, deep_clean=True):
        self.name = name
        self.type = attr_type
        if self.type == 'text':
            self.value = self._tokenize(value)
            self.raw_value = value
        elif self.type == 'person_entity' and deep_clean:
            self.value = self._clean_person_name(value)
            self.raw_value = value
        else:
            self.value = value
            self.raw_value = value

    def _tokenize(self, doc, to_stem=False):
        """
        :param doc: string of text value
        :param to_stem: whether stemmed or not
        :return: a lost of tokenized value
        """
        doc = doc.strip()
        doc = re.sub("[^a-zA-Z]", " ", doc)
        doc = doc.lower().split()
        if to_stem:
            ps = PorterStemmer()
            ps_stems = []
            for word in doc:
                ps_stems.append(ps.stem(word))
            return ps_stems
        else:
            return doc

    def _clean_person_name(self, value):
        last, *first = value.split('_')
        first = ' '.join(first).strip()
        return last, first


class Node:

    def __init__(self, node_id, edge, attrs):
        """
        :param node_id: int
        :param edge: edge object
        :param attrs:  list of Attribute()
        """
        self.id = node_id
        self.edge = edge
        attr_vals, raw_attr_vals = dict(), dict()
        for attr in attrs:
            attr_vals[attr.name] = attr.value
            raw_attr_vals[attr.name] = attr.raw_value
        self.attr_vals = attr_vals
        self.raw_attr_vals = raw_attr_vals

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if type(self) is type(other):
            return self.id == other.id
        return NotImplemented


class Edge:
    def __init__(self, edge_id, attrs=None):
        """
        :param edge_id: int
        :param attrs: list of Attribute()
        """
        self.id = edge_id
        self.nodes = []
        self.attrs = attrs

    def add_node(self, node):
        """
        :param node: Node() object
        """
        self.nodes.append(node)


class Graph:

    def __init__(self, nodes, edges, attr_types):
        """
        :param nodes: iterables (list) of Node()
        :param edges: iterables (list) of Edge()
        """
        self.nodes = list(nodes)
        self.edges = list(edges.values())
        self.attr_types = attr_types
        attr_vals = collections.defaultdict(list)
        raw_attr_vals = collections.defaultdict(list)
        for name in attr_types:
            attr_vals[name] = []
        for node in self.nodes:
            for name, value in node.attr_vals.items():
                attr_vals[name].append(value)
            for name, raw_value in node.raw_attr_vals.items():
                raw_attr_vals[name].append(raw_value)
        self.attr_vals = attr_vals
        self.raw_attr_vals = raw_attr_vals

    def add_nodes(self, new_nodes):
        """
        :param new_nodes:  iterables (list) of Node()
        """
        self.nodes.extend(new_nodes)

    def add_edges(self, new_edges):
        """
        :param new_edges:  iterables (list) of Edge()
        """
        self.nodes.extend(new_edges)

    def get_neighbors(self, node):
        """
        :param node object
        :return: list of Node() that are neighbors of node_id
        """
        return node.edge.nodes

    def get_ambiguity_adar(self, f1, f2, is_raw1,  is_raw2):
        """
        :param f1: the function that takes in the attribute dict of one node and output its first_attr_val
        :param f2: the function that takes in the attribute dict of one node and output its second_attr_val
        :param is_raw1: boolean, whether the first attribute value is raw or not
        :param is_raw2: boolean, whether the second  attribute value is raw or not
        :return: dictionary of node's ambiguous val
        """
        first_attrs, second_attrs = dict(), dict()
        first_attr2node = collections.defaultdict(list)
        cal_amb_a1_a2 = collections.defaultdict()
        for node in self.nodes:
            if is_raw1:
                a1_val = f1(node.raw_attr_vals)
                a2_val = f2(node.raw_attr_vals)

            else:
                a1_val = f1(node.attr_vals)
                a2_val = f2(node.attr_vals)
            first_attrs[node] = a1_val
            second_attrs[node] = a2_val
            first_attr2node[a1_val].append(node)
        for a1_val, nodes in first_attr2node.items():
            a2_vals = set()
            for node in nodes:
                a2_vals.add(second_attrs[node])
            cal_amb_a1_a2[a1_val] = len(a2_vals)/len(self.nodes)
        node_ambiguity = collections.defaultdict()
        for node, a1_val in first_attrs.items():
            node_ambiguity[node] = cal_amb_a1_a2[a1_val]
        return node_ambiguity
