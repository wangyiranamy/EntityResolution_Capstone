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
            self.raw_value = ' '.join(self.value)
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
        """
        :param value: name string e.g. 'S.D. Whitehead' or 'Whitehead, S.D.'
        :return: list of first name and last name parts
            e.g. ['whitehead', 's.d.']
        """
        value = value.lower()
        if ',' in value:
            first, last = value.split(',', 1)
            first = first.strip()
            last = ''.join(''.join(last.split('.')).split())
            return [first, last]
        else:
            names = value.split()
            first = names[-1]
            if len(names) == 1:
                last = ''
            else:
                last = ''.join(names[:-1]).strip()
            return [first, last]


class Node:

    def __init__(self, node_id, edge, attrs):
        """
        :param node_id: int
        :param edge: edge object
        :param attrs:  list of Attribute()
        """
        self.id = node_id
        self.edge = edge
        self.attrs = {}
        for attr in attrs:
            self.attrs[attr.name] = attr

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if type(self) is type(other):
            return self.id == other.id
        return NotImplemented

    def get_attr_names(self):
        '''
        :return: list of attribute names
        '''
        return list(self.attrs.keys())

    def get_attr(self, name, get_raw=False):
        '''
        :param name: name of attribute
        :return: list of tokenized words if atttribute type is string
        '''
        attr = self.attrs[name]
        if get_raw:
            return attr.raw_value
        return attr.value


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

    def get_attr_names(self):
        """
        :return: list of attribute names for each node in the graph
        """
        return self.nodes[0].get_attr_names()

    def get_attr_vals(self, get_raw=False):
        """
        :return: for text data only dictionary key: name values: list of list
            of tokenized attr with 'name'
        """
        attr_vals = {}
        for name in self.get_attr_names():
            attr_vals[name] = []
        for node in self.nodes:
            for name, node_attr in node.attrs.items():
                if get_raw:
                    attr_vals[name].append(node_attr.raw_value)
                else:
                    attr_vals[name].append(node_attr.value)

        return attr_vals

    def get_ambiguity_adar(self):
        """
        :return: dictionary with attribute names as key; value: ratio of
            count of distinct attribute values
        """
        attr_vals = self.get_attr_vals(get_raw=True)
        attr_adar_ambiguity = {}
        for attr_name, attr_val in attr_vals.items():
            counter = collections.Counter(attr_val)
            for key in counter:
                counter[key] /= len(attr_val)
            attr_adar_ambiguity[attr_name] = counter
        return attr_adar_ambiguity
