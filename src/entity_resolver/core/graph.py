import re
import collections
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class Attribute:

    def __init__(self, name, attr_type, value, deep_clean=True):
        self.name = name
        self.type = attr_type

        if self.type == 'text':
            self._tokenize(value)
            self.raw_value = ' '.join(self.value)
        elif self.type == 'person_entity' and deep_clean:
            self._clean_person_name(value)
            self.raw_value = ' '.join(self.value)
        else:
            self.value = value
            self.raw_value

    def _tokenize(self, doc, to_stem=False):
        """
        :param doc: string of text value
        :param to_stem: whether stemmed or not
        :return: a lost of tokenized value
        """
        doc = doc.strip()
        doc = re.sub("[^a-zA-Z]", " ", doc)
        doc = doc.lower().split()
        eng_stopwords = stopwords.words("english")
        doc = [w for w in doc if w not in eng_stopwords]
        if to_stem:
            ps = PorterStemmer()
            ps_stems = []
            for word in doc:
                ps_stems.append(ps.stem(word))
            self.value = ps_stems
        else:
            self.value = doc

    def _clean_person_name(self, value):
        """
        :param value: name string e.g. 'S.D. Whitehead' or 'Whitehead, S.D.'
        :return: list of first name and last name parts e.g. ['whitehead', 's.d.']
        """
        value = value.lower()
        cleaned_ordered_name = []  # [first name, last names]
        if ',' in value:
            first, last = value.split(',')
            cleaned_ordered_name.append(first.strip())
            last = ''.join(''.join(last.split('.')).split())
            cleaned_ordered_name.extend(last)
        else:
            names = value.split()
            first = names[-1]
            if len(names) == 1:
                last = ''
            else:
                last = ''.join(names[:-1]).strip()
            cleaned_ordered_name.extend([first, last])
        self.value = cleaned_ordered_name



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
        pass

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def get_attr_names(self):
        '''
        :return: list of attribute names
        '''
        return self.attrs.keys()
        pass

    def get_attr(self, name):
        '''
        :param name: name of attribute
        :return: list of tokenized words if atttribute type is string
        '''
        attr = self.attrs[name]
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
        pass

    def add_node(self, node):
        """
        :param node: Node() object
        """
        self.nodes.append(node)
        pass


class Graph:

    def __init__(self, nodes, edges, attr_types):
        """
        :param nodes: iterables (list) of Node()
        :param edges: iterables (list) of Edge()
        """
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.attr_types = attr_types
        self.id2node = {}
        self.id2edge = {}
        # self.attr_vals = self.get_attr_val()
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

    def get_neighbors(self, node):
        """
        :param node object
        :return: list of Node() that are neighbors of node_id
        """
        # edge_id = self.id2node[node_id].edge
        # return self.get_nodes(edge_id)

        return node.edge.nodes

    def get_attr_names(self):
        """
        :return: list of attribute names for each node in the graph
        """
        return self.nodes[0].get_attr_names()

    def get_attr_val(self, get_raw=False):
        """
        :return: for text data only dictionary key: name values: list of list of tokenized attr with 'name'
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

    def _get_attr_count(self):
        attr_vals = self.get_attr_val(get_raw=True)
        attr_counts = {}
        for name, attr_val in attr_vals.items:
            attr_counts[name] = collections.Counter(attr_val)
        self.attr_counts = attr_counts


    def get_ambiguity_adar(self):
        """
        :return:
        """
        pass




