""" Contains the `Graph` class as well its necessary component classes.

A `Graph` object is basically a combination of a list of `Node` and a list of
`Edge`. A `Node` object is constructed based on a list of `Attribute` objects
and contains a dictionary mapping attribute names to values. The preprocessing
of attriubte values are done when construting the `Attribute` objects.

Note:
    For each of the class in the module, their protected attributes are wrapped
    by read-only properties of names without the underscore prefixes. These
    properties' documentations are therefore omitted.
"""

import re
import collections
from typing import (
    Any, List, Tuple, Hashable,
    Iterable, Dict, Callable, Mapping
)


class Attribute:
    """ Store attribute names, types, and values and preprocess values.

    * If ``attr_type`` is ``'text'``, then the value is tokenized as a list
      of strings.
    * If ``attr_type`` is ``'person_entity'``, then the value is split into a
      tuple of (last name, first name).

    Args:
        name: The attribute name.
        attr_type: The attribute type. Currently only implemented
            ``'person_entity'`` and ``'text'``.
        value: The original unprocessed attribute value.

    Attributes:
        _name: (`str`): Same as ``name`` in the parameters section.
        _type: (`str`): Same as ``attr_type`` in the parameters section.
        _value: (`~typing.Any`): Preprocessed value depending on the type.
        _raw_value (`~typing.Any`): Same as ``value`` in the parameters
            section.
    """

    def __init__(self, name: str, attr_type: str, value: Any):
        self._name = name
        self._type = attr_type
        if attr_type == 'text':
            self._value = self._tokenize(value)
            self._raw_value = value
        elif attr_type == 'person_entity':
            self._value = self._clean_person_name(value)
            self._raw_value = value
        else:
            self._value = value
            self._raw_value = value

    @property
    def name(self):
        """ `str`: Omitted."""
        return self._name

    @property
    def type(self):
        """ `str`: Omitted."""
        return self._type

    @property
    def value(self):
        """ `~typing.Any`: Omitted."""
        return self._value

    @property
    def raw_value(self):
        """ `~typing.Any`: Omitted."""
        return self._raw_value

    @staticmethod
    def _tokenize(doc: str) -> List[str]:
        """ tokenize the text string into a list of words.

        Args:
            doc: string of text string.
        
        Returns:
            A list of tokenized words.
        """
        doc = doc.strip()
        doc = re.sub("[^a-zA-Z]", " ", doc)
        doc = doc.lower().split()
        return doc

    @staticmethod
    def _clean_person_name(name: str) -> Tuple[str, str]:
        """ Retrieve first and last names from normalized person name.
        
        Args:
            name: A normalized name string. It should consist only of lower
                case letters and underscores of the format:
                <last name>_<first name>_<middle name>, where there should be
                no spaces in the last name, and spaces in first name should all
                be replaced by underscores. Other punctuational marks should be
                removed. For example, ``'W. W. Wang'`` should be transformed to
                ``'wang_w_w'``.
        
        Returns:
            A tuple of (last name, first name).
        """
        last, *first = name.split('_')
        first = ' '.join(first).strip()
        return last, first


class Node:
    """ Store information about a reference.
    
    Args:
        node_id: A unique identifier for a reference. Distinct node object must
            have distinct ``node_id``.
        attrs: All attributes of this reference.

    Important:
        In order for the whole entity resolution to function as expected, the
        ``node_id`` should also be **comparable** besides being **hashable**.
    
    Attributes:
        _id (`~typing.Hashable`): Same as ``node_id`` in the parameters
            section.
        _attr_vals (`~typing.Dict`\ [`str`, `~typing.Any`]): Mapping attriubte
            names to their processed values.
        _raw_attr_vals (`~typing.Dict`\ [`str`, `~typing.Any`]): Mapping
            attribute names to their original unprocessed values.
    """

    def __init__(self, node_id: Hashable, attrs: Iterable[Attribute]):
        self._id = node_id
        attr_vals, raw_attr_vals = dict(), dict()
        for attr in attrs:
            attr_vals[attr.name] = attr.value
            raw_attr_vals[attr.name] = attr.raw_value
        self._attr_vals = attr_vals
        self._raw_attr_vals = raw_attr_vals

    @property
    def id(self):
        """ `~typing.Hashable`: Omitted."""
        return self._id

    @property
    def attr_vals(self):
        """ `~typing.Dict`\ [`str`, `~typing.Any`]: Omitted."""
        return self._attr_vals

    @property
    def raw_attr_vals(self):
        """ `~typing.Dict`\ [`str`, `~typing.Any`]: Omitted."""
        return self._raw_attr_vals

    def __hash__(self) -> int:
        """ Use the ``_id`` attribute as hash value."""
        return self._id

    def __eq__(self, other: 'Node') -> bool:
        """ Two nodes are equal if and only if they have the same ``_id``.
        
        Args:
            other: Another node to be compared with.
        """
        return self._id == other.id


class Edge:
    """ Store information about a hyper-edge.

    Args:
        edge_id: A unique identifier for a hyper-edge. Distinct edge object
            must have distinct ``edge_id``.
        
    
    Attributes:
        _id (`~typing.Hashable`): Same as in the ``edge_id`` in the parameters
            section.
        _nodes (`~typing.List`\ [`Node`]): A list of references connected
            through this hyper-edge.
    """

    def __init__(self, edge_id: Hashable):
        self._id = edge_id
        self._nodes = list()

    @property
    def id(self):
        """ `~typing.Hashable`: Omitted."""
        return self._id

    @property
    def nodes(self):
        """ `~typing.List`\ [`Node`]: Omitted."""
        return self._nodes

    def add_node(self, node: Node) -> None:
        """ Add a reference to this hyper-edge.
        
        Args:
            node: The reference to be added to the ``_nodes`` attribute.
        """
        self._nodes.append(node)


class Graph:
    """ Store information about the references and their relations.

    Args:
        edges: All hyper-edges this reference graph contains. Since each edge
            object contains all nodes objects, this sufficies to be the only
            input.
        attr_types: Mapping attribute names to the attribute type. Refer to the
            ``attr_type`` attribute of `Attribute` for details on attribute
            types.

    Important:
        The construction of this graph object depends heavily on the assumption
        that each reference only appears in one hyper-edge. Should the
        assumption fail, unexpected errors could occur. Refer to
        :doc:`../advanced_guide` for more details.
    
    Attributes:
        _nodes (`~typing.List`\ [`Node`]): A list of references contained in
            the graph.
        _edges (`~typing.List`\ [`Edge`]): A list of hyper-edges contained in
            the graph.
        _node_to_edge (`~typing.Dict`\ [`Node`, `Edge`]): Mapping references
            to the hyper-edges they are contained in.
        _attr_types (`~typing.Mapping`\ [`str`, `str`]): Same as ``attr_types``
            in the above parameters section.
        _attr_vals (`~typing.Dict`\ [`str`, `~typing.List`]): Mapping attribute
            names to a list of preprocessed attribute values of all references
            with the corresponding names.
        _raw_attr_vals (`~typing.Dict`\ [`str`, `~typing.List`]): Mapping
            attribute names to a list of original unprocessed attribute
            values of all references with the corresponding names.
    """

    def __init__(self, edges: Iterable[Edge], attr_types: Mapping[str, str]):
        self._edges = list()
        self._node_to_edge = dict()
        for edge in edges:
            self._edges.append(edge)
            for node in edge.nodes:
                self._node_to_edge[node] = edge
        self._nodes = list(self._node_to_edge.keys())
        self._attr_types = attr_types
        attr_vals = collections.defaultdict(list)
        raw_attr_vals = collections.defaultdict(list)
        for name in attr_types:
            attr_vals[name] = []
        for node in self.nodes:
            for name, value in node.attr_vals.items():
                attr_vals[name].append(value)
            for name, raw_value in node.raw_attr_vals.items():
                raw_attr_vals[name].append(raw_value)
        self._attr_vals = attr_vals
        self._raw_attr_vals = raw_attr_vals

    @property
    def nodes(self):
        """ `~typing.List`\ [`Node`]: Omitted."""
        return self._nodes

    @property
    def edges(self):
        """ `~typing.List`\ [`Edge`]: Omitted."""
        return self._edges

    @property
    def attr_types(self):
        """ `~typing.Mapping`\ [`str`, `str`]: Omitted."""
        return self._attr_types

    @property
    def attr_vals(self):
        """ `~typing.Dict`\ [`str`, `~typing.List`]: Omitted."""
        return self._attr_vals

    @property
    def raw_attr_vals(self):
        """ `~typing.Dict`\ [`str`, `~typing.List`]: Omitted."""
        return self._raw_attr_vals

    def get_neighbors(self, node: Node) -> List[Node]:
        """ Retrive the neighboring references of a given reference.
        
        Args:
            node: To find the neighboring references of this reference.
        
        Returns:
            A list of neighboring references. Note that this list will contain
            the input ``node`` as well.
        """
        return self._node_to_edge[node].nodes

    def get_ambiguity_adar(
        self,
        f1: Callable[[Dict[str, Any], Dict[str, Any]], Any],
        is_raw1: bool,
        f2: Callable[[Dict[str, Any], Dict[str, Any]], Any],
        is_raw2: bool
    ) -> Dict[Node, float]:
        """ Compute Adar ambiguity score of a reference based on attributes.

        For further details on computation of Adar ambiguity to understand
        the following documentation, please refer to :doc:`../advanced_guide`.

        Args:
            f1: this function should compute the first attribute for ambiguity
                computation.
            is_raw1: Indicate whether the input attribute value dictionaries
                into ``f1`` consist of unprocessed values.
            f2: this function should compute the second attribute for ambiguity
                computation.
            is_raw2: Indicate whether the input attribute value dictionaries
                into ``f2`` consist of unprocessed values.
        
        Returns:
            Mapping references to their ambiguity scores.
        """
        first_attrs, second_attrs = dict(), dict()
        first_attr2node = collections.defaultdict(list)
        cal_amb_a1_a2 = collections.defaultdict()
        for node in self._nodes:
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
            cal_amb_a1_a2[a1_val] = len(a2_vals)/len(self._nodes)
        node_ambiguity = collections.defaultdict()
        for node, a1_val in first_attrs.items():
            node_ambiguity[node] = cal_amb_a1_a2[a1_val]
        return node_ambiguity
