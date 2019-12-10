""" Contains core classes exposed to other packages.

This package contains classes from :doc:`entity_resolver.core.graph`,
:doc:`entity_resolver.core.resolver`, and
:doc:`entity_resolver.core.evaluator`. It is created for import convenience
similar to :doc:`entity_resolver`. For details, please follow the submodule
links to their documentation pages.
"""
from .evaluator import Evaluator
from .graph import Graph, Node, Edge, Attribute
from .resolver import Resolver
