""" The interface package exposed to users.

This package contains the single class in :doc:`entity_resolver.main` which
encapsulates the whole collective entity resolution algorithm, and provides all
functionalities to users for convenience. In short,

    >>> from entity_resolver import EntityResolver

is equivalent to

    >>> from entity_resolver.main import EntityResolver

For details, please refer to :doc:`entity_resolver.main`.
"""

from .main import EntityResolver
