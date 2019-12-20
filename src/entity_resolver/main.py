""" The interface module exposed to users.

This module contains a single class ``EntityResolver`` which serves as the
only interface for users to apply the collective entity resolution algorithm
to their data.

Example:

    >>> from entity_resolver import EntityResolver
    >>> er = EntityResolver()
    >>> data_path = 'path/to/your/data'
    >>> label_path = 'path/to/your/ground_truth'
    >>> score = er.resolve_and_eval(label_path, data_path)

Tip:
    Although currently the ``attr_types`` argument only support
    ``'person_entity'`` and ``'text'`` as values, the effects of this argument
    is restricted to certain preprocessing, default attribute similarity
    function, and deciding what values are passed when computing attribute
    similarities (thus affecting inputs if users are passing functions to
    ``attr_strategy``).

    If an unsupported type is input, then no preprocessings are done and their
    original values are passed to computation of attribute similarity
    functions. The attribute name with an unsupported type **must** be mapped
    to an attribute similarity strategy in ``attr_strategy`` as well.

See Also:

    * Detailed explanation for the above example can be found in
      :doc:`../tutorial`.
    * A more comprehensive explanation for the input parameters of this class
      can be found in :doc:`../advanced_guide`.
"""

import logging
import collections
from typing import Optional, Any, Union, Callable, Mapping, Tuple, List
from matplotlib import pyplot as plt
from .core import Resolver, Evaluator
from .core.utils import WithLogger, logtime
from .parser import GraphParser, GroundTruthParser


class EntityResolver(WithLogger):
    """ The interface class of the collective entity resolution algorithm.

    Args:
        attr_types: Mapping attribute names to their types.
        blocking_strategy: Describe the strategy for blocking. It should accept
            two dictionaries as reference attributes and return their
            distance.
        raw_blocking: Indicate if raw values of reference attributes are to be
            used in blocking. It affects the input of ``blocking strategy``.
        alpha: The ratio of relational similarity involved in calculating
            cluster similarities. It should be between 0 and 1 (inclusive).
        weights: Mapping attribute names to their ratio when calculating
            attribute similarities. The ratio should sum to 1. If it is
            ``None`` (default), each attribute is assigned equal weight.
        attr_strategy: Mapping attribute names to similarity strategy names.
            Valid values are either strings including ``'stfidf'``,
            ``'jaro_winkler'``, and ``'jaro'``. or a callable that takes two
            attribute values and return their similarity score. If not
            specified, a default strategy will be employed depending on the
            attribute type: ``'text'`` for soft-tfidf, and ``person_entity``
            for Jaro-Winkler. Refer to :doc:`../advanced_guide` for more
            details.
        rel_strategy: Name of the strategy to compute relational similarity.
            Valid values are ``'jaccard_coef'``, ``'jaraccard_coef_fr'``,
            ``'adar_neighbor'``, ``'adar_neighbor_fr'``, ``'adar_attr'``, and
            ``'adar_attr_fr'``. If it is ``None``, ``'jaccard_coef'`` for
            Jaccard Coefficient will be used. Refer to :doc:`../advanced_guide`
            for more details.
        blocking_threshold: The threshold for allowing two references to be put
            in one bucket during blocking. Only references with distance
            (computed by ``blocking_strategy``) **strictly less than** this
            threshold are put in the same bucket.
        bootstrap_strategy: Describe the strategy for bootstrapping. It should
            accept two dictionaries as reference attributes and return if they
            can be a bootstrap candidate pair. If it is ``None`` (default),
            only two references with all attributes being exactly the same can
            be bootstrap a candidate pair. See also ``edge_match_threshold``.
        raw_bootstrap: Indicate if raw values of reference attributes are to be
            used in bootstrapping. It affects the input of
            ``bootstrap_strategy``.
        edge_match_threshold: The number of extra
            (**greater than or equal to**) neighboring reference pairs needed
            to pass ``bootstrap_strategy`` check for a bootstrap candidate pair
            to be finally put in the same cluster during bootstrapping.
        first_attr: Describe how to obtain the first attribute value in
            computation of a reference ambiguity. It should either be ``None``
            (default) or a function accepting one dictionary as reference
            attributes and return a hashable value. If ``rel_strategy`` is
            ``'adar_attr'`` or ``'adar_attr_fr'``, this must be a valid
            function and cannot be ``None``. Refer to :doc:`../advanced_guide`
            for more details.
        first_attr_raw: Indicate if raw values of reference attributes are to
            be used in calculating reference ambiguity. It affects the input of
            ``first_attr``.
        second_attr: Similar to ``first_attr``. Describe how to obtain the
            second attribute value in computation of a reference ambiguity
            instead. Refer to :doc:`../advanced_guide` for more details.
        second_attr_raw: Similar to ``first_attr_raw``. It affects the input of
            ``second_attr`` instead.
        linkage: Describe how to compute cluster attribute similarities based
            on reference attribute similarities. Valid values are ``'min'``,
            ``'max'``, and ``'average'``. Refer to :doc:`../advanced_guide` for
            more details.
        similarity_threshold: When the entry with the largest similarity in the
            priority queue is less than this value, the algorithm stops.
        evaluator_strategy: The strategy name to be used for evaluation when
            needed. Valid values are either strings including
            ``'precision_recall'``, ``'ami'``, and ``'v_measure'``, or a
            callable that follows the signatures of class methods in
            `~entity_resolver.core.utils.ClusteringMetrics` (two
            `~collections.OrderedDict` as inputs and any performance indicator
            as output). Refer to :doc:`../advanced_guide` for more details.
        seed: The random seed to be used in blocking, which is the only source
            of randomness in this algorithm. If it is ``None`` (default), the
            current system time is used as the seed.
        plot_prc: Indicate whether to plot a precision-recall curve at the end
            of evaluation if ``resolve_and_eval`` method is called. It also
            affects the output of ``resolve_and_eval`` method. Note that
            setting this to ``True`` will **significantly increase** the time
            taken for the program to complete.
        verbose: Indicate how much information to be logged/printed in the
            console during the program execution.

            * 0 or smaller: no logging
            * 1: Some meta information are logged. This includes the input
              data, built graph, evaluation results, and time taken for major
              steps of the program.
            * 2 or larger: All messages are logged, most of them are used for
              debugging purpose.
        **kwargs: Additional keyword arguments that are only used in specific
            cases. Refer to :doc:`../advanced_guide` for more details. Valid
            arguments are:

            * second_sim (`str`): Indicate the secondary similarity
              measure used for SoftTfIdf similarity computation if it is set
              in ``attr_strategy``. Valid values are ``'jaro_winkler'``,
              ``'jaro'`` and ``'scaled_lev'``. Default is ``'jaro_winkler'``.
            * stfidf_threshold (`float`): Indicate the threshold of
              secondary similarity used in SoftTfIdf similarity computation if
              it is set in ``attr_strategy``. Default is ``0.5``.
            * jw_prefix_weight (`float`): Indicate the prefix weight used
              in Jaro-winkler similarity computation if it, or SoftTfIdf with
              Jaro-winkler as secondary similarity is set in ``attr_strategy``.
              Default is ``0.1``.
            * average_method (`str`): Indicate how to compute adjusted
              mutual information if it is set in ``evaluator_strategy``. Valid
              values are ``'min'``, ``'geometric'``, ``'arithmetic'``, and
              ``'max'``. Default is ``'max'``.

    Note:
        All protected attributes below which have corresponding parameters in
        the above section without prefix underscores are essentially the same.
        They are encapsulated by corresponding properties (without underscore)
        using ``@property`` to be protected against invalid value assignments.
        Their documentations are therefore omitted.

    Tip:
        Although setting ``plot_prc`` to ``True`` will automatically plot a
        precision-recall curve at the end, it will also significantly increase
        the running time, and the plot is not customizable. However, if it is
        set to ``True``, the ``resolve_and_eval`` method will return the list
        of precision and recall scores to plot the curve. It is recommended to
        store it for further usage.

    Attributes:
        _attr_types (`~typing.Mapping`\ [`str`, `str`]): Omitted.
        _blocking_strategy (`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`], `~typing.Mapping`\ [`str`, `str`]], `float`]):
            Omitted.
        _raw_blocking (`bool`): Omitted.
        _alpha (`~typing.Union`\ [`float`, `int`]): Omitted.
        _weights (`~typing.Optional`\ [`~typing.Mapping`\ [`str`, `float`]]):
            Omitted.
        _attr_strategy (`~typing.Mapping`\ [`str`, `~typing.Union`\ [`str`, `~typing.Callable`]]): Omitted.
        _rel_strategy (`str`): Omitted.
        _blocking_threshold (`~typing.Union`\ [`float`, `int`]): Omitted.
        _bootstrapping_strategy (`~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`], `~typing.Mapping`\ [`str`, `str`]], `bool`]]): Omitted.
        _raw_bootstrap (`bool`):
            Omitted.
        _edge_match_threshold (`int`): Omitted.
        _first_attr (`~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`]], `str`]]):
            Omitted.
        _first_attr_raw (`bool`): Omitted.
        _second_attr (`~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`]], `str`]]):
            Omitted.
        _second_attr_raw (`bool`): Omitted.
        _linkage (`str`): Omitted.
        _similarity_threshold (`float`): Omitted.
        _evaluator_strategy (`str`): Omitted.
        _seed (`~typing.Optional`\ [`int`]): Omitted.
        _plot_prc (`bool`): Omitted.
        _kwargs: Omitted.
        _graph_parser (`~entity_resolver.parser.graph_parser.GraphParser`):
            Created based on input parameters to parse the graph data.
        _ground_truth_parser (`~entity_resolver.parser.ground_truth_parser.GroundTruthParser`):
            Created based on input parameters to parse the ground truth data.
        _resolver (`~entity_resolver.core.resolver.Resolver`): Created based on
            input parameters to execute the main collective entity resolution
            algorithm. It is the core part of this whole project.
        _evaluator (`~entity_resolver.core.evaluator.Evaluator`): Created based
            on input parameters to evaluate the results of entity resolution.
"""

    def __init__(
        self,
        attr_types: Mapping[str, str],
        blocking_strategy: Callable[
            [Mapping[str, str], Mapping[str, str]], float
        ],
        raw_blocking: bool = False,
        alpha: Union[float, int] = 0,
        weights: Optional[Mapping[str, float]] = None,
        attr_strategy: Mapping[str, Union[str, Callable]] = dict(),
        rel_strategy: str = 'jaccard_coef',
        blocking_threshold: Union[float, int] = 3,
        bootstrap_strategy: Optional[
            Callable[[Mapping[str, str], Mapping[str, str]], bool]
        ] = None,
        raw_bootstrap: bool = False,
        edge_match_threshold: int = 1,
        first_attr: Optional[Callable[[Mapping[str, str]], str]] = None,
        first_attr_raw: bool = False,
        second_attr: Optional[Callable[[Mapping[str, str]], str]] = None,
        second_attr_raw: bool = False,
        linkage: str = 'max',
        similarity_threshold: float = 0.935,
        evaluator_strategy: Union[str, Callable] = 'precision_recall',
        seed: Optional[int] = None,
        plot_prc: bool = False,
        verbose: int = 0,
        **kwargs
    ):
        self._graph_parser = GraphParser(attr_types, verbose=verbose)
        self._ground_truth_parser = GroundTruthParser(verbose=verbose)
        self._resolver = Resolver(
            blocking_strategy, raw_blocking=raw_blocking, alpha=alpha,
            weights=weights, attr_strategy=attr_strategy,
            rel_strategy=rel_strategy, blocking_threshold=blocking_threshold,
            bootstrap_strategy=bootstrap_strategy, raw_bootstrap=raw_bootstrap,
            edge_match_threshold=edge_match_threshold, first_attr=first_attr,
            first_attr_raw=first_attr_raw, second_attr=second_attr,
            second_attr_raw=second_attr_raw, linkage=linkage,
            similarity_threshold=similarity_threshold, seed=seed,
            plot_prc=plot_prc, verbose=verbose, **kwargs
        )
        self._evaluator = Evaluator(
            strategy=evaluator_strategy,
            verbose=verbose, **kwargs
        )
        super().__init__(verbose=verbose)
        self.attr_types = attr_types
        self.blocking_strategy = blocking_strategy
        self.raw_blocking = raw_blocking
        self.alpha = alpha
        self.weights = weights
        self.attr_strategy = attr_strategy
        self.rel_strategy = rel_strategy
        self.blocking_threshold = blocking_threshold
        self.bootstrap_strategy = bootstrap_strategy
        self.raw_bootstrap = raw_bootstrap
        self.edge_match_threshold = edge_match_threshold
        self.first_attr = first_attr
        self.first_attr_raw = first_attr_raw
        self.second_attr = second_attr
        self.second_attr_raw = second_attr_raw
        self.linkage = linkage
        self.similarity_threshold = similarity_threshold
        self.evaluator_strategy = evaluator_strategy
        self.seed = seed
        self.plot_prc = plot_prc
        self._kwargs = {
            'second_sim': 'jaro_winkler', 'stfidf_threshold': 0.5,
            'jw_prefix_weight': 0.1, 'average_method': 'max'
        }
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def attr_types(self):
        """ `~typing.Mapping`\ [`str`, `str`]: Omitted."""
        return self._attr_types

    @attr_types.setter
    def attr_types(self, value):
        self._attr_types = value
        self._graph_parser.attr_types = value

    @property
    def blocking_strategy(self):
        """ `~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`], `~typing.Mapping`\ [`str`, `str`]], `float`]: Omitted."""
        return self._blocking_strategy

    @blocking_strategy.setter
    def blocking_strategy(self, value):
        self._blocking_strategy = value
        self._resolver.blocking_strategy = value

    @property
    def raw_blocking(self):
        """ `bool`: Omitted."""
        return self._raw_blocking

    @raw_blocking.setter
    def raw_blocking(self, value):
        self._raw_blocking = value
        self._resolver.raw_blocking = value

    @property
    def alpha(self):
        """ `~typing.Union`\ [`float`, `int`]: Omitted.

        Raises:
            ValueError: If set to a value not between 0 and 1 (inclusive).
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0 or value > 1:
            raise ValueError('alpha must be between 0 and 1 (inclusive).')
        self._alpha = value
        self._resolver.alpha = value

    @property
    def weights(self):
        """ `~typing.Optional`\ [`~typing.Mapping`\ [`str`, `float`]]: Omitted.

        Raises:
            ValueError: If set to a dictionary whose values do not sum to 1.
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is not None and sum(value.values()) != 1:
            raise ValueError('Weights must sum to 1 if it is not None.')
        self._weights = value
        self._resolver.weights = value

    @property
    def attr_strategy(self):
        """ `~typing.Mapping`\ [`str`, `str`]: Omitted."""
        return self._attr_strategy

    @attr_strategy.setter
    def attr_strategy(self, value):
        self._attr_strategy = value
        self._resolver.attr_strategy = value

    @property
    def rel_strategy(self):
        """ `str`: Omitted.

        Raises:
            ValueError: If set to any value other than ``'jaccard_coef'``,
                ``'jaraccard_coef_fr'``, ``'adar_neighbor'``,
                ``'adar_neighbor_fr'``, ``'adar_attr'``, or ``'adar_attr_fr'``.
        """
        return self._rel_strategy

    @rel_strategy.setter
    def rel_strategy(self, value):
        if value not in [
            'jaccard_coef', 'jaraccard_coef_fr', 'adar_neighbor',
            'adar_neighbor_fr', 'adar_attr', 'adar_attr_fr'
        ]:
            raise ValueError(
                'rel_strategy must be one of \'jaccard_coef\', '
                '\'jaraccard_coef_fr\', \'adar_neighbor\', '
                '\'adar_neighbor_fr\', \'adar_attr\', or \'adar_attr_fr\''
            )
        self._rel_strategy = value
        self._resolver.rel_strategy = value

    @property
    def blocking_threshold(self):
        """ `~typing.Union`\ [`float`, `int`]: Omitted."""
        return self._blocking_threshold

    @blocking_threshold.setter
    def blocking_threshold(self, value):
        self._blocking_threshold = value
        self._resolver.blocking_threshold = value

    @property
    def bootstrap_strategy(self):
        """ `~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`], `~typing.Mapping`\ [`str`, `str`]], `bool`]]: Omitted."""
        return self._bootstrap_strategy

    @bootstrap_strategy.setter
    def bootstrap_strategy(self, value):
        self._bootstrap_strategy = value
        self._resolver.bootstrap_strategy = value

    @property
    def raw_bootstrap(self):
        """ `bool`: Omitted."""
        return self._raw_bootstrap

    @raw_bootstrap.setter
    def raw_bootstrap(self, value):
        self._raw_bootstrap = value
        self._resolver.raw_bootstrap = value

    @property
    def edge_match_threshold(self):
        """ `int`: Omitted."""
        return self._edge_match_threshold

    @edge_match_threshold.setter
    def edge_match_threshold(self, value):
        self._edge_match_threshold = value
        self._resolver.edge_match_threshold = value

    @property
    def first_attr(self):
        """ `~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`]], `str`]]: Omitted."""
        return self._first_attr

    @first_attr.setter
    def first_attr(self, value):
        self._first_attr = value
        self._resolver.first_attr = value

    @property
    def first_attr_raw(self):
        """ `bool`: Omitted."""
        return self._first_attr_raw

    @first_attr_raw.setter
    def first_attr_raw(self, value):
        self._first_attr_raw = value
        self._resolver.first_attr_raw = value

    @property
    def second_attr(self):
        """ `~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`]], `str`]]: Omitted."""
        return self._second_attr

    @second_attr.setter
    def second_attr(self, value):
        self._second_attr = value
        self._resolver.second_attr = value

    @property
    def second_attr_raw(self):
        """ `bool`: Omitted."""
        return self._second_attr_raw

    @second_attr_raw.setter
    def second_attr_raw(self, value):
        self._second_attr_raw = value
        self._resolver.second_attr_raw = value

    @property
    def linkage(self):
        """ `str`: Omitted.

        Raises:
            ValueError: If set to any value other than ``'min'``, ``'max'``, or
                ``'average'``.
        """
        return self._linkage

    @linkage.setter
    def linkage(self, value):
        if value not in ['min', 'max', 'average']:
            raise ValueError(
                'linkage must be one of \'min\', \'max\', or \'average\'.'
            )
        self._linkage = value
        self._resolver.linkage = value

    @property
    def similarity_threshold(self):
        """ `float`: Omitted."""
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value):
        self._similarity_threshold = value
        self._resolver.similarity_threshold = value

    @property
    def evaluator_strategy(self):
        """ `str`: Omitted

        Raises:
            ValueError: If set to any value other than a callable,
                ``'precision_recall'``, ``'ami'``, or ``'v_measure'``.
        """
        return self._evaluator_strategy

    @evaluator_strategy.setter
    def evaluator_strategy(self, value):
        if (
            not callable(value)
            and value not in ['precision_recall', 'ami', 'v_measure']
        ):
            raise ValueError(
                'evaluator_strategy must be one of '
                '\'precision_recall\', \'ami\', or \'v_measure\'.'
            )
        self._evaluator_strategy = value
        self._evaluator.strategy = value

    @property
    def seed(self):
        """ `~typing.Optional`\ [`int`]: Omitted."""
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._resolver.seed = value

    @property
    def plot_prc(self):
        """ `bool`: Omitted"""
        return self._plot_prc

    @plot_prc.setter
    def plot_prc(self, value):
        self._plot_prc = value
        self._resolver.plot_prc = value

    @property
    def second_sim(self):
        """ `str`: Refer to ``kwargs`` parameter above.

        Raises:
            ValueError: If set to any value other than ``'jaro_winkler'``,
                ``'jaro'``, or ``'scaled_lev'``.
        """
        return self._kwargs['second_sim']

    @second_sim.setter
    def second_sim(self, value):
        if value not in ['jaro_winkler', 'jaro', 'scaled_lev']:
            raise ValueError(
                'second_sim must be one of '
                '\'jaro_winkler\', \'jaro\', or \'scaled_lev\'.'
            )
        self._kwargs['second_sim'] = value
        self._resolver.kwargs['second_sim'] = value
        self._evaluator.kwargs['second_sim'] = value

    @property
    def stfidf_threshold(self):
        """ `float`: Refer to ``kwargs`` parameter above."""
        return self._kwargs['stfidf_threshold']

    @stfidf_threshold.setter
    def stfidf_threshold(self, value):
        self._kwargs['stfidf_threshold'] = value
        self._resolver.kwargs['stfidf_threshold'] = value
        self._evaluator.kwargs['stfidf_threshold'] = value

    @property
    def jw_prefix_weight(self):
        """ `float`: Refer to ``kwargs`` parameter above."""
        return self._kwargs['jw_prefix_weight']

    @jw_prefix_weight.setter
    def jw_prefix_weight(self, value):
        self._kwargs['jw_prefix_weight'] = value
        self._resolver.kwargs['jw_prefix_weight'] = value
        self._evaluator.kwargs['jw_prefix_weight'] = value

    @property
    def average_method(self):
        """ `str`: Refer to ``kwargs`` parameter above.

        Raises:
            ValueError: If set to any value other than ``'min'``,
                ``'geometric'``, ``'arithmetic'``, or ``'max'``
        """
        return self._kwargs['average_method']

    @average_method.setter
    def average_method(self, value):
        if value not in ['min', 'geometric', 'arithmetic', 'max']:
            raise ValueError(
                'average_method must be one of '
                '\'min\', \'geometric\', \'arithmetic\', \'max\'.'
            )
        self._kwargs['average_method'] = value
        self._resolver.kwargs['average_method'] = value
        self._evaluator.kwargs['average_method'] = value

    def _set_verbose(self, value: int) -> None:
        """ Override the parent verbose setter.

        Update the ``verbose`` attributes of objectes contained in this class
        when its own ``verbose`` attribute changes.

        Args:
            value: The verbose value to be set to.
        """
        self._graph_parser.verbose = value
        self._ground_truth_parser.verbose = value
        self._resolver.verbose = value
        self._evaluator.verbose = value

    @logtime('Time taken for the whole resolution process')
    def resolve(self, graph_path: str) -> collections.OrderedDict:
        """ Resolve entities in the given data and return the entity mapping.

        Args:
            graph_path: The path to the input data for entity resolution. The
                data file has to **strictly follow** the format as described in
                :doc:`../tutorial`.

        Returns:
            Mapping reference ids to cluster ids. The dictionary is sorted
            (key-value pairs are inserted) in ascending order of reference ids.

        Raises:
            ValueError: If ``plot_prc`` is set to ``True``.
        """
        if self.plot_prc:
            raise ValueError(
                'Cannot plot precision-recall curve when only resolving'
                'entities. Use resolve_and_eval instead.'
            )
        graph = self._graph_parser.parse(graph_path)
        resolved_mapping = self._resolver.resolve(graph)
        self._resolver.log_time()
        return resolved_mapping

    def evaluate(
        self, ground_truth_path: str, resolved_mapping: Mapping
    ) -> Any:
        """ Evaluate the resolved result using ground truth data.

        Args:
            ground_truth_path: The path to ground truth data used for
                evaluation. The data file has to **strictly follow** the format
                as described in :doc:`../tutorial`.
            resolved_mapping: Mapping reference id to cluster id. The reference
                ids must correspond to those in ground truth data, while their
                cluster ids may differ.

        Returns:
            * If ``evaluator_strategy`` is set to ``'precision_recall'``, the
              result is a tuple of three numbers representing precision,
              recall, and f1 in the order.
            * If ``strategy`` is set to ``'ami'``, or ``'v_measure'``, the
              result is a single number representing the score computed as
              specified by ``evaluator_strategy``.
            * If ``strategy`` is a user-defined function, the return is the
              same as that function.

        Raises:
            ValueError: If ``plot_prc`` is set to ``True``.

        See Also:
            It ultimately calls the ``evaluate`` method of an ``Evaluator``
            object in :doc:`entity_resolver.core.evaluator`.
        """
        if self.plot_prc:
            raise ValueError(
                'Cannot plot precision-recall curve when only evaluating the'
                'performance. Use resolve_and_eval instead.'
            )
        ground_truth = self._parse_ground_truth(ground_truth_path)
        return self._evaluator.evaluate(ground_truth, resolved_mapping)

    def resolve_and_eval(
        self, ground_truth_path: str, graph_path: str
    ) -> Union[Any, Tuple[Any, List[Tuple[float, float]]]]:
        """ Resolve entities in the given data and evaluate the result.

        Args:
            ground_truth_path: The path to ground truth data used for
                evaulation. The data file has to **strictly follow** the format
                as described in :doc:`../tutorial`.
            graph_path: The path to the input data for entity resolution. The
                data file has to **strictly follow** the format as described in
                :doc:`../tutorial`.

        Returns:
            This depends on the values of ``evaluator_strategy`` and
            ``plot_prc``.

            * If ``evaluator_strategy == 'precision_recall'``:

              * If ``plot_prc == False``, the result is a tuple of three
                numbers representing precision, recall, and f1 in the order.
              * If ``plot_prc == True``, the result is a tuple of two objects:

                * The first one is a tuple of three numbers representing
                  precision, recall, and f1 in the order, same as above.
                * The second is a list of tuples, each consisting of the
                  precision and recall value at different steps of the
                  algorithm (reflecting the behavior with different
                  ``similarity_threshold``).
            * If ``evaluator_strategy != 'precision_recall'``:

              * If ``plot_prc == False``, the result is a single number
                representing the score computed as specified by
                ``evaluator_strategy`` if the strategy is not a user-defined
                function, or the score returned by the user-defined function.
              * If ``plot_prc == True``, the result is a tuple of two objects:

                * The first one is a single number same as in the
                  ``plot_prc == False`` case.
                * The second one is a list of tuples same as in the second
                  object in the ``evaluator_strategy == 'precision_recall'``
                  and ``plot_prc == True`` case.
        """
        ground_truth = self._parse_ground_truth(ground_truth_path)
        graph = self._graph_parser.parse(graph_path)
        resolver_res = self._resolver.resolve(graph, ground_truth)
        self._resolver.log_time()
        if self.plot_prc:
            resolved_mapping, prc_list = resolver_res
            self._plot(prc_list)
            score = self._evaluator.evaluate(ground_truth, resolved_mapping)
            return score, prc_list
        else:
            resolved_mapping = resolver_res
            return self._evaluator.evaluate(ground_truth, resolved_mapping)

    def _parse_ground_truth(
        self, ground_truth_path: str
    ) -> collections.OrderedDict:
        """ A helper method for parsing the ground truth data.

        Args:
            ground_truth_path: The path to ground truth data. The data file has
                to **strictly follow** the format as described in
                :doc:`../tutorial`.

        Returns:
            Mapping reference ids to ground truth cluster ids. The dictionary
            is sorted (key-value pairs are inserted) in ascending order of node
            id.

        See Also:
            It ultimately calls the ``parse`` method of the
            ``GroundTruthParser`` object in
            :doc:`entity_resolver.parser.ground_truth_parser`.
        """
        return self._ground_truth_parser.parse(ground_truth_path)

    def _plot(self, prc_list: List[Tuple[float, float]]) -> None:
        """ A helper method for plotting the precision-recall curve.

        Args:
            prc_list: A list consisting of (precision, recall) tuples computed
                at different steps of the algorithm (reflecting the behavior
                with different ``similarity_threshold``).
        """
        precisions, recalls = zip(*prc_list)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.title('Precision-Recall Curve')
        plt.plot(list(precisions), list(recalls))
        plt.show()
