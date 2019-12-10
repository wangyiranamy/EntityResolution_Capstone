""" Contains a single class that implements the entity resolution algorithm.

Related hyperparameters are inputs to the `Resolver` object during
instantiation. That object will execute the core entity resolution algorithm
if its ``resolve`` method is called, which includes blocking, relational
bootstrapping, and clustering. For details of the algorithm and the
hyperparameters, please refer to :doc:`../advanced_guide`.

Example:

    >>> from entity_resolver.core import Resolver
    >>> resolver1 = Resolver(plot_prc=False)  # No plot
    >>> resolver.resolve(graph)  # graph is a Graph object
    >>> resolver2 = Resolver(plot_prc=True)  # Plot
    >>> resolver2.resolve(graph, ground_truth)  # Need ground_truth to plot
"""

import itertools
import collections
import random
import multiprocessing
from typing import (
    Hashable, Union, Optional, Collection,
    List, Tuple, Mapping, Set, Callable, Dict
)
import numpy as np
from .graph import Graph, Node
from .utils import (
    SimFuncFactory, ClusteringMetrics, DSU, PriorityQueue, SimilarityEntry,
    WithLogger, logtime, timeit
)


class Resolver(WithLogger):
    """ Execute the core entity resolution program.

    Most of the arguments and paramters are duplicates of those in
    :doc:`entity_resolver.main`. Their descriptions are simply copied below.
    However, these attributes are not encapsulated using ``@property`` and
    hence not protected against invalid attribute assignments.

    Args:
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
            Valid values are ``'stfidf'``, ``'jaro_winkler'``, and ``'jaro'``.
            Refer to :doc:`../advanced_guide` for more details.
        rel_strategy: Name of the strategy to compute relational similarity.
            Valid values are ``'jaccard_coef'``, ``'jaraccard_coef_fr'``,
            ``'adar_neighbor'``, ``'adar_neighbor_fr'``, ``'adar_attr'``,
            ``'adar_attr_fr'``. Refer to :doc:`../advanced_guide` for more
            details.
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

    Note:
        * All attributes below which have the same names as in the above
          parameters section are essentially the same. Besides, the ``_kwargs``
          attribute is also the same as the ``kwrags`` parameter. Their
          documentations are therefore omitted.
        * Some attributes are initialized during the process of the entity
          resolution algorithm as cached values for furture use. They are
          marked as '*(cache)*' in the following attribute documentations.
        * Since clusters are simply represented by their ids, the phrase
          'cluster id' and 'cluster' may be used interchangeably throughout
          this module's documentation

    Todo:
        Remove all cached attributes at the end of the ``resolve`` method.

    Attributes:
        blocking_strategy (`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`], `~typing.Mapping`\ [`str`, `str`]], `float`]):            Omitted.
        raw_blocking (`bool`): Omitted.
        alpha (`~typing.Union`\ [`float`, `int`]): Omitted.
        weights (`~typing.Optional`\ [`~typing.Mapping`\ [`str`, `float`]]):
            Omitted.
        attr_strategy (`~typing.Mapping`\ [`str`, `str`]): Omitted.
        rel_strategy (`~typing.Optional`\ [`str`]): Omitted.
        blocking_threshold (`~typing.Union`\ [`float`, `int`]): Omitted.
        bootstrapping_strategy (`~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`], `~typing.Mapping`\ [`str`, `str`]], `bool`]]):
            Omitted.
        raw_bootstrap (`bool`): Omitted.
        edge_match_threshold (`int`): Omitted.
        first_attr (`~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`]], `str`]]):
            Omitted.
        first_attr_raw (`bool`): Omitted.
        second_attr (`~typing.Optional`\ [`~typing.Callable`\ [[`~typing.Mapping`\ [`str`, `str`]], `str`]]):
            Omitted.
        second_attr_raw (`bool`): Omitted.
        linkage (`str`): Omitted.
        similarity_threshold (`float`): Omitted.
        seed (`~typing.Optional`\ [`int`]): Omitted.
        plot_prc (`bool`): Omitted.
        verbose (`int`): Omitted.
        time_dict (`~typing.Dict`\ [`str`, `~typing.List`\ [`~typing.Union`\ [`float`, `int`]]]):
            A `~collections.defaultdict` object mapping method names to a list
            of two entries:

            [<number of times called>, <total time spent by all calls>].

            It is used to store the time statistics of methods decorated with
            `~entity_resolver.core.utils.timeit`.
        kwargs: Omitted.
        _sim_func_producers (`~typing.Callable`\ [`str`, `~typing.Callable`]):
            Mapping from strategy strings to corresponding similarity functions
            producers.
        _default_strategies (`~typing.Callable`\ [`str`, `str`]): Mapping
            attribute types to default strategy strings. The default strategy
            strings are used if no strategy strings are specified in
            ``attr_strategy``.
        _prc_list (`~typing.List`\ [`~typing.Tuple`\ [`float`, `float`]]):
            *(cache)*. It is used to store the precision and recall scores
            computed during the entity resolution algorithm if ``plot_prc`` is
            set to ``True``. Each item of the list is (precision, recall). It
            is not initialized if ``plot_prc`` is ``False``.
        _graph (`~entity_resolver.core.graph.Graph`): *(cache)*. The graph used
            for the entity resolution.
        _attr_weights (`~typing.Dict`\ [`str`, `float`]): *(cache)*. Mapping
            attribute names to their ratio in computing attribute similarities.
        _attr_funcs (`~typing.Dict`\ [`str`, `~typing.Callable`]): *(cache)*.
            Mapping attribute names to functions that compute the similarities
            of two values in the attributes.
        _rel_func (`~typing.Callable`\ [[`~typing.Collection`, `~typing.Collection`, `~typing.Callable`\ [[`~typing.Hashable`], `float`]], `float`]`):
            *(cache)*. The relational similarity function used to compute the
            relational similarity between two clusters.
        _ambiguities (`~typing.Dict`\ [`~entity_resolver.core.graph.Node`, float]):
            *(cache)*. Mapping references to their attribute ambiguities.
        _use_nbr_cache (`bool`): *(cache)*. Indicate if the cached
            ``_cluster_neighbors`` should be used during computation of
            relational similarities. It can only be used when the computation
            does not need to count multiplicities of neighboring clusters.
        _use_amb_type (`~typing.Optional`\ [`str`]): *(cache)*. Indicate which
            type of ambiguity values are required during computation of
            similarities. Valid values are ``'neighbor'``, ``'attr'``, and
            ``None``, which refer to neighbor ambiguities, attribute
            ambiguities, and not used respectively.
        _use_ambiguities (`bool`): *(cache)*. Indicate if the similarity
            computation requires computation of attribute ambiguities.
        _clusters (`~typing.Dict`\ [`~typing.Hashable`, `~typing.Set`\ [`~entity_resolver.core.graph.Node`]]):
            *(cache)*. A `~collections.defaultdict` mapping cluster ids to sets
            of references belonging to the clusters.
        _inv_clusters (`~typing.Dict`\ [`~entity_resolver.core.graph.Node`, `~typing.Hashable`]):
            *(cache)*. An `~collections.OrderedDict` mapping references to the
            cluster ids they belong to. The dictionary is sorted (key-value
            pairs are inserted) in ascending order of reference ids.
        _cluster_neighbors (`~typing.Dict`\ [`~typing.Hashable`, `~typing.Set`]):
            *(cache)*. Mapping cluster ids to its neighboring cluster ids.
        _sim_clusters_pool (`~typing.Set`\ [`~typing.Tuple`\ [`~typing.Hashable`, `~typing.Hashable`]]):
            *(cache)*. A set of potentially similar pairs of clusters.
    """

    def __init__(
        self,
        blocking_strategy: Callable[
            [Mapping[str, str], Mapping[str, str]], float
        ],
        raw_blocking: bool = False,
        alpha: Union[float, int] = 0,
        weights: Optional[Mapping[str, float]] = None,
        attr_strategy: Mapping[str, str] = dict(),
        rel_strategy: Optional[str] = None,
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
        seed: Optional[int] = None,
        plot_prc: bool = False,
        verbose: int = 0,
        **kwargs
    ):
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
        self.plot_prc = plot_prc
        self.seed = seed
        self.kwargs = kwargs
        self._sim_func_producers = {
            'stfidf': SimFuncFactory.produce_stfidf,
            'jaro_winkler': SimFuncFactory.produce_jaro_winkler,
            'jaro': SimFuncFactory.produce_jaro,
            'jaccard_coef': SimFuncFactory.produce_jaccard_coef,
            'jaccard_coef_fr': SimFuncFactory.produce_jaccard_coef_fr,
            'adar_neighbor': SimFuncFactory.produce_adar_neighbor,
            'adar_neighbor_fr': SimFuncFactory.produce_adar_neighbor_fr,
            'adar_attr': SimFuncFactory.produce_adar_attr,
            'adar_attr_fr': SimFuncFactory.produce_adar_attr_fr
        }
        self._default_strategies = {
            'text': 'stfidf',
            'person_entity': 'jaro_winkler',
            'relation': 'jaccard_coef'
        }
        self.time_dict = collections.defaultdict(lambda: [0, 0])
        random.seed(seed)
        super().__init__(verbose)

    @logtime('Time taken for the core resolution algorithm')
    def resolve(
        self, graph: Graph,
        ground_truth: Optional[collections.OrderedDict] = None
    ) -> Union[
        collections.OrderedDict,
        Tuple[collections.OrderedDict, List[Tuple[float, float]]]
    ]:
        """ Resolve the entities in given graph.

        Args:
            graph: The reference graph for entity resolution.
            ground_truth: The parsed ground truth data corresponding to the
                graph. It maps reference ids to ground truth cluster ids. The
                dictionary is sorted (key-value pairs are inserted) in
                ascending order of reference ids. It must be present and is
                only used when ``plot_prc`` attribute is set to ``True``. It is
                used to plot the precison-recall graph by computing the scores
                during the entity resolution process.

        Raises:
            ValueError: If ``ground_truth`` is ``None`` while ``plot_prc`` is
                ``True``.

        Returns:
            * If ``plot_prc`` is ``False``, the return is the mapping from
              reference ids to cluster ids.
            * If ``plot_prc`` is ``True``, the return is a tuple of two items.
              The first item is a mapping from reference ids to cluster ids,
              same as in ``plot_prc`` is ``False`` case. The second item is a
              list of (precision, recall) scores computed during the entity
              resolution process for plotting the precision-recall curve.
        """
        self._init_cache(graph)
        buckets = self._blocking(graph)
        self._relational_boostrapping(buckets)
        if ground_truth is not None:
            self._cluster_nodes(list(ground_truth.values()))
        elif self.plot_prc:
            raise ValueError(
                'ground_truth cannot be None if plot_prc is True.'
            )
        else:
            self._cluster_nodes()
        resolved_mapping = collections.OrderedDict()
        for node, cluster in self._inv_clusters.items():
            resolved_mapping[node.id] = cluster
        self._clear_cache()
        if self.plot_prc:
            return resolved_mapping, self._prc_list
        return resolved_mapping

    @logtime('Time taken for blocking')
    def _blocking(self, graph: Graph) -> List[List[Node]]:
        """ Find potentially similar reference pairs via blocking.

        Args:
            graph: The reference graph for entity resolution.

        Returns:
            Each item in the list is a bucket contained references that has
            the potential to end up being the same entity.
        """
        buckets = list()
        nodes = random.sample(graph.nodes, k=len(graph.nodes))
        for node in nodes:
            assigned = False
            for bucket in buckets:
                if self.raw_blocking:
                    rep_attrs = bucket[0].raw_attr_vals
                    node_attrs = node.raw_attr_vals
                else:
                    rep_attrs = bucket[0].attr_vals
                    node_attrs = node.attr_vals
                dist = self.blocking_strategy(rep_attrs, node_attrs)
                if dist < self.blocking_threshold:
                    bucket.append(node)
                    assigned = True
            if not assigned:
                buckets.append([node])
        self.logger.info(f'Number of buckets in blocking: {len(buckets)}')
        return buckets

    @logtime('Time taken for relational bootstrapping')
    def _relational_boostrapping(self, buckets: List[List[Node]]) -> None:
        """ Relational bootstrap by merging highly similar reference pairs.

        Args:
            buckets: The list of buckets containing potentially similar
                references. This is the return of the ``_blocking`` method.
        """
        self._init_clusters(buckets)
        self._init_sim_clusters_pool(buckets)

    @timeit
    def _init_clusters(self, buckets: List[List[Node]]) -> None:
        """ Merge references in the same buckets to the same cluster.

        Args:
            buckets: The list of buckets containing potentially similar
                references. This is the return of the ``_blocking`` method.
        """
        nodes = self._graph.nodes
        clusters = collections.defaultdict(set)
        inv_clusters = collections.OrderedDict()
        dsu = DSU(nodes)
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                exact_match = self._check_exact_match(node1, node2)
                edge_check_passed = self._check_edge_match(node1, node2)
                if exact_match and edge_check_passed:
                    dsu.union(node1, node2)
        for node in sorted(dsu.items, key=lambda node: node.id):
            parent = dsu.find(node)
            clusters[parent.id].add(node)
            inv_clusters[node] = parent.id
        self.logger.info(f'Number of initial clusters: {len(clusters)}')
        self._clusters = clusters
        self._inv_clusters = inv_clusters
        cluster_neighbors = {
            cluster: set(self._get_cluster_neighbors(cluster))
            for cluster in clusters
        }
        self._cluster_neighbors = cluster_neighbors

    @timeit
    def _init_sim_clusters_pool(self, buckets: List[List[Node]]) -> None:
        """ Put pairs of potentially similar clusters in a set.

        Args:
            buekcts: The list of buckets containing potentially similar
                references. This is the return of the ``_blocking`` method.
        """
        sim_clusters_pool = set()
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                cluster1 = self._inv_clusters[node1]
                cluster2 = self._inv_clusters[node2]
                if (
                    cluster1 != cluster2
                    and cluster1 not in self._cluster_neighbors[cluster2]
                    and (cluster1, cluster2) not in sim_clusters_pool
                    and (cluster2, cluster1) not in sim_clusters_pool
                ):
                    sim_clusters_pool.add((cluster1, cluster2))
        num_pairs = len(sim_clusters_pool)
        self.logger.info(f'Number of initial similar pairs: {num_pairs}')
        self._sim_clusters_pool = sim_clusters_pool

    @timeit
    def _check_exact_match(self, node1: Node, node2: Node) -> bool:
        """ Check if attributes of two nodes match exactly.

        * If ``bootstrap_strategy`` is **not** ``None``, then the strategy
          function is called to check if the references match exactly.
          Otherwise two references are considered a exact match if and only if
          all attributes are exactly the same.
        * If ``raw_bootstrap`` is ``True``, then the raw values of attributes
          of each node is compared. Otherwise the processed values are
          compared.

        Args:
            node1: One of the two nodes to be compared.
            node2: One of the two nodes to be compared.

        Return:
            If attributes of the two input nodes should be considered as an
            exact match.
        """
        if self.bootstrap_strategy is not None:
            return self.bootstrap_strategy(node1.attr_vals, node2.attr_vals)
        for name in node1.attr_vals:
            if self.raw_bootstrap:
                value1 = node1.raw_attr_vals[name]
                value2 = node2.raw_attr_vals[name]
            else:
                value1 = node1.attr_vals[name]
                value2 = node2.attr_vals[name]
            if value1 != value2:
                return False
        return True

    @timeit
    def _check_edge_match(self, node1: Node, node2: Node) -> bool:
        """ Check if edges of the two nodes pass the matching threshold.

        Args:
            node1: One of the two nodes to be compared.
            node2: One of the two nodes to be compared.

        Returns:
            If edges of the two input nodes pass the matching threshold.
        """
        nbr_nodes1 = self._graph.get_neighbors(node1)
        nbr_nodes2 = self._graph.get_neighbors(node2)
        count = 0
        added_pairs = set([(node1, node2), (node2, node1)])
        for nbr_node1, nbr_node2 in itertools.product(nbr_nodes1, nbr_nodes2):
            if (
                self._check_exact_match(nbr_node1, nbr_node2)
                and (nbr_node1, nbr_node2) not in added_pairs
            ):
                count += 1
                added_pairs.update(set([
                    (nbr_node1, nbr_node2),
                    (nbr_node2, nbr_node1)
                ]))
            if count >= self.edge_match_threshold:
                return True
        return False

    @logtime('Time taken for clustering')
    def _cluster_nodes(
        self, ground_truth: Optional[collections.OrderedDict] = None
    ) -> None:
        """ Execute the main clustering algorithm.

        Args:
            ground_truth: Mapping reference ids to ground truth cluster ids.
                The dictionary is sorted (key-value pairs are inserted) in
                ascending order of reference ids.
        """
        sim_clusters = collections.defaultdict(set)
        cluster_entries = collections.defaultdict(dict)
        self.logger.debug('Start building priority queue.')
        pqueue = self._init_queue_entries(cluster_entries, sim_clusters)
        counter = 0
        self.logger.debug('Finish building priority queue. Start popping.')
        if self.plot_prc:
            # Add a point to be plotted every <step> time to reduce computation
            # 1600 is completely empirical and can be arbitrarily modified
            step = len(pqueue) // 1600
        while pqueue:
            entry = pqueue.pop()
            counter += 1
            if (counter % 10000 == 0):
                self.logger.debug(f'Number of pops from queue: {counter}')
            if entry.similarity < self.similarity_threshold:
                break
            cluster1 = entry.cluster1
            cluster2 = entry.cluster2
            if cluster1 in self._cluster_neighbors[cluster2]:
                sim_clusters[cluster1].remove(cluster2)
                sim_clusters[cluster2].remove(cluster1)
                cluster_entries[cluster1].pop(cluster2)
                cluster_entries[cluster2].pop(cluster1)
                continue
            # merge the smaller cluster to the larger one
            if len(self._clusters[cluster1]) < len(self._clusters[cluster2]):
                cluster1, cluster2 = cluster2, cluster1
            self._merge_clusters(cluster1, cluster2, sim_clusters)
            self._update_pqueue(
                cluster1, cluster2, pqueue,
                cluster_entries, sim_clusters
            )
            if self.plot_prc and counter % step == 0:
                self._add_pr(ground_truth)
        self.logger.debug(f'Total number of pops from queue: {counter}')
        if self.plot_prc:
            num_pr = len(self._prc_list)
            self.logger.debug(f'Number of precision-recall pairs: {num_pr}')

    @timeit
    def _init_queue_entries(
        self,
        cluster_entries: Mapping[Hashable, Mapping[Hashable, SimilarityEntry]],
        sim_clusters: Mapping[Hashable, Set],
    ) -> PriorityQueue:
        """ Initialize a priority queue with potentially similar cluster pairs.

        Args:
            cluster_entries: A two-dimensional mapping that maps cluster pairs
                to their corresponding
                `~entity_resolver.core.utils.SimilarityEntry` objects. For
                example, ``cluster_entries[1][2]`` is a
                `~entity_resolver.core.utils.SimilarityEntry` object that
                contains the similarity of cluster with id ``1`` and cluster
                with id  ``2``. It is used to keep track of all entries in the
                priority queue for update or removal.
            sim_clusters: Mapping cluster ids to a set of clusters that are
                potentially similar to them.

        Returns:
            A priority queue consisting all the
            `~entity_resolver.core.utils.SimilarityEntry` objects of
            potentially similar cluster pairs.
        """
        sim_clusters_pool = self._sim_clusters_pool
        entries = list()
        while sim_clusters_pool:
            cluster1, cluster2 = sim_clusters_pool.pop()
            similarity = self._calc_similarity(cluster1, cluster2)
            entry = SimilarityEntry(cluster1, cluster2, similarity)
            sim_clusters[cluster1].add(cluster2)
            sim_clusters[cluster2].add(cluster1)
            cluster_entries[cluster1][cluster2] = entry
            cluster_entries[cluster2][cluster1] = entry
            entries.append(entry)
        return PriorityQueue(entries)

    @timeit
    def _merge_clusters(
        self,
        cluster1: Hashable,
        cluster2: Hashable,
        sim_clusters: Mapping[Hashable, Set]
    ) -> None:
        """ Merge ``cluster2`` into ``cluster1``.

        This method also modifies related entries in ``sim_clusters`` and the
        cached ``_cluster_neighbors`` attribute.

        Args:
            cluster1: The cluster to be merged into.
            cluster2: The cluster to be merged.
            sim_clusters: Mapping cluster ids to a set of clusters that are
                potentially similar to them. Same as in the
                ``_init_queue_entries`` method.
        """
        self._clusters[cluster1] |= self._clusters[cluster2]
        self._clusters.pop(cluster2)
        cluster_neighbors = self._cluster_neighbors
        for node in self._clusters[cluster1]:
            self._inv_clusters[node] = cluster1
        self._update_adj_set(cluster1, cluster2, sim_clusters, True)
        self._update_adj_set(cluster1, cluster2, cluster_neighbors, False)

    @timeit
    def _update_adj_set(
        self,
        cluster1: Hashable,
        cluster2: Hashable,
        adjacency_set: Mapping[Hashable, Set],
        remove_self: bool
    ) -> None:
        """ Help update related entries when merging clusters.

        Args:
            cluster1: The cluster to be merged into.
            cluster2: The cluster to be merged.
            adjacency_set: an abstraction of ``sim_clusters`` and the
                ``_cluster_neighbors`` attribute, mapping a cluster to a set of
                clusters.
            remove_self: If the cluster itself should be removed from the set
                it is mapped to. Moreover, if it is ``False``, it means that
                each cluster is guaranteed to be contained in the set it is
                mapped to. This differentiates the behaviors of
                ``sim_clusters`` and ``_cluster_neighbors``.
        """
        for cluster in adjacency_set[cluster2]:
            if cluster != cluster2:
                adjacency_set[cluster].remove(cluster2)
                adjacency_set[cluster].add(cluster1)
        adjacency_set[cluster1] |= adjacency_set[cluster2]
        if remove_self:
            adjacency_set[cluster1].remove(cluster1)
        else:
            adjacency_set[cluster1].remove(cluster2)
        adjacency_set.pop(cluster2)

    @timeit
    def _update_pqueue(
        self,
        cluster1: Hashable,
        cluster2: Hashable,
        pqueue: PriorityQueue,
        cluster_entries: Mapping[Hashable, Mapping[Hashable, SimilarityEntry]],
        sim_clusters: Mapping[Hashable, Set]
    ) -> None:
        """ Update entries in the priority queue after merge two clusters.

        Args:
            cluster1: The cluster to be merged into.
            cluster2: The cluster to be merged.
            pqueue: The priority queue of
                `~entity_resolver.core.utils.SimilarityEntry` objects.
            cluster_entries: A two-dimensional mapping that maps cluster pairs
                to their corresponding
                `~entity_resolver.core.utils.SimilarityEntry` objects. Same as
                in the ``_init_queue_entries`` method.
            sim_clusters: Mapping cluster ids to a set of clusters that are
                potentially similar to them. Same as in the
                ``_init_queue_entries`` method.
        """
        for entry in cluster_entries[cluster1].values():
            pqueue.discard(entry)
        for entry in cluster_entries[cluster2].values():
            pqueue.discard(entry)
        cluster_entries[cluster1].clear()
        cluster_entries.pop(cluster2)
        added_pairs = set()
        self._add_sim_entries(
            cluster1, pqueue, cluster_entries,
            sim_clusters, added_pairs
        )
        self._update_nbr_entries(
            cluster1, pqueue, cluster_entries,
            sim_clusters, added_pairs
        )

    @timeit
    def _add_pr(self, ground_truth: collections.OrderedDict) -> None:
        """ Compute and add (precision, recall) to the ``_prc_list`` attribute.

        This method is called each fixed times an entry is popped from the
        priority queue if ``plot_prc`` is ``True``.

        Args:
            ground_truth: Mapping reference ids to ground truth cluster ids.
                The dictionary is sorted (key-value pairs are inserted) in
                ascending order of reference ids.
        """
        precision, recall, _ = ClusteringMetrics.precision_recall(
            ground_truth,
            list(self._inv_clusters.values()),
            log=False
        )
        self._prc_list.append((precision, recall))

    @timeit
    def _add_sim_entries(
        self,
        cluster: Hashable,
        pqueue: PriorityQueue,
        cluster_entries: Mapping[Hashable, Mapping[Hashable, SimilarityEntry]],
        sim_clusters: Mapping[Hashable, Set],
        added_pairs: Set[Tuple[Hashable, Hashable]]
    ) -> None:
        """ Add new entries to the priority queue after merging two clusters.

        The added entries are clusters potentially similar to the original two
        clusters before merging paired with the merged cluster. Their
        similarities are re-computed as well.

        Args:
            cluster: The merged cluster.
            pqueue: The priority queue of
                `~entity_resolver.core.utils.SimilarityEntry` objects.
            cluster_entries:  A two-dimensional mapping that maps cluster pairs
                to their corresponding
                `~entity_resolver.core.utils.SimilarityEntry` objects. Same as
                in the ``_init_queue_entries`` method.
            sim_clusters: Mapping cluster ids to a set of clusters that are
                potentially similar to them. Same as in the
                ``_init_queue_entries`` method.
            added_pairs: Store pairs of clusters that have already been added
                into the priority queue to avoid duplicates. Each time an entry
                is added into the priority queue, the tuple of the two clusters
                with both permutations are added into it.
        """
        for sim_cluster in sim_clusters[cluster]:
            similarity = self._calc_similarity(cluster, sim_cluster)
            entry = SimilarityEntry(cluster, sim_cluster, similarity)
            pqueue.push(entry)
            cluster_entries[cluster][sim_cluster] = entry
            cluster_entries[sim_cluster][cluster] = entry
            added_pairs.update(set([
                (cluster, sim_cluster),
                (sim_cluster, cluster)
            ]))

    @timeit
    def _update_nbr_entries(
        self,
        cluster: Hashable,
        pqueue: PriorityQueue,
        cluster_entries: Mapping[Hashable, Mapping[Hashable, SimilarityEntry]],
        sim_clusters: Mapping[Hashable, Set],
        added_pairs: Set[Tuple[Hashable, Hashable]]
    ) -> None:
        """ Update entries in the priority queue after merging two clusters.

        An entry is updated if it consists of a pair of a neighboring cluster
        of the merged cluster and a cluster potentially similar to that
        neighbor. Neighboring clusters of the merged cluster are neighbors of
        one of the original two clusters before merging.

        Args:
            cluster: The merged cluster.
            pqueue: The priority queue of
                `~entity_resolver.core.utils.SimilarityEntry` objects.
            cluster_entries:  A two-dimensional mapping that maps cluster pairs
                to their corresponding
                `~entity_resolver.core.utils.SimilarityEntry` objects. Same as
                in the ``_init_queue_entries`` method.
            sim_clusters: Mapping cluster ids to a set of clusters that are
                potentially similar to them. Same as in the
                ``_init_queue_entries`` method.
            added_pairs: Store pairs of clusters that have already been added
                into the priority queue to avoid duplicates. Each time an entry
                is added into the priority queue, the tuple of the two clusters
                with both permutations are added into it.
        """
        for cluster_nbr in self._cluster_neighbors[cluster]:
            for sim_cluster in sim_clusters[cluster_nbr]:
                if (cluster_nbr, sim_cluster) in added_pairs:
                    continue
                similarity = self._calc_similarity(cluster_nbr, sim_cluster)
                entry = SimilarityEntry(cluster_nbr, sim_cluster, similarity)
                old_entry = cluster_entries[cluster_nbr][sim_cluster]
                pqueue.update(old_entry, entry)
                cluster_entries[cluster_nbr][sim_cluster] = entry
                cluster_entries[sim_cluster][cluster_nbr] = entry
                added_pairs.update(set([
                    (cluster_nbr, sim_cluster),
                    (sim_cluster, cluster_nbr)
                ]))

    def _init_cache(self, graph: Graph) -> None:
        """ Initialize some cache attributes.

        Args:
            graph: The reference graph for entity resolution.
        """
        self._graph = graph
        parsed_result = self._parse_strategy()
        self._attr_weights, self._attr_funcs, self._rel_func = parsed_result
        self._init_ambiguities()
        if self.plot_prc:
            self._prc_list = list()

    def _parse_strategy(
        self
    ) -> Tuple[
        Dict[str, float],
        Dict[str, Callable],
        Callable[[Collection, Collection, Callable[[Hashable], float]], float]
    ]:
        """ Parse weights and strategy attributes to create relevant functions.

        Returns:
            A tuple of three items in the following order

            #. Mapping attribute names to their ratio in computing
               attribute similarities.
            #. Mapping attribute names to functions that compute the
               similarities of two values in the attributes.
            #. The relational similarity function used to compute the
               relational similarity between two clusters.

            These three values are cached in ``_attr_weights``,
            ``_attr_funcs``, and ``_rel_func`` attributes respectively after
            parsing.
        """
        attr_weights = self._parse_weights()
        attr_funcs = self._parse_attr_strats(attr_weights)
        rel_func = self._parse_rel_strat()
        return attr_weights, attr_funcs, rel_func

    def _parse_weights(self) -> Dict[str, float]:
        """ Parse the ``weights`` attribute.

        Returns:
            Mapping attribute names to their ratio in computing attribute
            similarities. If ``weights`` is ``None``, then assign equal weights
            to each of the attributes and return it. Otherwise, return the
            ``weights`` attribute.
        """
        attr_names = self._graph.attr_vals.keys()
        if self.weights is None:
            attr_weights = {name: 1 / len(attr_names) for name in attr_names}
        else:
            attr_weights = self.weights
        return attr_weights

    def _parse_attr_strats(
        self, attr_weights: Mapping[str, float]
    ) -> Dict[str, Callable]:
        """ Parse the ``attr_strategy`` attribute.

        Args:
            attr_weights: Mapping attribute names to their ratio in computing
                attribute similarities. It is the result returned by the
                ``_parse_weights`` method.

        Raises:
            ValueError: If an attribute with type not being ``'text'`` or
                ``'person_entity'`` does not have its name registered with an
                attribute similarity strategy in the ``attr_strategy`` attribute.

        Returns:
            Mapping attribute names to functions that compute the similarities
            of two values in the attributes. If the strategy string of an
            attribute is specified in ``attr_strategy``, then create the
            corresponding similarity function and add the pair in the resulting
            dictionary. Otherwise, use strategies specified in the
            ``_default_strategies`` depending on the attribute types.
        """
        attr_sim_funcs = dict()
        attr_types = self._graph.attr_types
        for name, attr_type in attr_types.items():
            weight = attr_weights[name]
            attr_type = attr_types[name]
            try:
                attr_strategy = self.attr_strategy[name]
            except KeyError:
                if attr_type not in self._default_strategies:
                    raise ValueError(
                        'Attribute similarity strategy must be specified in '
                        f'attr_strategy for attribute with unkown attr_type: '
                        f'{attr_type}'
                    )
                attr_strategy = self._default_strategies[attr_type]
            attr_sim_producer = self._sim_func_producers[attr_strategy]
            attr_sim_funcs[name] = attr_sim_producer(
                weight, self._graph.attr_vals[name], **self.kwargs
            )
        return attr_sim_funcs

    def _parse_rel_strat(self) -> Callable[
        [Collection, Collection, Callable[[Hashable], float]],
        float
    ]:
        """ Parse the ``rel_strategy`` attribute.

        Returns:
            A similarity function used to compute the relational similarity
            between two clusters. Its inputs are two collections of
            ids of neighboring clusters (of the two clusters one intend to
            compute similarity on) and a function which outputs the uniqueness
            of given a cluster id. It returns the similarity score as a
            `float`. Also set some flags about uniqueness calculation.
        """
        if self.rel_strategy is None:
            rel_strategy = self._default_strategies['relation']
        else:
            rel_strategy = self.rel_strategy
        rel_sim_producer = self._sim_func_producers[rel_strategy]
        if rel_strategy.endswith('fr'):
            self._use_nbr_cache = True
        else:
            self._use_nbr_cache = False
        if rel_strategy.startswith('adar_neighbor'):
            self._use_amb_type = 'neighbor'
            self._use_ambiguities = False
        elif rel_strategy.startswith('adar_attr'):
            self._use_amb_type = 'attr'
            self._use_ambiguities = True
        else:
            self._use_amb_type = None
            self._use_ambiguities = False
        return rel_sim_producer(**self.kwargs)

    @timeit
    def _calc_node_attr_sim(self, node1: Node, node2: Node) -> float:
        """ Compute the attribute similarity between two references.

        Args:
            node1: One of the two references for computation.
            node2: One of the two references for computation.

        Note:
            If the type of the attribute is ``'person_entity'``, its raw value
            is put into the attribute similarity function. Otherwise, the
            processed value is used.

        Returns:
            The attribute similarity score between two references.
        """
        attr_score = 0
        attr_types = self._graph.attr_types
        for name in node1.attr_vals:
            attr_sim_func = self._attr_funcs[name]
            if attr_types[name] == 'person_entity':
                value1 = node1.raw_attr_vals[name]
                value2 = node2.raw_attr_vals[name]
            else:
                value1 = node1.attr_vals[name]
                value2 = node2.attr_vals[name]
            attr_score += attr_sim_func(value1, value2)
        return attr_score

    @timeit
    def _calc_similarity(
        self, cluster1: Hashable, cluster2: Hashable
    ) -> float:
        """ Compute similarity between two clusters.

        Args:
            cluster1: One of the clusters for computation.
            cluster2: One of the clusters for computation.

        Returns:
            The similarity score between two clusters (a weighted average
            between attribute and relational similarities).
        """
        attr_score = self._calc_attr_sim(cluster1, cluster2)
        rel_score = self._calc_rel_sim(cluster1, cluster2)
        return (1-self.alpha)*attr_score + self.alpha*rel_score

    @timeit
    def _calc_attr_sim(self, cluster1: Hashable, cluster2: Hashable) -> float:
        """ Compute the attribute similarity between two clusters.

        Args:
            cluster1: One of the clusters for computation.
            cluster2: One of the clusters for computation.

        Returns:
            The attribute similarity score between two clusters.
        """
        nodes1 = self._clusters[cluster1]
        nodes2 = self._clusters[cluster2]
        if self.linkage == 'average':
            total_score = 0
            for node1, node2 in itertools.product(nodes1, nodes2):
                total_score += self._calc_node_attr_sim(node1, node2)
            return total_score / (len(nodes1)*len(nodes2))
        elif self.linkage == 'min':
            return min(
                self._calc_node_attr_sim(node1, node2)
                for node1, node2 in itertools.product(nodes1, nodes2)
            )
        elif self.linkage == 'max':
            return max(
                self._calc_node_attr_sim(node1, node2)
                for node1, node2 in itertools.product(nodes1, nodes2)
            )

    @timeit
    def _calc_rel_sim(self, cluster1: Hashable, cluster2: Hashable) -> float:
        """ Compute the relational similarity between two clusters.

        Args:
            cluster1: One of the clusters for computation.
            cluster2: One of the clusters for computation.

        Returns:
            The attribute similarity score between two clusters.
        """
        get_uniqueness = self._calc_cluster_uniq()
        nbrs1 = self._get_cluster_neighbors(cluster1)
        nbrs2 = self._get_cluster_neighbors(cluster2)
        rel_score = self._rel_func(nbrs1, nbrs2, get_uniqueness)
        return rel_score

    def _init_ambiguities(self) -> None:
        """ Initialize amgibuity values of each reference.

        Raies:
            ValueError: If either ``first_attr`` or ``second_attr`` attribute
                is None.
        """
        if self._use_ambiguities:
            if self.first_attr is None or self.second_attr is None:
                raise ValueError(
                    'Using ambiguities requires both first_attr and '
                    'second_attr to be valid functions instead of None'
                )
            self._ambiguities = self._graph.get_ambiguity_adar(
                self.first_attr, self.first_attr_raw,
                self.second_attr, self.second_attr_raw
            )

    @timeit
    def _calc_cluster_uniq(self) -> Optional[Callable[[Hashable], float]]:
        """ Return a function which can compute a cluster uniqueness.

        Returns:
            This returned function returns the uniqueness score given a cluster
            id. If such function is not required during computation of
            similarities, ``None`` is returned. Only a function is returned
            because most of cluster uniqueness values needed to be computed, so
            computing on demand vastly improves program efficiency.
        """
        neighbor_amb, attr_amb = dict(), dict()
        if self._use_amb_type == 'neighbor':
            def get_uniqueness(cluster):
                neighbors = self._get_cluster_neighbors(cluster)
                if type(neighbors) is not set:
                    num_neighbors = len(set(neighbors))
                else:
                    num_neighbors = len(neighbors)
                return 1 / (1+np.log(num_neighbors))
        elif self._use_amb_type == 'attr':
            def get_uniqueness(cluster):
                attr_amb = 0
                for node in self._clusters[cluster]:
                    attr_amb += self._ambiguities[node]
                attr_amb /= len(self._clusters[cluster])
                return 1 / attr_amb
        else:
            get_uniqueness = None
        return get_uniqueness

    @timeit
    def _get_cluster_neighbors(self, cluster: Hashable) -> List[Hashable]:
        """ Return a list of neighboring clusters of the input cluster.

        Args:
            cluster: The neighboring clusters of this cluster is to be
                returned.

        Returns:
            A list of neighboring clusters of the input cluster. It includes
            neighbors with multiplicities because some relational similarity
            functions are based on multisets.
        """
        cluster_neighbors = getattr(self, '_cluster_neighbors', None)
        if not self._use_nbr_cache or cluster_neighbors is None:
            nodes = self._clusters[cluster]
            neighbors_list = (
                self._graph.get_neighbors(node) for node in nodes
            )
            neighbor_nodes = itertools.chain(*neighbors_list)
            result = [self._inv_clusters[node] for node in neighbor_nodes]
        else:
            result = cluster_neighbors[cluster]
        return result

    def _clear_cache(self) -> None:
        """ Clear cache attributes not needed after ``resolve`` finishes."""
        delattr(self, '_graph')
        delattr(self, '_attr_weights')
        delattr(self, '_attr_funcs')
        delattr(self, '_rel_func')
        if self._use_ambiguities:
            delattr(self, '_ambiguities')
        delattr(self, '_use_nbr_cache')
        delattr(self, '_use_amb_type')
        delattr(self, '_use_ambiguities')
        delattr(self, '_clusters')
        delattr(self, '_inv_clusters')
        delattr(self, '_cluster_neighbors')
        delattr(self, '_sim_clusters_pool')
        self.logger.debug('Cache attributes deleted.')

    def log_time(self) -> None:
        """ Log the ``time_dict`` attribute.

        It also computes average execution time for each method by dividing
        total time spent by total number of times called, and reset the
        ``time_dict`` attribute after logging.
        """
        for f_name, [total, count] in self.time_dict.items():
            average = total / count
            self.logger.debug(f'Total time taken by {f_name}: {total}s')
            self.logger.debug(f'Total number of calls to {f_name}: {count}')
            self.logger.debug(f'Average time taken by {f_name}: {average}s')
        self.time_dict = collections.defaultdict(lambda: [0, 0])
