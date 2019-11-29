import itertools
import collections
import random
import multiprocessing
import numpy as np
from .utils import (
    SimFuncFactory, ClusteringMetrics, DSU, PriorityQueue, SimilarityEntry,
    WithLogger, logtime, timeit
)


class Resolver(WithLogger):

    def __init__(
        self, blocking_strategy, raw_blocking=False, alpha=0, weights=None,
        attr_strategy=dict(), rel_strategy=None,
        blocking_threshold=3, bootstrap_strategy=None, raw_bootstrap=False,
        edge_match_threshold=1, first_attr=None, first_attr_raw=False,
        second_attr=None, second_attr_raw=False, linkage='max',
        similarity_threshold=0.935, seed=None, plot_prc=False, verbose=0,
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
        self._kwargs = kwargs
        self._sim_func_producers = {
            'stfidf_jaro_winkler': SimFuncFactory.produce_stfidf_jaro_winkler,
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
            'text': 'stfidf_jaro_winkler',
            'person_entity': 'jaro_winkler',
            'relation': 'jaccard_coef'
        }
        self._time_dict = collections.defaultdict(lambda: [0, 0])
        random.seed(seed)
        if plot_prc:
            self._prc_list = list()
        super().__init__(verbose)

    @property
    def time_dict(self):
        return self._time_dict

    @logtime('Time taken for the core resolution algorithm')
    def resolve(self, graph, ground_truth=None):
        self._init_cache(graph)
        buckets = self._blocking(graph)
        self._relational_boostrapping(buckets)
        self._cluster_nodes(list(ground_truth.values()))
        resolved_mapping = collections.OrderedDict()
        for node, cluster in self._inv_clusters.items():
            resolved_mapping[node.id] = cluster
        if self.plot_prc:
            return resolved_mapping, self._prc_list
        return resolved_mapping

    @logtime('Time taken for blocking')
    def _blocking(self, graph):
        '''
        Initialize possible reference pairs using blocking techniques
        :param graph: reference graph
        :return:list of buckets contains similar references
        '''
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
        self._logger.info(f'Number of buckets in blocking: {len(buckets)}')
        return buckets

    @logtime('Time taken for relational bootstrapping')
    def _relational_boostrapping(self, buckets):
        self._init_clusters(buckets, self.edge_match_threshold)
        self._init_sim_clusters_pool(buckets)

    @timeit
    def _init_clusters(self, buckets, edge_match_threshold):
        nodes = self._graph.nodes
        clusters = collections.defaultdict(set)
        inv_clusters = collections.OrderedDict()
        dsu = DSU(nodes)
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                # todo check if redundant
                exact_match = self._check_exact_match(node1, node2)
                edge_check_passed = self._check_edge_match(node1, node2)
                if exact_match and edge_check_passed:
                    dsu.union(node1, node2)
        for node in sorted(dsu.items, key=lambda node: node.id):
            parent = dsu.find(node)
            clusters[parent.id].add(node)
            inv_clusters[node] = parent.id
        self._logger.info(f'Number of initial clusters: {len(clusters)}')
        self._clusters = clusters
        self._inv_clusters = inv_clusters
        cluster_neighbors = {
            cluster: set(self._get_cluster_neighbors(cluster))
            for cluster in clusters
        }
        self._cluster_neighbors = cluster_neighbors

    @timeit
    def _init_sim_clusters_pool(self, buckets):
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
        self._logger.info(f'Number of initial similar pairs: {num_pairs}')
        self._sim_clusters_pool = sim_clusters_pool

    @timeit
    def _check_exact_match(self, node1, node2):
        if self.bootstrap_strategy is not None:
            return self.bootstrap_strategy(node1.attr_vals, node2.attr_vals)
        for name in set(node1.attr_vals) & set(node2.attr_vals):
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
    def _check_edge_match(self, node1, node2):
        nbr_nodes1 = self._graph.get_neighbors(node1)
        nbr_nodes2 = self._graph.get_neighbors(node2)
        count = 0
        added_pairs = set([(node1, node2), (node2, node1)])
        for nbr_node1, nbr_node2 in itertools.product(nbr_nodes1, nbr_nodes2):
            # todo not calulate this every time
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
    def _cluster_nodes(self, ground_truth):
        sim_clusters = collections.defaultdict(set)
        cluster_entries = collections.defaultdict(dict)
        self._logger.debug('Start building priority queue.')
        pqueue = self._init_queue_entries(cluster_entries, sim_clusters)
        counter = 0
        self._logger.debug('Finish building priority queue. Start popping.')
        if self.plot_prc:
            # Add a point to be plotted every <step> time to reduce computation
            # 1600 is completely empirical and can be arbitrarily modified
            step = len(pqueue) // 1600
        while pqueue:
            entry = pqueue.pop()
            counter += 1
            if (counter % 10000 == 0):
                self._logger.debug(f'Number of pops from queue: {counter}')
            if entry.similarity < self.similarity_threshold:
                break
            cluster1, cluster2 = entry.clusters
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
        self._logger.debug(f'Total number of pops from queue: {counter}')
        if self.plot_prc:
            num_pr = len(self._prc_list)
            self._logger.debug(f'Number of precision-recall pairs: {num_pr}')

    @timeit
    def _init_queue_entries(self, cluster_entries, sim_clusters):
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
    def _merge_clusters(self, cluster1, cluster2, sim_clusters):
        self._clusters[cluster1] |= self._clusters[cluster2]
        self._clusters.pop(cluster2)
        cluster_neighbors = self._cluster_neighbors
        for node in self._clusters[cluster1]:
            self._inv_clusters[node] = cluster1
        self._update_adj_set(cluster1, cluster2, sim_clusters, True)
        self._update_adj_set(cluster1, cluster2, cluster_neighbors, False)

    @timeit
    def _update_adj_set(self, cluster1, cluster2, adjacency_set, remove_self):
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
        self, cluster1, cluster2, pqueue,
        cluster_entries, sim_clusters
    ):
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
    def _add_pr(self, ground_truth):
        precision, recall, _ = ClusteringMetrics.precision_recall(
            ground_truth,
            list(self._inv_clusters.values()),
            log=False
        )
        self._prc_list.append((precision, recall))

    @timeit
    def _add_sim_entries(
        self, cluster, pqueue, cluster_entries,
        sim_clusters, added_pairs
    ):
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
        self, cluster, pqueue, cluster_entries,
        sim_clusters, added_pairs
    ):
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

    def _init_cache(self, graph):
        self._graph = graph
        parsed_result = self._parse_strategy()
        self._attr_weights, self._attr_funcs, self._rel_func = parsed_result
        self._init_ambiguities()

    def _parse_strategy(self):
        attr_weights = self._parse_weights()
        attr_funcs = self._parse_attr_strats(attr_weights)
        rel_func = self._parse_rel_strat()
        return attr_weights, attr_funcs, rel_func

    def _parse_weights(self):
        attr_names = self._graph.attr_vals.keys()
        if self.weights is None:
            attr_weights = {name: 1 / len(attr_names) for name in attr_names}
        else:
            attr_weights = self.weights
        return attr_weights

    def _parse_attr_strats(self, attr_weights):
        attr_sim_funcs = dict()
        attr_types = self._graph.attr_types
        for name, attr_type in attr_types.items():
            weight = attr_weights[name]
            attr_type = attr_types[name]
            try:
                attr_strategy = self.attr_strategy[name]
            except KeyError:
                attr_strategy = self._default_strategies[attr_type]
            attr_sim_producer = self._sim_func_producers[attr_strategy]
            if attr_type == 'text' or attr_type == 'person_entity':
                attr_sim_funcs[name] = attr_sim_producer(
                    weight, self._graph.attr_vals[name], **self._kwargs
                )
        return attr_sim_funcs

    def _parse_rel_strat(self):
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
        return rel_sim_producer(**self._kwargs)

    @timeit
    def _calc_node_attr_sim(self, node1, node2):
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
    def _calc_similarity(self, cluster1, cluster2):
        attr_score = self._calc_attr_sim(cluster1, cluster2)
        rel_score = self._calc_rel_sim(cluster1, cluster2)
        return (1-self.alpha)*attr_score + self.alpha*rel_score

    @timeit
    def _calc_attr_sim(self, cluster1, cluster2):
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
    def _calc_rel_sim(self, cluster1, cluster2):
        get_uniqueness = self._calc_cluster_uniq()
        nbrs1 = self._get_cluster_neighbors(cluster1)
        nbrs2 = self._get_cluster_neighbors(cluster2)
        rel_score = self._rel_func(nbrs1, nbrs2, get_uniqueness)
        return rel_score

    def _init_ambiguities(self):
        if self._use_ambiguities:
            if self.first_attr is None or self.second_attr is None:
                raise ValueError(
                    'Using ambiguities requires both first_attr and'
                    'second_attr to be valid functions instead of None'
                )
            self._ambiguities = self._graph.get_ambiguity_adar(
                self.first_attr, self.first_attr_raw,
                self.second_attr, self.second_attr_raw
            )

    @timeit
    def _calc_cluster_uniq(self):
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
    def _get_cluster_neighbors(self, cluster):
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

    def log_time(self):
        for f_name, [total, count] in self._time_dict.items():
            average = total / count
            self._logger.debug(f'Total time taken by {f_name}: {total}s')
            self._logger.debug(f'Total number of calls to {f_name}: {count}')
            self._logger.debug(f'Average time taken by {f_name}: {average}s')
