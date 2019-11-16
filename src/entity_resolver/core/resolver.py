import itertools
import collections
import random
import multiprocessing
import numpy as np
from .utils import SimFuncFactory, DSU, PriorityQueue, SimilarityEntry


class Resolver:

    def __init__(
        self, blocking_strategy, raw_blocking=False, alpha=0.5, weights=None,
        attr_strategy=dict(), rel_strategy=None,
        blocking_threshold=3, bootstrap_strategy=None, raw_bootstrap=False,
        edge_match_threshold=1, similarity_threshold=0.8, **kwargs
    ):
        self.alpha = alpha
        self.blocking_strategy = blocking_strategy
        self.raw_blocking = raw_blocking
        self.weights = weights
        self.attr_strategy = attr_strategy
        self.rel_strategy = rel_strategy
        self.blocking_threshold = blocking_threshold
        self.bootstrap_strategy = bootstrap_strategy
        self.raw_bootstrap = raw_bootstrap
        self.edge_match_threshold = edge_match_threshold
        self.similarity_threshold = similarity_threshold
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

    def __getattr__(self, name):
        try:
            return self._kwargs[name]
        except KeyError:
            raise AttributeError(f'No attribute named {name}')

    def resolve(self, graph):
        self._init_cache(graph)
        buckets = self._blocking(graph)
        self._relational_boostrapping(buckets)
        self._cluster_nodes()
        return self._clusters, self._inv_clusters

    def _blocking(self, graph):
        '''
        Initialize possible reference pairs using Blocking techniques
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
        print(f'number of buckets: {len(buckets)}')
        return buckets

    def _relational_boostrapping(self, buckets):
        self._init_clusters(buckets, self.edge_match_threshold)
        self._init_sim_clusters_pool(buckets)

    def _init_clusters(self, buckets, edge_match_threshold):
        nodes = self._graph.nodes
        clusters = collections.defaultdict(set)
        inv_clusters = dict()
        dsu = DSU(nodes)
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                # todo check if redundant
                exact_match = self._check_exact_match(node1, node2)
                edge_check_passed = self._check_edge_match(node1, node2)
                if exact_match and edge_check_passed:
                    dsu.union(node1, node2)
        for node in dsu.items:
            parent = dsu.find(node)
            clusters[parent.id].add(node)
            inv_clusters[node] = parent.id
        print(f'number of clusters: {len(clusters)}')
        self._clusters = clusters
        self._inv_clusters = inv_clusters
        cluster_neighbors = {
            cluster: set(self._get_cluster_neighbors(cluster))
            for cluster in clusters
        }
        self._cluster_neighbors = cluster_neighbors

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
        print(f'number of similar pairs: {len(sim_clusters_pool)}')
        self._sim_clusters_pool = sim_clusters_pool

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

    def _cluster_nodes(self):
        sim_clusters = collections.defaultdict(set)
        cluster_entries = collections.defaultdict(dict)
        pqueue = self._init_queue_entries(cluster_entries, sim_clusters)
        while pqueue:
            entry = pqueue.pop()
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

    def _merge_clusters(self, cluster1, cluster2, sim_clusters):
        self._clusters[cluster1] |= self._clusters[cluster2]
        self._clusters.pop(cluster2)
        cluster_neighbors = self._cluster_neighbors
        for node in self._clusters[cluster1]:
            self._inv_clusters[node] = cluster1
        self._update_adj_set(cluster1, cluster2, sim_clusters, True)
        self._update_adj_set(cluster1, cluster2, cluster_neighbors, False)

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
            if attr_type == 'text':
                attr_sim_funcs[name] = attr_sim_producer(
                    weight, self._graph.attr_vals[name], **self._kwargs
                )
            elif attr_type == 'person_entity':
                attr_sim_funcs[name] = attr_sim_producer(
                    weight, **self._kwargs
                )
        return attr_sim_funcs

    def _parse_rel_strat(self):
        if self.rel_strategy is None:
            rel_strategy = self._default_strategies['relation']
        else:
            rel_strategy = self.rel_strategy
        rel_sim_producer = self._sim_func_producers[rel_strategy]
        return rel_sim_producer(**self._kwargs)

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

    def _calc_similarity(self, cluster1, cluster2):
        attr_score = self._calc_attr_sim(cluster1, cluster2)
        rel_score = self._calc_rel_sim(cluster1, cluster2)
        return (1-self.alpha)*attr_score + self.alpha*rel_score
        return attr_score

    def _calc_attr_sim(self, cluster1, cluster2):
        nodes1 = self._clusters[cluster1]
        nodes2 = self._clusters[cluster2]
        # total_score = 0
        # for node1, node2 in itertools.product(nodes1, nodes2):
        #     total_score += self._calc_node_attr_sim(node1, node2)
        # return total_score / (len(nodes1)*len(nodes2))
        return max(
            self._calc_node_attr_sim(node1, node2)
            for node1, node2 in itertools.product(nodes1, nodes2)
        )

    def _calc_rel_sim(self, cluster1, cluster2):
        ambiguities = self._calc_cluster_amb()
        cluster_neighbors = getattr(self, '_cluster_neighbors', None)
        if cluster_neighbors is None:
            nbrs1 = self._get_cluster_neighbors(cluster1)
            nbrs2 = self._get_cluster_neighbors(cluster2)
        else:
            nbrs1 = cluster_neighbors[cluster1]
            nbrs2 = cluster_neighbors[cluster2]
        rel_score = self._rel_func(nbrs1, nbrs2, ambiguities)
        return rel_score

    def _init_ambiguities(self):
        ambiguities_raw = self._graph.get_ambiguity_adar()
        ambiguities = dict()
        for node in self._graph.nodes:
            # Tentatively use minimum ambiguity among all attributes
            ambiguities[node] = min(
                ambiguities_raw[name][node.raw_attr_vals[name]]
                for name in ambiguities_raw
            )
        self._ambiguities = ambiguities

    def _calc_cluster_amb(self):
        neighbor_amb, attr_amb = dict(), dict()
        ambiguities = self._ambiguities
        for cluster, nodes in self._clusters.items():
            num_neighbors = len(set(self._get_cluster_neighbors(cluster)))
            # Add 1 to the denominator to avoid zero divider,
            # but might not be appropriate
            neighbor_amb[cluster] = 1 / (1+np.log(num_neighbors))
            cluster_attr_amb = 0
            for node in nodes:
                cluster_attr_amb += self._ambiguities[node]
            cluster_attr_amb /= len(nodes)
            attr_amb[cluster] = 1 / cluster_attr_amb
        return {'neighbor': neighbor_amb, 'attr': attr_amb}

    def _get_cluster_neighbors(self, cluster):
        nodes = self._clusters[cluster]
        neighbors_list = (self._graph.get_neighbors(node) for node in nodes)
        neighbor_nodes = itertools.chain(*neighbors_list)
        return [self._inv_clusters[node] for node in neighbor_nodes]
