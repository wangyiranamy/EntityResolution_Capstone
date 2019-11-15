import itertools
import functools
import collections
import random
import multiprocessing
import numpy as np
from py_stringmatching.similarity_measure import jaro_winkler, soft_tfidf


class Resolver:

    def __init__(
        self, alpha=0.9, weights=None,
        attr_strategy=dict(), rel_strategy=None, hard_threshold=0.9,
        soft_threshold=0.8, edge_match_threshold=1,
        similarity_threshold=0.9, ambiguity_threshold=0.1
    ):
        self.alpha = alpha
        self.weights = weights
        self.attr_strategy = attr_strategy
        self.rel_strategy = rel_strategy
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.edge_match_threshold = edge_match_threshold
        self.similarity_threshold = similarity_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self._sim_func_producers = {
            'jaro_winkler': SimFuncFactory.produce_jaro_winkler,
            'jaccard_coef': SimFuncFactory.produce_jaccard_coef,
            'jaccard_coef_fr': SimFuncFactory.produce_jaccard_coef_fr,
            'adar_neighbor': SimFuncFactory.produce_adar_neighbor,
            'adar_neighbor_fr': SimFuncFactory.produce_adar_neighbor_fr,
            'adar_attr': SimFuncFactory.produce_adar_attr,
            'adar_attr_fr': SimFuncFactory.produce_adar_attr_fr
        }
        self._default_strategies = {
            'text': 'jaro_winkler',
            'relation': 'jaccard_coef'
        }

    def _missing_factory(self):
        return collections.defaultdict(float)

    def resolve(self, graph):
        self._init_cache(graph)
        buckets = self._blocking(
            graph, self.hard_threshold, self.soft_threshold
        )
        self._relational_boostrapping(buckets)
        self._cluster_nodes()
        return self._clusters, self._inv_clusters

    def _blocking(self, graph, hard_threshold, soft_threshold):
        '''
        Initialize possible reference pairs using Blocking techniques
        :param graph: reference graph
        :return:list of buckets contains similar references
        '''
        buckets = list()
        candidates = set(graph.nodes)
        similarity_matrix = self._attr_sim_matrix
        '''
        random select noda A and find nodes that > soft_threshold put in same
        bucket and remove nodes > hard_threshold from candidate
        then random select nodes B until buckets cover all the data
        '''
        while candidates:
            sample_node = random.sample(candidates, 1)[0]
            sim_dict = similarity_matrix[sample_node]
            bucket = [i for i, x in sim_dict.items() if x >= soft_threshold]
            buckets.append(bucket)
            for node, similarity in sim_dict.items():
                if similarity >= hard_threshold:
                    candidates.discard(node)
            candidates.remove(sample_node)
        return buckets

    def _relational_boostrapping(self, buckets):
        self._init_clusters(buckets, self.edge_match_threshold)
        self._init_sim_clusters_pool(buckets)

    def _init_clusters(self, buckets, edge_match_threshold):
        nodes = self._graph.nodes
        id_to_node = {node.id: node for node in nodes}
        clusters = collections.defaultdict(set)
        inv_clusters = dict()
        dsu = DSU(nodes)
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                # todo check if redundant
                match = self._check_exact_match(node1, node2)
                node_check_passed = (
                    self._check_ambig(node1)
                    and self._check_ambig(node2)
                )
                edge_check_passed = self._check_edge_match(
                    node1.edge, node2.edge, edge_match_threshold
                )
                if match and (node_check_passed or edge_check_passed):
                    dsu.union(node1, node2)
        for node in dsu.items:
            parent = dsu.find(node)
            clusters[parent.id].add(node)
            inv_clusters[node] = parent.id
        self._clusters = clusters
        self._inv_clusters = inv_clusters

    def _init_sim_clusters_pool(self, buckets):
        sim_clusters_pool = set()
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                cluster1 = self._inv_clusters[node1]
                cluster2 = self._inv_clusters[node2]
                if cluster1 != cluster2:
                    sim_clusters_pool.add((cluster1, cluster2))
        self._sim_clusters_pool = sim_clusters_pool

    def _check_exact_match(self, node1, node2):
        for name in set(node1.get_attr_names()) & set(node2.get_attr_names()):
            value1 = node1.get_attr(name)
            value2 = node2.get_attr(name)
            if value1 != value2:
                return False
        return True

    def _check_ambig(self, node):
        ambig = self._ambiguities
        if ambig[node] < self.ambiguity_threshold:
            return True
        else:
            return False

    def _check_edge_match(self, edge1, edge2, edge_match_threshold):
        nodes1 = edge1.nodes
        nodes2 = edge2.nodes
        count = 0
        for node1, node2 in itertools.product(nodes1, nodes2):
            # todo not calulate this every time
            if self._check_exact_match(node1, node2):
                count += 1
            if count >= edge_match_threshold:
                return True
        return False

    def _cluster_nodes(self):
        pqueue = PriorityQueue()
        sim_clusters = collections.defaultdict(set)
        cluster_entries = collections.defaultdict(dict)
        cluster_neighbors = {
            cluster: set(self._get_cluster_neighbors(cluster))
            for cluster in self._clusters
        }
        self._init_queue_entries(
            pqueue, cluster_entries,
            sim_clusters, cluster_neighbors
        )
        while pqueue:
            entry = pqueue.pop()
            if entry.similarity < self.similarity_threshold:
                break
            cluster1, cluster2 = entry.clusters
            if cluster1 in cluster_neighbors[cluster2]:
                continue
            self._merge_clusters(
                cluster1, cluster2,
                sim_clusters, cluster_neighbors
            )
            self._update_pqueue(
                cluster1, cluster2, pqueue, cluster_entries,
                sim_clusters, cluster_neighbors
            )

    def _init_queue_entries(
        self, pqueue, cluster_entries,
        sim_clusters, cluster_neighbors,
    ):
        sim_clusters_pool = self._sim_clusters_pool
        while sim_clusters_pool:
            cluster1, cluster2 = sim_clusters_pool.pop()
            similarity = self._calc_similarity(
                cluster1, cluster2, cluster_neighbors
            )
            entry = SimilarityEntry(cluster1, cluster2, similarity)
            sim_clusters[cluster1].add(cluster2)
            sim_clusters[cluster2].add(cluster1)
            cluster_entries[cluster1][cluster2] = entry
            cluster_entries[cluster2][cluster1] = entry
            pqueue.push(entry)

    def _merge_clusters(
        self, cluster1, cluster2,
        sim_clusters, cluster_neighbors
    ):
        # merge to cluster with smaller id
        if cluster2 < cluster1:
            cluster1, cluster2 = cluster2, cluster1
        self._clusters[cluster1] |= self._clusters[cluster2]
        self._clusters.pop(cluster2)
        for node in self._clusters[cluster1]:
            self._inv_clusters[node] = cluster1
        self._update_adj_set(cluster1, cluster2, sim_clusters, True)
        self._update_adj_set(cluster1, cluster2, cluster_neighbors, False)

    def _update_adj_set(self, cluster1, cluster2, adjacency_set, remove_self):
        for item_set in adjacency_set[cluster2].values():
            item_set.remove(cluster2)
            item_set.add(cluster1)
        adjacency_set[cluster1] |= adjacency_set[cluster2]
        if remove_self:
            adjacency_set[cluster1].remove(cluster1)
        adjacency_set.pop(cluster2)

    def _update_pqueue(
        self, cluster1, cluster2, pqueue,
        cluster_entries, sim_clusters, cluster_neighbors
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
            sim_clusters, cluster_neighbors, added_pairs
        )
        self._update_nbr_entries(
            cluster1, pqueue, cluster_entries,
            sim_clusters, cluster_neighbors, added_pairs
        )

    def _add_sim_entries(
        self, cluster, pqueue, cluster_entries,
        sim_clusters, cluster_neighbors, added_pairs
    ):
        for sim_cluster in sim_clusters[cluster]:
            similarity = self._calc_similarity(
                cluster, sim_cluster, cluster_neighbors
            )
            entry = SimilarityEntry(cluster, sim_cluster, similarity)
            pqueue.push(entry)
            cluster_entries[cluster1][sim_cluster] = entry
            cluster_entries[sim_cluster][cluster1] = entry
            added_pairs.update(set([
                (cluster, sim_cluster),
                (sim_cluster, cluster)
            ]))

    def _update_nbr_entries(
        self, cluster, pqueue, cluster_entries,
        sim_clusters, cluster_neighbors, added_pairs
    ):
        for cluster_nbr in cluster_neighbors[cluster]:
            for sim_cluster in sim_clusters[cluster_nbr]:
                if (cluster_nbr, sim_cluster) in added_pairs:
                    continue
                similarity = self._calc_similarity(
                    cluster_nbr, sim_cluster, cluster_neighbors
                )
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
        attr_vals = self._graph.get_attr_vals()
        attr_weights = self._parse_weights()
        attr_funcs = self._parse_attr_strats(attr_vals, attr_weights)
        rel_func = self._parse_rel_strat()
        return attr_weights, attr_funcs, rel_func

    def _parse_weights(self):
        attr_names = self._graph.get_attr_names()
        if self.weights is None:
            attr_weights = {name: 1 / len(attr_names) for name in attr_names}
        else:
            attr_weights = self.weights
        return attr_weights

    def _parse_attr_strats(self, attr_vals, attr_weights):
        attr_sim_funcs = dict()
        attr_types = self._graph.attr_types
        for name, attr_values in attr_vals.items():
            weight = attr_weights[name]
            attr_type = attr_types[name]
            try:
                attr_strategy = self.attr_strategy[name]
            except KeyError:
                attr_strategy = self._default_strategies[attr_type]
            attr_sim_producer = self._sim_func_producers[attr_strategy]
            if attr_type == 'text':
                attr_sim_funcs[name] = attr_sim_producer(weight, attr_values)
        return attr_sim_funcs

    def _parse_rel_strat(self):
        if self.rel_strategy is None:
            rel_strategy = self._default_strategies['relation']
        else:
            rel_strategy = self.rel_strategy
        rel_sim_producer = self._sim_func_producers[rel_strategy]
        return rel_sim_producer()

    def _calc_node_attr_sim(self, node1, node2):
        attr_score = 0
        for name in set(node1.get_attr_names()) & set(node2.get_attr_names()):
            attr_sim_func = self._attr_funcs[name]
            value1 = node1.get_attr(name)
            value2 = node2.get_attr(name)
            attr_score += attr_sim_func(value1, value2)
        return attr_score

    def _calc_similarity(self, cluster1, cluster2, cluster_neighbors=None):
        attr_score = self._calc_attr_sim(cluster1, cluster2)
        rel_score = self._calc_rel_sim(cluster1, cluster2, cluster_neighbors)
        return (1-self.alpha)*attr_score + self.alpha*rel_score

    def _calc_attr_sim(self, cluster1, cluster2):
        nodes1 = self._clusters[cluster1]
        nodes2 = self._clusters[cluster2]
        total_score = 0
        for node1 in nodes1:
            for node2 in nodes2:
                total_score += self._calc_node_attr_sim(node1, node2)
        return total_score / (len(nodes1)*len(nodes2))

    def _calc_rel_sim(self, cluster1, cluster2, cluster_neighbors):
        ambiguities = self._calc_cluster_amb()
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
                ambiguities_raw[name][node.get_attr(name, get_raw=True)]
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


class SimFuncFactory:

    @classmethod
    def produce_jaro_winkler(cls, weight, corpus_list):
        sim_func = jaro_winkler.JaroWinkler().get_raw_score
        soft_tfidf_obj = soft_tfidf.SoftTfIdf(corpus_list, sim_func)

        def jaro_winkler_sim(value1, value2):
            score1 = soft_tfidf_obj.get_raw_score(value1, value2)
            score2 = soft_tfidf_obj.get_raw_score(value2, value1)
            return weight * max(score1, score2)
        return jaro_winkler_sim

    @classmethod
    def produce_jaccard_coef(cls):
        def jaccard_coef(neighbors1, neighbors2, ambiguities):
            set1 = set(neighbors1)
            set2 = set(neighbors2)
            intersect = len(set1 & set2)
            union = len(set1 | set2)
            return intersect / union
        return jaccard_coef

    @classmethod
    def produce_jaccard_coef_fr(cls):
        def jaccard_coef_fr(neighbors1, neighbors2, ambiguities):
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            union, intersect = cls._union_intersect_counter(
                counter1, counter2,
                lambda curr, count, key: curr + count
            )
            return intersect / union
        return jaccard_coef_fr

    @classmethod
    def produce_adar_neighbor(cls):
        def adar_neighbor(neighbors1, neighbors2, ambiguities):
            ambiguities = ambiguities['neighbor']
            set1 = set(neighbors1)
            set2 = set(neighbors2)
            union = set1 | set2
            intersect = set1 & set2
            union_amb = sum(ambiguities[cluster] for cluster in union)
            intersect_amb = sum(ambiguities[cluster] for cluster in intersect)
            return intersect_amb / union_amb
        return adar_neighbor

    @classmethod
    def produce_adar_neighbor_fr(cls):
        def adar_neighbor_fr(neighbors1, neighbors2, ambiguities):
            ambiguities = ambiguities['neighbor']
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            union_amb, intersect_amb = cls._union_intersect_counter(
                counter1, counter2,
                lambda curr, count, key: curr + count*ambiguities[key]
            )
            return intersect_amb / union_amb
        return adar_neighbor_fr

    @classmethod
    def produce_adar_attr(cls):
        def adar_attr(neighbors1, neighbors2, ambiguities):
            ambiguities = ambiguities['attr']
            set1 = set(neighbors1)
            set2 = set(neighbors2)
            union = set1 | set2
            intersect = set1 & set2
            union_amb = sum(ambiguities[cluster] for cluster in union)
            intersect_amb = sum(ambiguities[cluster] for cluster in intersect)
            return intersect_amb / union_amb
        return adar_attr

    @classmethod
    def produce_adar_attr_fr(cls):
        def adar_attr_fr(neighbors1, neighbors2, ambiguities):
            ambiguities = ambiguities['attr']
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            union_amb, intersect_amb = cls._union_intersect_counter(
                counter1, counter2,
                lambda curr, count, key: curr + count*ambiguities[key]
            )
            return intersect_amb / union_amb
        return adar_attr_fr

    @staticmethod
    def _union_intersect_counter(counter1, counter2, accumulator, initial=0):
        union, intersect = initial, initial
        for key in set(counter1.keys()) | set(counter2.keys()):
            count1 = counter1.get(key, 0)
            count2 = counter2.get(key, 0)
            union_count = max(count1, count2)
            intersect_count = min(count1, count2)
            union = accumulator(union, union_count, key)
            intersect = accumulator(intersect, intersect_count, key)
        return union, intersect

class DSU:

    def __init__(self, items):
        self.items = {item: item for item in items}
        self.rank = {item: 1 for item in items}

    def union(self, item1, item2):
        parent1, parent2 = self.find(item1), self.find(item2)
        if self.rank[parent1] < self.rank[parent2]:
            self.items[parent1] = parent2
            self.rank[parent2] += self.rank[parent1]
        else:
            self.items[parent2] = parent1
            self.rank[parent1] += self.rank[parent2]

    def find(self, item):
        parent = self.items[item]
        if parent == item:
            return item
        res = self.find(parent)
        self.items[item] = res
        return res


class PriorityQueue:

    def __init__(self, items=[]):
        self._queue = list(items)
        self._indicies = {item: index for index, item in enumerate(items)}
        self._heapify()

    def __len__(self):
        return len(self._queue)

    def __contains__(self, item):
        return item in self._indicies

    def _heapify(self):
        for i in range(len(self._queue) - 1, 0, -1):
            self._shiftdown(i)

    def _shiftup(self, pos):
        while pos > 0:
            parent_pos = ((pos+1) >> 1)-1
            if self._queue[pos] < self._queue[parent_pos]:
                self._swap(pos, parent_pos)
                pos = parent_pos
            else:
                break

    def _shiftdown(self, pos):
        while pos < len(self._queue):
            left_pos = ((pos+1) << 1) - 1
            right_pos = ((pos+1) << 1)
            curr = self._queue[pos]
            left, right = None, None
            if left_pos < len(self._queue):
                left = self._queue[left_pos]
            if right_pos < len(self._queue):
                right = self._queue[right_pos]
            if left is None:
                break
            if right is None and left >= curr:
                break
            if (
                (right is None and left < curr)
                or (left < curr and left < right)
            ):
                self._swap(pos, left_pos)
                pos = left_pos
            elif right < curr and right < left:
                self._swap(pos, right_pos)
                pos = right_pos
            else:
                break

    def _swap(self, pos1, pos2):
        item1, item2 = self._queue[pos1], self._queue[pos2]
        self._queue[pos1] = item2
        self._queue[pos2] = item1
        self._indicies[item1] = pos2
        self._indicies[item2] = pos1

    def push(self, item):
        self._queue.append(item)
        index = len(self._queue) - 1
        self._indicies[item] = index
        self._shiftup(index)

    def pop(self):
        return self.remove(self._queue[0])

    def remove(self, item):
        item_index = self._indicies[item]
        self._swap(item_index, len(self._queue) - 1)
        removed_item = self._queue.pop()
        self._indicies.pop(removed_item)
        self._shiftdown(item_index)
        self._shiftup(item_index)
        return removed_item

    def discard(self, item):
        if item in self._indicies:
            self.remove(item)

    def update(self, item, new_item):
        item_index = self._indicies[item]
        self._queue[item_index] = new_item
        self._shiftdown(item_index)
        self._shiftup(item_index)


@functools.total_ordering
class SimilarityEntry:

    def __init__(self, cluster1, cluster2, similarity):
        self.similarity = similarity
        self.clusters = (cluster1, cluster2)

    def __hash__(self):
        return hash(self.clusters)

    def __eq__(self, other):
        if type(self) is type(other):
            return self.clusters == other.clusters
        return NotImplemented

    def __lt__(self, other):
        if type(self) is type(other):
            return self.similarity > other.similarity
        return NotImplemented

    def swap(self):
        cluster1, cluster2 = self.clusters
        self.clusters = (cluster2, cluster1)
