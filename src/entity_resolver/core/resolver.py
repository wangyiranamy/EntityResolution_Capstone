import itertools
import collections
import random
import heapq
import numpy as np
from py_stringmatching.similarity_measure import jaro_winkler, soft_tfidf


class Resolver:

    def __init__(
        self, alpha=0.5, weights=None,
        attr_strategy=dict(), rel_strategy=None
    ):
        self.alpha = alpha
        self.weights = weights
        self.attr_strategy = attr_strategy
        self.rel_strategy = rel_strategy
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

    def _pq_add(self, q, c1, c2):
        cluster_sim = self._calc_similarity(c1, c2)
        # adjust priority queue order by using negative sim score
        tmp = [-cluster_sim, (c1, c2)]
        heapq.heappush(q, tmp)
        # update entries tracking
        self.tracking_pq_index[c1].append(tmp)
        self.tracking_pq_index[c2].append(tmp)

    def resolve(
        self, graph, threshold1=0.85,
        threshold2=0.7, similarity_thresh=0.8
    ):
        self._init_cache(graph)
        print('blocking')
        buckets = self.blocking(graph, threshold1, threshold2)
        print(len(buckets))
        print('boostrapping')
        self.relational_boostrapping(buckets)
        print('done')
        q = []
        new_cluster_number = len(graph.nodes)
        # keep track of pq entries involve cluster
        self.tracking_pq_index = {
            cluster: [] for cluster in self._clusters.keys()
        }
        # keep track of merged clusters
        self._merged_clusters = set()
        for c1, c2 in self._sim_clusters_pool:
            self._pq_add(q, c1, c2)
        i = 1
        while q:
            print(i)
            print(q[0])
            sim_score, tmp_c = heapq.heappop(q)
            if tmp_c != (-1, -1):
                c1, c2 = tmp_c
            else:
                i += 1
                continue
            if -sim_score < similarity_thresh:
                break
            # merge clusters into
            new_cluster = self._clusters.pop(c1)+self._clusters.pop(c2)
            self._clusters[new_cluster_number] = new_cluster
            self._merged_clusters.add(c1)
            self._merged_clusters.add(c2)
            # update inv cluster
            for node in self._clusters[new_cluster_number]:
                self._inv_clusters[node] = new_cluster_number

            # remove other entries from pq
            for entries in self.tracking_pq_index[c1]:
                entries[-1] = (-1, -1)
            for entries in self.tracking_pq_index[c2]:
                entries[-1] = (-1, -1)
            # del self.tracking_pq_index[c1]
            # del self.tracking_pq_index[c2]
            # update sim_cluster
            new_cluster = self._sim_clusters.pop(c1).union(
                self._sim_clusters.pop(c2)
            )
            self._sim_clusters[new_cluster_number] = new_cluster
            # remove old clusters in sim clusters
            self._sim_clusters[new_cluster_number] -= self._merged_clusters
            for c in self._sim_clusters[new_cluster_number]:
                self._pq_add(q, c, new_cluster_number)

            # update neighbor_cluster
            # new_cluster = self._neighbor_clusters[c1].union(
            #     self._neighbor_clusters[c2]
            # )
            # self._neighbor_clusters[new_cluster_number] = new_cluster
            # del self._neighbor_clusters[c1]
            # del self._neighbor_clusters[c2]
            neighbors = self._get_cluster_neighbors(new_cluster_number)
            print('length of neighbors', len(set(neighbors)))
            for c in set(neighbors):
                if c != new_cluster_number:
                    entries = self.tracking_pq_index[c]
                    for entry in entries:
                        if entry[-1] != (-1, -1):
                            c1, c2 = entry[-1]
                            entry[0] = self._calc_similarity(c1, c2)
                            # entry[-1] = (-1,-1)
                            # self._pq_add(q,c1,c2)
            heapq.heapify(q)
            # increment new cluster number
            new_cluster_number += 1
            i += 1
        return self._clusters, self._inv_clusters

    def blocking(self, graph, threshold1, threshold2):
        '''
        Initialize possible reference pairs using Blocking techniques
        :param graph: reference graph
        :return:list of buckets contains similar references
        '''
        buckets = list()
        candidates = set(graph.nodes)  # list of nodes
        similarity_matrix = self._attr_sim_matrix
        '''
        random select noda A and find nodes that > threshold2 put in same
        bucket and remove nodes >threshold1 from candidate
        then random select nodes B until buckets cover all the data
        '''
        while candidates:
            sample_node = random.sample(candidates, 1)[0]
            sim = similarity_matrix[sample_node]
            bucket = [i for i, x in sim.items() if x >= threshold2]
            buckets.append(bucket)
            nodes_to_remove = set(
                [i for i, x in sim.items() if x >= threshold1] +
                [sample_node]
            )
            candidates = candidates-nodes_to_remove
        return buckets

    def relational_boostrapping(self, buckets, k=1):
        # initialize clusters here
        candidates_pair = []
        nodes = self._graph.nodes
        clusters = dict()
        inv_clusters = {node: 0 for node in nodes}
        data = [i for i in range(len(nodes))]
        for bucket in buckets:
            for node1, node2 in itertools.combinations(bucket, 2):
                # todo check if rendandent
                match = self.check_exact_match(node1, node2)
                if match:
                    if (
                        not self.check_ambig(node1) and
                        not self.check_ambig(node2)
                    ):
                        candidates_pair.append((
                            nodes.index(node1),
                            nodes.index(node2)
                        ))
                    else:
                        edge1 = node1.edge
                        edge2 = node2.edge
                        if self.check_edge_match(edge1, edge2, k):
                            candidates_pair.append((
                                nodes.index(node1),
                                nodes.index(node2)
                            ))
        for node1, node2 in candidates_pair:
            # build cluster union find
            self.union(data, node1, node2)
        for i, cluster in enumerate(data):
            if cluster in clusters:
                clusters[cluster].append(nodes[i])
                inv_clusters[nodes[i]] = cluster
            else:
                clusters[cluster] = [nodes[i]]
        self._clusters = clusters
        self._inv_clusters = inv_clusters
        # store similar clusters
        sim_c = {cluster: set() for cluster in clusters.keys()}
        sim_c_pool = set()
        for bucket in buckets:
            cluster_set = set()
            for node in bucket:
                cluster_set.add(inv_clusters[node])
            for x in itertools.combinations(cluster_set, 2):
                sim_c_pool.add(x)
            for c in cluster_set:
                sim_c[c].union(cluster_set-{c})
        self._sim_clusters = sim_c
        self._sim_clusters_pool = sim_c_pool

    def find(self, data, i):
        if i != data[i]:
            data[i] = self.find(data, data[i])
        return data[i]

    def union(self, data, i, j):
        pi, pj = self.find(data, i), self.find(data, j)
        if pi != pj:
            data[pi] = pj

    def check_exact_match(self, node1, node2):
        for name in set(node1.get_attr_names()) & set(node2.get_attr_names()):
            value1 = node1.get_attr(name)
            value2 = node2.get_attr(name)
            if value1 != value2:
                return False
        return True

    def check_ambig(self, node, threshold=0.8):
        ambig = self._ambiguities
        if ambig[node] < threshold:
            return False
        else:
            return True

    def check_edge_match(self, edge1, edge2, k):
        nodes1 = edge1.nodes
        nodes2 = edge2.nodes
        combinations = list(itertools.product(nodes1, nodes2))
        count = 0
        for combo in combinations:
            # todo not calulate this every time
            if self.check_exact_match(combo[0], combo[1]):
                count += 1
            if count >= k:
                return True
        return False

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

    def _calc_similarity(self, cluster1, cluster2):
        attr_score = self._calc_attr_sim(cluster1, cluster2)
        rel_score = self._calc_rel_sim(cluster1, cluster2)
        return (1-self.alpha)*attr_score + self.alpha*rel_score

    def _calc_attr_sim(self, cluster1, cluster2):
        nodes1 = self._clusters[cluster1]
        nodes2 = self._clusters[cluster2]
        total_score = 0
        for node1 in nodes1:
            for node2 in nodes2:
                total_score += self._calc_node_attr_sim(node1, node2)
        return total_score / (len(nodes1)*len(nodes2))

    def _calc_rel_sim(self, cluster1, cluster2):
        ambiguities = self._calc_cluster_amb()
        neighbors1 = self._get_cluster_neighbors(cluster1)
        neighbors2 = self._get_cluster_neighbors(cluster2)
        rel_score = self._rel_func(neighbors1, neighbors2, ambiguities)
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
