import itertools
import collections
import numpy as np
from py_stringmatching.similarity_measure import soft_tfidf, jaro_winkler


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

    def resolve(self, graph):
        self._init_cache(graph)

    def _init_cache(self, graph):
        self._graph = graph
        self._clusters = {node.id: set([node]) for node in graph.nodes}
        self._inv_clusters = {node: node.id for node in graph.nodes}
        parsed_result = self._parse_strategy()
        self._attr_weights, self._attr_funcs, self._rel_func = parsed_result
        self._init_attr_sims()
        self._ambiguities = graph.get_ambiguity_adar()

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

    def _init_attr_sims(self):
        def missing_factory():
            return collections.defaultdict(float)

        attr_sim_matrix = collections.defaultdict(missing_factory)
        for node1, node2 in itertools.combinations(self._graph.nodes, 2):
            similarity = self._calc_node_attr_sim(node1, node2)
            attr_sim_matrix[node1][node2] = similarity
            attr_sim_matrix[node2][node1] = similarity
        self._attr_sim_matrix = attr_sim_matrix

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
                total_score += self._attr_sim_matrix[node1][node2]
        return total_score / (len(nodes1)*len(nodes2))

    def _calc_rel_sim(self, cluster1, cluster2):
        ambiguities = self._calc_cluster_amb()
        neighbors1 = self._get_cluster_neighbors(cluster1)
        neighbors2 = self._get_cluster_neighbors(cluster2)
        rel_score = self._rel_func(neighbors1, neighbors2, ambiguities)
        return rel_score

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
                # Tentatively use minimum ambiguity among all attributes
                cluster_attr_amb += min(
                    ambiguities[name][node.get_attr(name, get_raw=True)]
                    for name in ambiguities
                )
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
