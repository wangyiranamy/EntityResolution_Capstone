import itertools
import collections
from py_stringmatching.similarity_measure import soft_tfidf, jaro_winkler


class SimFuncFactory:

    @staticmethod
    def produce_jaro_winkler(weight, corpus_list):
        sim_func = jaro_winkler.JaroWinkler()
        soft_tfidf = soft_tfidf.SoftTfIdf(corpus_list, sim_func)

        def jaro_winkler(value1, value2):
            return weight * soft_tfidf.get_raw_score(value1, value2)
        return jaro_winkler

    @staticmethod
    def produce_common_neighbors(denominator):
        def common_neighbors(neighbors1, neighbors2):
            return len(set(neighbors1) & set(neighbors2)) / denominator
        return common_neighbors

    @staticmethod
    def produce_common_neighbors_fr(denominator):
        def common_neighbors_fr(neighbors1, neighbors2):
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            common_neighbors = 0
            for node, count in counter1.items():
                common_neighbors += min(count, counter2.get(node, 0))
            return common_neighbors / denominator
        return common_neighbors_fr

    @staticmethod
    def produce_jaccard_coef():
        def jaccard_coef(neighbors1, neighbors2):
            neighbors1_set = set(neighbors1)
            neighbors2_set = set(neighbors2)
            intersection = len(neighbors1_set & neighbors2_set)
            union = len(neighbors1_set | neighbors2_set)
            return intersection / union
        return jaccard_coef

    @staticmethod
    def produce_jaccard_coef_fr():
        def jaccard_coef_fr(neighbors1, neighbors2):
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            common_neighbors, total_neighbors = 0, 0
            for node in set(counter1.keys()) | set(counter2.keys()):
                count1 = counter1.get(node, 0)
                count2 = counter2.get(node, 0)
                common_neighbors += min(count1, count2)
                total_neighbors += max(count1, count2)
            return common_neighbors / total_neighbors
        return jaccard_coef_fr


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
            'common_neighbors': SimFuncFactory.produce_common_neighbors,
            'common_neighbors_fr': SimFuncFactory.produce_common_neighbors_fr,
            'jaccard_coef': SimFuncFactory.produce_jaccard_coef,
            'jaccard_coef_fr': SimFuncFactory.produce_jaccard_coef
        }
        self._default_strategies = {
            'text': 'jaro_winkler',
            'relation': 'common_neighbors'
        }
        pass

    def resolve(self, graph):
        self._init_cache(graph)
        pass

    def _init_cache(self, graph):
        self._graph = graph
        self._clusters = {node.id: set([node]) for node in graph.nodes}
        self._inv_clusters = {node: node.id for node in graph.nodes}
        parsed_result = self._parse_strategy(graph)
        self._attr_weights, self._attr_funcs, self._rel_func = parsed_result
        self._init_attr_sims()
        self._ambiguities = graph.get_ambiguity_adar()
        pass

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
        attr_types = self._graph.get_attr_types()
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
            rel_strategy = 'relation'
        else:
            rel_strategy = self.rel_strategy
        rel_sim_producer = self._sim_func_producers[rel_strategy]
        num_nodes = len(self._graph.nodes)
        return rel_sim_producer(num_nodes)

    def _init_attr_sims(self):
        attr_sim_matrix = dict()
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
        neighbors1 = self._get_cluster_neighbors(cluster1)
        neighbors2 = self._get_cluster_neighbors(cluster2)
        rel_score = self._rel_func(neighbors1, neighbors2)
        return rel_score

    def _get_cluster_neighbors(self, cluster):
        nodes = self._clusters[cluster]
        neighbors_list = (self._graph.get_neighbors(node) for node in nodes)
        neighbor_nodes = itertools.chain(*neighbors_list)
        return [self._inv_clusters[node] for node in neighbor_nodes]
