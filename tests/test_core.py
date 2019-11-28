import os
import json
import collections
import pytest
from entity_resolver.core import Evaluator, Graph, Resolver
from entity_resolver.parser import GraphParser


class TestEvaluator:

    @pytest.fixture
    def ground_truth(self):
        return {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

    @pytest.fixture
    def resolved_mapping(self):
        return {1: 1000, 2: 1000, 3: 1000, 4: 1001, 5: 1010}

    def test_ami(self, ground_truth, resolved_mapping):
        ami_evaluator = Evaluator(strategy='ami')
        ami_score = ami_evaluator.evaluate(ground_truth, resolved_mapping)
        assert round(ami_score, 2) == 0.23

    def test_v_measure(self, ground_truth, resolved_mapping):
        v_measure_evaluator = Evaluator(strategy='v-measure')
        v_measure_score = v_measure_evaluator.evaluate(
            ground_truth, resolved_mapping
        )
        assert round(v_measure_score, 2) == 0.67

    def test_precision_recall(self, ground_truth, resolved_mapping):
        precision_recall_evaluator = Evaluator(strategy='precision-recall')
        precision_recall_score = precision_recall_evaluator.evaluate(
            ground_truth, resolved_mapping
        )
        assert round(precision_recall_score[0], 2) == 0.33
        assert round(precision_recall_score[1], 2) == 0.5
        assert round(precision_recall_score[2], 2) == 0.4


class TestResolver:

    def _init_resolver(self, resolver, graph):
        resolver._init_cache(graph)
        nodes = {node.id: node for node in graph.nodes}
        resolver._clusters = {
            1: set([nodes[1], nodes[3]]),
            2: set([nodes[2], nodes[5], nodes[7]]),
            3: set([nodes[4]]),
            4: set([nodes[6], nodes[8]])
        }
        resolver._inv_clusters = {
            nodes[1]: 1,
            nodes[2]: 2,
            nodes[3]: 1,
            nodes[4]: 3,
            nodes[5]: 2,
            nodes[6]: 4,
            nodes[7]: 2,
            nodes[8]: 4
        }
        resolver._cluster_neighbors = {
            1: [1, 2, 1, 3],
            2: [2, 1, 2, 4, 2, 4],
            3: [4, 1],
            4: [4, 2, 4, 2]
        }

    @pytest.fixture
    def simple_graph_suite(self):
        parser = GraphParser({'text1': 'text', 'text2': 'text'})
        tmp_path = 'simple_graph.json'
        graph_dict = [
            {
                'node_id': 1,
                'edge_id': 1,
                'attr_dict': {'text1': 'a', 'text2': 'b'}
            },
            {
                'node_id': 2,
                'edge_id': 2,
                'attr_dict': {'text1': 'a', 'text2': 'a'}
            }
        ]
        with open(tmp_path, 'w') as f:
            json.dump(graph_dict, f)

        def cleanup():
            os.remove(tmp_path)
        return parser.parse(tmp_path), cleanup

    @pytest.fixture
    def complex_attr_graph_suite(self):
        parser = GraphParser({'text': 'text'})
        tmp_path = 'complex_attr_graph.json'
        graph_dict = [
            {'node_id': 1, 'edge_id': 1, 'attr_dict': {'text': 'a aa'}},
            {'node_id': 2, 'edge_id': 2, 'attr_dict': {'text': 'b ab'}},
            {'node_id': 3, 'edge_id': 3, 'attr_dict': {'text': 'a ba'}},
            {'node_id': 4, 'edge_id': 4, 'attr_dict': {'text': 'b bb'}}
        ]
        with open(tmp_path, 'w') as f:
            json.dump(graph_dict, f)

        def cleanup():
            os.remove(tmp_path)
        return parser.parse(tmp_path), cleanup

    @pytest.fixture
    def complex_rel_graph_suite(self):
        parser = GraphParser({'text': 'text'})
        tmp_path = 'complex_rel_graph.json'
        graph_dict = [
            {'node_id': 1, 'edge_id': 100, 'attr_dict': {'text': 'd'}},
            {'node_id': 2, 'edge_id': 100, 'attr_dict': {'text': 'b'}},
            {'node_id': 3, 'edge_id': 101, 'attr_dict': {'text': 'b'}},
            {'node_id': 4, 'edge_id': 101, 'attr_dict': {'text': 'c'}},
            {'node_id': 5, 'edge_id': 110, 'attr_dict': {'text': 'a'}},
            {'node_id': 6, 'edge_id': 110, 'attr_dict': {'text': 'd'}},
            {'node_id': 7, 'edge_id': 111, 'attr_dict': {'text': 'b'}},
            {'node_id': 8, 'edge_id': 111, 'attr_dict': {'text': 'a'}},
        ]
        with open(tmp_path, 'w') as f:
            json.dump(graph_dict, f)

        def cleanup():
            os.remove(tmp_path)
        return parser.parse(tmp_path), cleanup

    def test_cache(self, simple_graph_suite):
        graph, cleanup = simple_graph_suite
        resolver = Resolver(None, alpha=0.5)
        assert resolver.alpha == 0.5
        assert len(resolver.attr_strategy) == 0
        resolver._init_cache(graph)
        weights = resolver._attr_weights
        assert weights['text1'] == 0.5
        assert weights['text2'] == 0.5
        node1 = [node for node in graph.nodes if node.id == 1][0]
        node2 = [node for node in graph.nodes if node.id == 2][0]
        assert resolver._calc_node_attr_sim(node1, node2) == 0.5
        cleanup()

    def test_attr_sim(self, complex_attr_graph_suite):
        graph, cleanup = complex_attr_graph_suite
        resolver = Resolver(
            None, attr_strategy={'text': 'stfidf_jaro_winkler'}
        )
        resolver._init_cache(graph)
        node1 = [node for node in graph.nodes if node.id == 1][0]
        node2 = [node for node in graph.nodes if node.id == 2][0]
        node3 = [node for node in graph.nodes if node.id == 3][0]
        node4 = [node for node in graph.nodes if node.id == 4][0]
        assert round(resolver._calc_node_attr_sim(node1, node2), 2) == 0.90
        assert round(resolver._calc_node_attr_sim(node1, node3), 2) == 0.73
        assert round(resolver._calc_node_attr_sim(node1, node4), 2) == 0.00
        assert round(resolver._calc_node_attr_sim(node2, node3), 2) == 0.68
        assert round(resolver._calc_node_attr_sim(node2, node4), 2) == 0.73
        assert round(resolver._calc_node_attr_sim(node3, node4), 2) == 0.90
        cleanup()

    def test_rel_sim(self, complex_rel_graph_suite):
        graph, cleanup = complex_rel_graph_suite
        jaccard_coef_resolver = Resolver(None, rel_strategy='jaccard_coef')
        jaccard_coef_fr_resolver = Resolver(
            None, rel_strategy='jaccard_coef_fr'
        )
        adar_neighbor_resolver = Resolver(None, rel_strategy='adar_neighbor')
        adar_neighbor_fr_resolver = Resolver(
            None, rel_strategy='adar_neighbor_fr'
        )
        adar_attr_resolver = Resolver(
            None, rel_strategy='adar_attr',
            first_attr=lambda d: d['text'], first_attr_raw=True,
            second_attr=lambda d: d['text'], second_attr_raw=True,
        )
        adar_attr_fr_resolver = Resolver(
            None, rel_strategy='adar_attr_fr',
            first_attr=lambda d: d['text'], first_attr_raw=True,
            second_attr=lambda d: d['text'], second_attr_raw=True,
        )
        self._init_resolver(jaccard_coef_resolver, graph)
        self._init_resolver(jaccard_coef_fr_resolver, graph)
        self._init_resolver(adar_neighbor_resolver, graph)
        self._init_resolver(adar_neighbor_fr_resolver, graph)
        self._init_resolver(adar_attr_resolver, graph)
        self._init_resolver(adar_attr_fr_resolver, graph)
        assert jaccard_coef_resolver._calc_rel_sim(1, 2) == 0.5
        assert jaccard_coef_fr_resolver._calc_rel_sim(1, 2) == 0.25
        assert round(adar_neighbor_resolver._calc_rel_sim(1, 2), 2) == 0.45
        assert round(adar_neighbor_fr_resolver._calc_rel_sim(1, 2), 2) == 0.23
        round(adar_attr_resolver._calc_rel_sim(1, 2), 2) == 1.00
        round(adar_attr_fr_resolver._calc_rel_sim(1, 2), 2) == 1.00
        cleanup()
