import os
import json
from entity_resolver.scripts import run


class TestScripts:

    def test_citeseer(self):
        file_path = 'data/citeseer/citeseer-mrdm05.dat'
        graph_path = 'testgraph.json'
        ground_truth_path = 'testtruth.json'
        run([
            'norm-citeseer', file_path,
            graph_path, ground_truth_path
        ])
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        assert graph[0]['attr_dict']['name'] == 'aamodt_a'
        assert graph[0]['node_id'] == 0
        assert graph[0]['edge_id'] == 1019
        assert ground_truth[0]['node_id'] == 0
        assert ground_truth[0]['cluster_id'] == 10
        os.remove(graph_path)
        os.remove(ground_truth_path)

    def test_arxiv(self):
        file_path = 'data/arxiv/arxiv-mrdm05.dat'
        graph_path = 'testgraph.json'
        ground_truth_path = 'testtruth.json'
        run([
            'norm-citeseer', file_path,
            graph_path, ground_truth_path
        ])
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        assert graph[0]['attr_dict']['name'] == 'itzykson_c_'
        assert graph[0]['node_id'] == 0
        assert graph[0]['edge_id'] == 2
        assert ground_truth[0]['node_id'] == 0
        assert ground_truth[0]['cluster_id'] == 34481
        os.remove(graph_path)
        os.remove(ground_truth_path)
