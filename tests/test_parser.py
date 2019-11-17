import os
from entity_resolver.parser import GraphParser, GroundTruthParser
from entity_resolver.scripts import run


class TestParser:

    def test_parser(self):
        file_path = 'data/citeseer/citeseer-mrdm05.dat'
        graph_output_path = 'testdata.json'
        ground_truth_output_path = 'testtruth.json'
        run([
            'norm-citeseer', file_path,
            graph_output_path, ground_truth_output_path
        ])
        graph_parser = GraphParser({'name': 'person_entity'})
        graph = graph_parser.parse(graph_output_path)
        ground_truth_parser = GroundTruthParser()
        ground_truth = ground_truth_parser.parse(ground_truth_output_path)
        assert len(graph.get_neighbors(graph.nodes[0])) == 2
        assert len(graph.nodes) == 2888
        assert len(graph.edges) == 1504
        assert len(graph.edges[0].nodes) == 2
        assert list(graph.attr_vals.keys()) == ['name']
        assert len(ground_truth) == 2888
        assert ground_truth[0] == 10
        os.remove(graph_output_path)
        os.remove(ground_truth_output_path)
