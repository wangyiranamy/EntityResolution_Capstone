import os
from entity_resolver.parser import GraphParser, GroundTruthParser
from entity_resolver.scripts import run


class TestParser:

    def test_parser(self):
        file_path = 'data/citeseer/citeseer-mrdm05.dat'
        graph_output_path = 'testdata.json'
        ground_truth_output_path = 'testtruth.json'
        run([
            'prep-citeseer',
            '--data', file_path,
            '--graph', graph_output_path,
            '--ground_truth', ground_truth_output_path
        ])
        graph_parser = GraphParser({'name': 'person_entity'})
        graph = graph_parser.parse(graph_output_path)
        ground_truth_parser = GroundTruthParser()
        ground_truth = ground_truth_parser.parse(ground_truth_output_path)
        nodes_ambiguity = graph.get_ambiguity_adar(
            self.get_first_attr, False,
            self.get_second_attr, False
        )
        assert len(graph.get_neighbors(graph.nodes[0])) == 2
        assert len(graph.nodes) == 2888
        assert len(graph.edges) == 1504
        assert len(graph.edges[0].nodes) == 2
        assert list(graph.attr_vals.keys()) == ['name']
        assert len(ground_truth) == 2888
        assert ground_truth[0] == 10
        assert len(nodes_ambiguity.keys()) == len(ground_truth)
        assert 0 <= list(nodes_ambiguity.values())[0] <= 1
        os.remove(graph_output_path)
        os.remove(ground_truth_output_path)

    def get_first_attr(self, node_attr):
        return node_attr['name'][0][0]

    def get_second_attr(self, node_attr):
        return node_attr['name'][1]
