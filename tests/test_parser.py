import os
from entity_resolver.parser import GraphParser, GroundTruthParser
from entity_resolver.scripts import run


class TestGraphParser:

    def test_graph_parser(self):
        file_path = 'data/citeseer/citeseer-mrdm05.dat'
        output_path = 'testdata.json'
        run(['norm-citeseer', file_path, output_path])
        graph_parser = GraphParser({'title': 'text', 'name': 'person_entity'})
        graph = graph_parser.parse(output_path)
        assert len(graph.get_neighbors(graph.nodes[0])) == 2
        assert len(graph.nodes) == 2884
        assert len(graph.edges) == 1499
        assert len(graph.edges[0].nodes) == 2
        assert graph.nodes[2].get_attr('title') == [
            'knowledge', 'intensive', 'case', 'based', 'reasoning', 'learning'
        ]
        assert graph.get_attr_names() == ['title', 'name']
        assert graph.get_ambiguity_adar()['name']['mahadevan s'] == 18
        os.remove(output_path)


class TestGroundTruthParser:

    def test_dummy(self):
        ground_truth_parser = GroundTruthParser()
        assert 'parse' in dir(ground_truth_parser)
